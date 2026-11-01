"""Streaming PII redaction.

The problem: an LLM response arrives in small chunks, and PII can be split
across them ("john.d" + "oe@gmail" + ".com"). Checking each chunk alone misses
it; waiting for the whole response kills time-to-first-token and keeps the
full text in memory.

The approach: every PII pattern here is a run of "candidate" characters
without ordinary spaces (emails), or of only digits, spaces and dashes (card
numbers, SSNs). So only the *trailing run* of such characters can still turn
into PII when more text arrives. StreamRedactor emits everything before that
run right away (redacted) and holds back just the run.

    "Contact me at john.d"   -> emits "Contact me at ", holds "john.d"
    "oe@gmail.com today"     -> emits "[REDACTED] ",    holds "today"

It also never cuts inside a PII match, so the streamed result is the same as
redacting the complete text in one go, however the stream is chunked.
Usually only one partial word is held, so latency stays at about one token and
memory stays constant, whatever the response length.

The held run is capped by the real size limits of each kind of PII (a card is
at most 37 characters, an email local part at most 64), and settled card
checks are cached, so even hostile input (no breaks, card-like digits
everywhere) stays fast and small.
"""

import re

REDACTED = "[REDACTED]"

# Lengths follow the standards: local part (before @) at most 64 characters
# (RFC 5321), domain labels at most 63 (DNS). The bounds also matter for
# safety: an unbounded domain could swallow a real address and the text after
# it into one absurd 300-character "address", which is then rejected as too
# long, leaving the real address unredacted.
EMAIL = re.compile(r"[A-Za-z0-9._%+_-]{1,64}@(?:[A-Za-z0-9-]{1,63}\.){1,8}[A-Za-z]{2,24}")

# US SSN: AAA-GG-SSSS (dashes or spaces, used consistently). Numbers that can
# never be SSNs (area 000/666/9xx, group 00, serial 0000) are skipped to cut false positives.
SSN = re.compile(r"(?<![\d-])(?!000|666|9\d\d)\d{3}([- ])(?!00)\d{2}\1(?!0000)\d{4}(?![\d-])")

# Cards: 13-19 digits, optionally grouped with single spaces or dashes, that
# pass the Luhn checksum (real card numbers do). card_spans() checks every
# start position and every length, so neither an overlapping candidate nor a
# longer one that fails Luhn ("4111 1111 1111 1111 9 items") can hide a card.
CARD_START = re.compile(r"(?<![\d-])[0-9]")
DIGITS = "0123456789"

# Characters that can be part of an email / a card or SSN.
EMAIL_TAIL = re.compile(r"[A-Za-z0-9._%+_@-]*\Z")
NUMBER_TAIL = re.compile(r"[0-9 -]*\Z")

# How much of the trailing run is held, at most. These match the longest PII
# that can still form, so holding more would never change the result; they also
# keep the work per chunk small, even for a hostile stream with no breaks.
MAX_NUMBER_HOLD = 40  # longest card: 19 digits + 18 separators = 37 characters
MAX_LOCAL_PART = 64  # before an @ arrives, only the last 64 characters can become an email
MAX_HOLD = 256  # an email with its @
MAX_EMAIL_LENGTH = 254  # the longest valid address


def luhn_valid(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


# A card starting at position p depends only on text[p-1 : p+38] (at most 37
# characters, plus the next one for the "not followed by a digit" check).
# Once 38 more characters have arrived, the result for p can't change.
CARD_SETTLED_AFTER = 38


def _card_at(text: str, p: int) -> tuple[int, int] | None:
    """The longest Luhn-valid card starting at p, if any."""
    digits, ends, i, n = [], [], p, len(text)
    while i < n and text[i] in DIGITS and len(digits) < 19:
        digits.append(text[i])
        i += 1
        ends.append(i)
        if i + 1 < n and text[i] in " -" and text[i + 1] in DIGITS:
            i += 1  # one separator between two digits
    for count in range(len(digits), 12, -1):  # 19 digits down to 13
        end = ends[count - 1]
        if end < n and text[end] in "0123456789-":
            continue  # a card can't run straight into more digits or a dash
        if luhn_valid("".join(digits[:count])):
            return p, end
    return None


def card_spans(text: str, start: int = 0) -> list[tuple[int, int]]:
    """Luhn-valid cards starting at `start` or later (overlapping ones included)."""
    found = (_card_at(text, m.start()) for m in CARD_START.finditer(text, start))
    return [span for span in found if span]


def pii_spans(text: str, start: int = 0, cards: list[tuple[int, int]] | None = None) -> list[tuple[int, int]]:
    """Start/end of every PII match beginning at `start` or later, overlaps merged, in order.

    text[:start] is only context for the patterns' look-behind checks. Matches
    starting inside it are dropped *before* merging, so they can't swallow
    real matches that overlap them. `cards` passes in card matches that were
    already found (StreamRedactor caches them).
    """
    # Addresses longer than 254 characters aren't valid (RFC 5321), and the
    # stream never holds that much, so they don't count on either path.
    spans = [m.span() for m in EMAIL.finditer(text, start) if m.end() - m.start() <= MAX_EMAIL_LENGTH]
    spans += [m.span() for m in SSN.finditer(text, start)]
    spans += card_spans(text, start) if cards is None else cards
    return _merge(spans)


def _merge(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def _render(text: str, spans: list[tuple[int, int]]) -> str:
    parts, pos = [], 0
    for start, end in spans:
        parts += [text[pos:start], REDACTED]
        pos = end
    parts.append(text[pos:])
    return "".join(parts)


def redact(text: str) -> str:
    """Redact all PII in a complete piece of text."""
    return _render(text, pii_spans(text))


def _trailing_run_start(text: str) -> int:
    """Where the part of the trailing run that could still become PII begins."""
    email_run = EMAIL_TAIL.search(text).group()
    at = email_run.find("@")
    if at == -1:
        email_hold = min(len(email_run), MAX_LOCAL_PART)
    else:  # keep up to 64 characters before the first @, and the rest of the run
        email_hold = min(len(email_run) - max(0, at - MAX_LOCAL_PART), MAX_HOLD)
    number_run = NUMBER_TAIL.search(text).group()
    # Only spaces/dashes can't start PII by themselves, so they need no holding.
    number_hold = min(len(number_run), MAX_NUMBER_HOLD) if any(ch.isdigit() for ch in number_run) else 0
    return len(text) - max(email_hold, number_hold)


class StreamRedactor:
    """Feed chunks in, get redacted text out, with only a short tail held back."""

    def __init__(self) -> None:
        self._held = ""
        # The last character already sent. When a long run is cut in the middle,
        # the patterns' "not preceded by a digit or dash" checks must still see it.
        self._before = ""
        # After a forced cut through a match: how many held characters still
        # belong to that match and must be redacted whatever comes next.
        self._redact_first = 0
        # Card checks are the expensive part (a Luhn sum per candidate), so a
        # candidate's result is kept once it is settled instead of being redone
        # on every chunk. Positions here are absolute (from the start of the stream).
        self._offset = 0  # absolute position of the current buffer's first character
        self._cards: list[tuple[int, int]] = []  # settled Luhn-valid matches
        self._settled = 0  # candidates starting before this position are settled

    def _spans(self, text: str, ctx: int) -> list[tuple[int, int]]:
        base = self._offset
        scan_from = max(self._settled, base + ctx)
        fresh = [(s + base, e + base) for s, e in card_spans(text, scan_from - base)]
        settled_until = base + len(text) - CARD_SETTLED_AFTER
        self._cards = [c for c in self._cards if c[0] >= base + ctx] + [c for c in fresh if c[0] < settled_until]
        self._settled = max(self._settled, settled_until)
        unsettled = [c for c in fresh if c[0] >= settled_until]
        cards = [(s - base, e - base) for s, e in self._cards + unsettled]

        spans = pii_spans(text, ctx, cards)  # the context character only informs look-behind checks
        if self._redact_first:
            spans = _merge([*spans, (ctx, ctx + self._redact_first)])
        return spans

    def feed(self, chunk: str) -> str:
        """Add a chunk; return the text that is now safe to send (already redacted)."""
        ctx = len(self._before)
        text = self._before + self._held + chunk
        spans = self._spans(text, ctx)
        cut = max(_trailing_run_start(text), ctx)
        # Never cut inside a match, or right at its end (more text could still extend it).
        for start, end in reversed(spans):
            if start < cut <= end:
                cut = start

        # Hostile input (e.g. "1 2 3 4 ..." where overlapping card-like matches
        # chain on and on) could make the held text grow without bound. Past a
        # limit, cut anyway, and redact the part of any match that crosses the
        # cut plus its continuation: over-redact rather than leak.
        forced = len(text) - cut > 2 * MAX_HOLD
        if forced:
            cut = len(text) - MAX_HOLD
        crossing = [(s, e) for s, e in spans if s < cut < e] if forced else []
        # Still to redact at the start of the held text: what's left of an earlier
        # forced cut's continuation, plus whatever this cut runs through.
        still_pending = ctx + self._redact_first - cut
        self._redact_first = max(still_pending, *(e - cut for _, e in crossing), 0)

        self._held = text[cut:]
        if cut > ctx:
            self._before = text[cut - 1]
            self._offset += cut - 1
        emitted = [(s, min(e, cut)) for s, e in spans if s < cut and (e <= cut or forced)]
        return _render(text[:cut], emitted)[ctx:]

    def flush(self) -> str:
        """End of stream: redact and return whatever is still held."""
        ctx = len(self._before)
        text = self._before + self._held
        spans = self._spans(text, ctx)
        self.__init__()
        return _render(text, spans)[ctx:]

    @property
    def held(self) -> int:
        """Characters currently held back (for tests and metrics)."""
        return len(self._held)
