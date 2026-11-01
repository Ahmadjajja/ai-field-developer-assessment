"""Unit tests for the streaming redactor, SSE parsing and per-choice redaction."""

import asyncio
import random
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_gateway import ChoiceRedaction, sse_events  # noqa: E402
from redactor import MAX_HOLD, REDACTED, StreamRedactor, luhn_valid, pii_spans, redact  # noqa: E402


def stream(text: str, sizes) -> str:
    r = StreamRedactor()
    out, i = "", 0
    for n in sizes:
        if i >= len(text):
            break
        out += r.feed(text[i:i + n])
        i += n
    out += r.feed(text[i:])
    return out + r.flush()


# --- What counts as PII -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "jane.doe@example.com",
        "first_last+tag@sub.example.co.uk",
        "A.B-C%D@x-y.io",
        "4111 1111 1111 1111",  # Visa test number, spaced
        "4111-1111-1111-1111",
        "4111111111111111",
        "5500 0000 0000 0004",  # Mastercard
        "3782 822463 10005",  # Amex, 15 digits
        "123-45-6789",
        "123 45 6789",
    ],
)
def test_pii_is_redacted(text):
    assert redact(f"see {text} now") == f"see {REDACTED} now"


@pytest.mark.parametrize(
    "text",
    [
        "1234567890123",  # 13 digits, fails Luhn: an order number, not a card
        "4111 1111 1111 1112",  # one digit off: fails Luhn
        "2026-01-15",
        "555-123-4567",  # phone number
        "000-12-3456",  # area 000 is never issued
        "666-12-3456",
        "900-12-3456",
        "123-00-6789",
        "123-45-0000",
        "123-45 6789",  # mixed separators
        "user at example dot com",
        "@handle",
        "not-an-email@localhost",
    ],
)
def test_lookalikes_are_left_alone(text):
    assert redact(f"see {text} now") == f"see {text} now"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # A longer digit window that fails Luhn must not hide the real card inside it.
        ("Card 4111 1111 1111 1111 9 items", f"Card {REDACTED} 9 items"),
        ("Card 4111 1111 1111 1111 12 left", f"Card {REDACTED} 12 left"),
        ("Pay with 4111111111111111 2 times", f"Pay with {REDACTED} 2 times"),
    ],
)
def test_card_followed_by_more_digits(text, expected):
    assert redact(text) == expected
    assert stream(text, [3] * len(text)) == expected


def test_luhn():
    assert luhn_valid("4111111111111111")
    assert not luhn_valid("4111111111111112")


# --- Streaming = whole-text redaction, however the text is chunked ----------------------------------


SAMPLES = [
    "Contact me at john.doe@gmail.com today.",
    "Card 4111 1111 1111 1111 exp 12/27, also 4111-1111-1111-1111 and 4111111111111111.",
    "SSN 123-45-6789 or 123 45 6789; not 000-12-3456 nor 2024-01-15.",
    "Email:a.b+c@sub.example.co.uk,next first_last@x.io.",
]


@pytest.mark.parametrize("text", SAMPLES)
def test_every_two_chunk_split_gives_the_same_result(text):
    expected = redact(text)
    for cut in range(len(text) + 1):
        assert stream(text, [cut]) == expected, cut


@pytest.mark.parametrize("text", SAMPLES)
def test_one_character_at_a_time(text):
    assert stream(text, [1] * len(text)) == redact(text)


@pytest.mark.parametrize("seed", range(5))
def test_random_adversarial_streams_match_whole_text_redaction(seed):
    rng = random.Random(seed)
    pieces = ["a", "b.", "@", "x.com", "-", "4111 1111 1111 1111", "123-45-6789", " ", "1", "9", "co",
              "\n", "_", "+", "first_last@ex.io", "5500-0000-0000-0004", "0", "--", "  "]
    for _ in range(2000):
        text = "".join(rng.choice(pieces) for _ in range(rng.randint(1, 30)))
        sizes = [rng.randint(1, 7) for _ in range(len(text))]
        assert stream(text, sizes) == redact(text), text


# --- Latency and memory ------------------------------------------------------------------------------


def test_only_the_trailing_partial_word_is_held():
    r = StreamRedactor()
    assert r.feed("Hello, my name is Ja") == "Hello, my name is "
    assert r.held == 2
    assert r.feed("ne and I like tea. ") == "Jane and I like tea. "
    assert r.held == 0


def test_text_is_released_as_soon_as_pii_is_ruled_out():
    r = StreamRedactor()
    assert r.feed("mail jane.doe@exa") == "mail "
    assert r.feed("mple.com, thanks") == f"{REDACTED}, "
    assert r.flush() == "thanks"


def test_memory_stays_bounded_on_a_hostile_stream():
    r = StreamRedactor()
    longest = 0
    for _ in range(200_000):  # 200k chars with no break at all
        r.feed("a")
        longest = max(longest, r.held)
    assert longest <= 2 * MAX_HOLD


def test_held_text_stays_small_for_normal_prose():
    prose = ("The quick brown fox emailed jane@example.com about card 4111 1111 1111 1111. " * 2000)
    r, longest = StreamRedactor(), 0
    for i in range(0, len(prose), 4):
        r.feed(prose[i:i + 4])
        longest = max(longest, r.held)
    # The card plus one character: a match is never released until the next character proves it's complete.
    assert longest <= len("4111 1111 1111 1111") + 1


def test_throughput():
    prose = "Lorem ipsum dolor sit amet, contact a.b@example.com or call 555-123-4567. " * 13_000  # ~1 MB
    r = StreamRedactor()
    start = time.perf_counter()
    for i in range(0, len(prose), 16):
        r.feed(prose[i:i + 16])
    r.flush()
    assert time.perf_counter() - start < 5  # typically well under a second


@pytest.mark.parametrize(
    ("label", "unit"),
    [
        ("letters, no break", "abcdefghij"),
        ("digits, no break", "1234567890"),
        ("email characters, no @", "a.b-c_d+e%"),
        ("many @", "a@b@c@d@e@"),
        ("digit-dash", "1-2-3-4-5-"),
        ("digit-space: overlapping card-like matches everywhere", "1 2 3 4 5 "),
    ],
)
def test_hostile_streams_stay_fast(label, unit):
    # 100 KB in 4-character chunks (about one token each). Before caching settled
    # card checks, the digit-space case took ~20 s for this; now well under 2 s.
    data = (unit * 10_000)[:100_000]
    r = StreamRedactor()
    start = time.perf_counter()
    for i in range(0, len(data), 4):
        r.feed(data[i:i + 4])
    r.flush()
    assert time.perf_counter() - start < 3, label


def test_streamed_output_never_contains_pii_even_for_hostile_runs():
    # Long runs hit the hold caps; even then nothing detectable may get through.
    rng = random.Random(99)
    pieces = ["4111 1111 1111 1111", "123-45-6789", "first_last@ex.io", " ", "a", "1", "-", "@", "x.com",
              "1 2 3 4 5 6 7 8 9 0 " * 5, "a" * 70, "7" * 40, "1-" * 25]
    for _ in range(1500):
        text = "".join(rng.choice(pieces) for _ in range(rng.randint(1, 20)))
        out = stream(text, [rng.randint(1, 9) for _ in range(len(text))])
        assert pii_spans(out.replace(REDACTED, "\x00")) == [], text


@pytest.mark.parametrize("seed", range(3))
def test_forced_cut_in_an_endless_chain_over_redacts_instead_of_leaking(seed):
    # Random digits with spaces: card-like matches overlap all the way, so the
    # held text hits the 512-character limit and must be cut inside a match.
    rng = random.Random(seed)
    text = " ".join(rng.choice("0123456789") for _ in range(1500))
    r = StreamRedactor()
    out = "".join(r.feed(text[i:i + 4]) for i in range(0, len(text), 4)) + r.flush()
    assert pii_spans(out.replace(REDACTED, "\x00")) == []


def test_forced_cut_through_a_real_card_redacts_its_continuation():
    # Back-to-back real cards chain into one long match; the forced cut lands
    # inside a card, and no digit of any card may come out unredacted.
    text = "4111 1111 1111 1111 " * 40
    r = StreamRedactor()
    out = "".join(r.feed(text[i:i + 3]) for i in range(0, len(text), 3)) + r.flush()
    assert not any(ch.isdigit() for ch in out)


def test_email_local_part_and_length_limits():
    local = "a" * 70
    assert redact(f"{local}@example.com") == "a" * 6 + REDACTED  # only the last 64 can be the local part
    too_long = "a@" + "b" * 260 + ".com"
    assert redact(too_long) == too_long  # over 254 characters: not a valid address


# --- SSE parsing ------------------------------------------------------------------------------------------


def parse(chunks: list[bytes]) -> list[str]:
    async def gen():
        for c in chunks:
            yield c

    async def collect():
        return [e async for e in sse_events(gen())]

    return asyncio.run(collect())


def test_sse_events_split_anywhere():
    raw = 'data: {"a": 1}\n\ndata: {"b": "é\U0001F600"}\r\n\r\n: comment\n\ndata: [DONE]\n\n'.encode()
    expected = ['{"a": 1}', '{"b": "é\U0001F600"}', "[DONE]"]
    for cut in range(len(raw)):
        assert parse([raw[:cut], raw[cut:]]) == expected
    assert parse([bytes([b]) for b in raw]) == expected  # one byte at a time: emoji split mid-character


def test_sse_multiline_data_and_missing_final_blank_line():
    assert parse([b"data: line1\ndata: line2\n\ndata: tail"]) == ["line1\nline2", "tail"]


# --- Several choices (n > 1) ------------------------------------------------------------------------------


def test_each_choice_is_redacted_separately():
    redaction = ChoiceRedaction()
    out = {0: "", 1: ""}
    for index, piece in [(0, "mail jane@exa"), (1, "card 4111 1111 "), (0, "mple.com ok"), (1, "1111 1111 done")]:
        chunk = {"id": "x", "choices": [{"index": index, "delta": {"content": piece}, "finish_reason": None}]}
        redaction.apply(chunk)
        out[index] += chunk["choices"][0]["delta"]["content"]
    for event in redaction.flush_events():
        choice = event["choices"][0]
        out[choice["index"]] += choice["delta"]["content"]
    assert out == {0: f"mail {REDACTED} ok", 1: f"card {REDACTED} done"}
