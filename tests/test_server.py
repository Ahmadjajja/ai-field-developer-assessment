"""End-to-end tests: the server runs as a real child process over stdio."""

import json
import select
import threading
import time

import pytest

from mcp_client import INITIALIZE, INITIALIZED, Client, run_session, start, tool_call

INVALID_PARAMS = -32602
REASON = "Duplicate charge on order 5512"


@pytest.fixture(scope="module")
def client():
    c = Client()
    yield c
    c.close()


def error_fields(response: dict) -> list[str]:
    assert "result" not in response, f"expected a JSON-RPC error, got {response}"
    assert response["error"]["code"] == INVALID_PARAMS
    return [e["field"] for e in response["error"]["data"]]


def tool_result(response: dict) -> dict:
    assert "error" not in response, f"expected a result, got {response}"
    return response["result"]


# --- Protocol ----------------------------------------------------------------


def test_lists_both_tools_with_schemas_and_annotations(client):
    tools = {t["name"]: t for t in client.request("tools/list")["result"]["tools"]}

    assert set(tools) == {"get_customer_record", "trigger_refund"}
    assert tools["get_customer_record"]["inputSchema"]["properties"]["customer_id"]["pattern"] == "^CUST-[0-9]{5}$"
    assert tools["trigger_refund"]["inputSchema"]["required"] == ["customer_id", "amount", "reason"]
    assert tools["get_customer_record"]["annotations"]["readOnlyHint"] is True
    assert tools["trigger_refund"]["annotations"]["destructiveHint"] is True


def test_unknown_tool_is_invalid_params(client):
    response = client.call("delete_everything", {})
    assert response["error"]["code"] == INVALID_PARAMS
    assert "Unknown tool" in response["error"]["message"]


def test_unknown_method_is_method_not_found(client):
    assert client.request("tools/explode")["error"]["code"] == -32601


# --- get_customer_record -----------------------------------------------------


def test_get_customer_record_valid(client):
    result = tool_result(client.call("get_customer_record", {"customer_id": "CUST-10003"}))
    record = json.loads(result["content"][0]["text"])

    assert result["isError"] is False
    assert record["name"] == "Li Wei"
    assert record["refundable_balance"] == 1200.0


@pytest.mark.parametrize(
    "customer_id",
    [
        "abc",
        "cust-10001",  # wrong case
        "CUST-1234",  # too short
        "CUST-123456",  # too long
        "CUST-1234a",  # letter
        " CUST-10001",  # leading space
        "CUST-10001\n",  # trailing newline
        "CUST-١٢٣٤٥",  # non-ASCII digits
        "",
        12345,  # not a string
        None,
        ["CUST-10001"],
    ],
)
def test_get_customer_record_rejects_bad_ids(client, customer_id):
    assert error_fields(client.call("get_customer_record", {"customer_id": customer_id})) == ["customer_id"]


def test_get_customer_record_missing_field(client):
    assert error_fields(client.call("get_customer_record", {})) == ["customer_id"]


def test_get_customer_record_extra_field(client):
    assert error_fields(client.call("get_customer_record", {"customer_id": "CUST-10001", "admin": True})) == ["admin"]


def test_any_valid_customer_id_gets_a_mock_record(client):
    result = tool_result(client.call("get_customer_record", {"customer_id": "CUST-54321"}))
    record = json.loads(result["content"][0]["text"])

    assert result["isError"] is False
    assert record["customer_id"] == "CUST-54321"
    assert record["refundable_balance"] == 500.0


# --- trigger_refund ----------------------------------------------------------


def test_trigger_refund_valid_reduces_balance(client):
    result = tool_result(client.call("trigger_refund", {"customer_id": "CUST-10003", "amount": 100.5, "reason": REASON}))
    refund = json.loads(result["content"][0]["text"])
    assert refund["status"] == "processed"
    assert refund["refund_id"].startswith("RF-")

    record = json.loads(tool_result(client.call("get_customer_record", {"customer_id": "CUST-10003"}))["content"][0]["text"])
    assert record["refundable_balance"] == 1099.5


@pytest.mark.parametrize("amount", [1, 0.01, 99.99, 1e2])
def test_trigger_refund_accepts_positive_amounts(client, amount):
    result = tool_result(client.call("trigger_refund", {"customer_id": "CUST-10003", "amount": amount, "reason": REASON}))
    assert result["isError"] is False


@pytest.mark.parametrize(
    ("amount", "message"),
    [
        (0.001, "at least 0.01"),
        (5e-324, "at least 0.01"),
        (10.555, "whole cents"),
    ],
)
def test_sub_cent_amounts_are_valid_input_but_refused(client, amount, message):
    # The spec only says "positive float", so these pass the schema (no -32602)
    # and are refused as a business rule instead.
    result = tool_result(client.call("trigger_refund", {"customer_id": "CUST-10003", "amount": amount, "reason": REASON}))
    assert result["isError"] is True
    assert message in result["content"][0]["text"]


def test_money_math_is_exact(client):
    for _ in range(3):
        tool_result(client.call("trigger_refund", {"customer_id": "CUST-60001", "amount": 0.1, "reason": REASON}))
    record = json.loads(tool_result(client.call("get_customer_record", {"customer_id": "CUST-60001"}))["content"][0]["text"])

    assert record["total_refunded"] == 0.3  # not 0.30000000000000004
    assert record["refundable_balance"] == 499.7


def test_trigger_refund_works_for_any_valid_customer_id(client):
    result = tool_result(client.call("trigger_refund", {"customer_id": "CUST-12345", "amount": 50.0, "reason": REASON}))
    assert result["isError"] is False


@pytest.mark.parametrize(
    "amount",
    [0, -5, -0.01, "40", True, None, [40], {"value": 40}, float("nan"), float("inf"), float("-inf")],
)
def test_trigger_refund_rejects_bad_amounts(client, amount):
    assert error_fields(client.call("trigger_refund", {"customer_id": "CUST-10003", "amount": amount, "reason": REASON})) == ["amount"]


def test_trigger_refund_huge_amount_is_tool_error_not_protocol_error(client):
    result = tool_result(client.call("trigger_refund", {"customer_id": "CUST-10003", "amount": 1e308, "reason": REASON}))
    assert result["isError"] is True
    assert "exceeds refundable balance" in result["content"][0]["text"]


@pytest.mark.parametrize(
    "reason",
    [
        "",
        "short",
        "123456789",
        "          ",
        "   short   ",
        "\n\t short \n\t",
        "​" * 12,  # zero-width spaces: invisible, so they don't count
        "short​​​​​​",  # 5 visible + 6 invisible
        "dup\x00charge order",  # NUL
        "duplicate\x07 charge",  # BEL
        1234567890,
        None,
    ],
)
def test_trigger_refund_rejects_bad_reasons(client, reason):
    assert error_fields(client.call("trigger_refund", {"customer_id": "CUST-10003", "amount": 1, "reason": reason})) == ["reason"]


@pytest.mark.parametrize("reason", ["طلب استرداد للرسوم المكررة", "💸" * 10, "Duplicate charge,\nsee ticket 42"])
def test_trigger_refund_accepts_unicode_and_newlines_in_reason(client, reason):
    result = tool_result(client.call("trigger_refund", {"customer_id": "CUST-10003", "amount": 1, "reason": reason}))
    assert result["isError"] is False


def test_oversized_reason_is_refused_and_not_echoed(client):
    response = client.call("trigger_refund", {"customer_id": "CUST-10003", "amount": 1, "reason": "x" * 1_000_000})
    result = tool_result(response)
    assert result["isError"] is True
    assert "at most 1000 characters" in result["content"][0]["text"]
    assert len(json.dumps(response)) < 1000


def test_reason_code_is_optional_enum_in_schema(client):
    tools = {t["name"]: t for t in client.request("tools/list")["result"]["tools"]}
    schema = tools["trigger_refund"]["inputSchema"]

    assert "reason_code" not in schema["required"]
    assert schema["properties"]["reason_code"]["enum"] == [
        "duplicate_charge", "billing_error", "service_outage", "cancellation", "goodwill",
    ]


def test_trigger_refund_with_reason_code(client):
    arguments = {"customer_id": "CUST-10003", "amount": 1, "reason": REASON, "reason_code": "duplicate_charge"}
    refund = json.loads(tool_result(client.call("trigger_refund", arguments))["content"][0]["text"])
    assert refund["reason_code"] == "duplicate_charge"


def test_trigger_refund_without_reason_code_still_works(client):
    refund = json.loads(tool_result(client.call("trigger_refund", {"customer_id": "CUST-10003", "amount": 1, "reason": REASON}))["content"][0]["text"])
    assert refund["reason_code"] is None


@pytest.mark.parametrize("reason_code", ["free_money", "DUPLICATE_CHARGE", "", None, 1])
def test_trigger_refund_rejects_bad_reason_codes(client, reason_code):
    arguments = {"customer_id": "CUST-10003", "amount": 1, "reason": REASON, "reason_code": reason_code}
    assert error_fields(client.call("trigger_refund", arguments)) == ["reason_code"]


def test_trigger_refund_reports_every_bad_field_at_once(client):
    fields = error_fields(client.call("trigger_refund", {"customer_id": "bad", "amount": -1, "reason": "x"}))
    assert sorted(fields) == ["amount", "customer_id", "reason"]


def test_trigger_refund_missing_fields(client):
    assert sorted(error_fields(client.call("trigger_refund", {}))) == ["amount", "customer_id", "reason"]


def test_trigger_refund_error_does_not_echo_rejected_values(client):
    secret = "SECRET-VALUE-123"
    response = client.call("trigger_refund", {"customer_id": secret, "amount": 1, "reason": REASON})
    assert secret not in json.dumps(response)


def test_trigger_refund_over_balance_is_tool_error(client):
    # CUST-10002 paid 60 and already got 20 back, so 40 is left.
    result = tool_result(client.call("trigger_refund", {"customer_id": "CUST-10002", "amount": 40.01, "reason": REASON}))
    assert result["isError"] is True
    assert "exceeds refundable balance" in result["content"][0]["text"]


@pytest.mark.parametrize(
    ("line", "code"),
    [
        ("this is not json", -32700),
        ('{"jsonrpc":"2.0","id":5', -32700),  # truncated JSON
        ('{"jsonrpc":"2.0","id":5}', -32600),  # no method
        ('{"id":5,"method":"tools/list"}', -32600),  # no jsonrpc version
        ("[1, 2]", -32600),
    ],
)
def test_malformed_messages_get_jsonrpc_errors_and_session_continues(line, code):
    stdout, _ = run_session([line, tool_call(1, "get_customer_record", {"customer_id": "CUST-10001"})])
    messages = assert_pure_jsonrpc(stdout)

    error = next(m for m in messages if m.get("id") is None)
    assert error["error"]["code"] == code
    assert any(m.get("id") == 1 and m["result"]["isError"] is False for m in messages)


def test_blank_lines_are_ignored_quietly():
    stdout, _ = run_session(["", "   ", "\t", tool_call(1, "get_customer_record", {"customer_id": "CUST-10001"})], null_id_replies=0)
    messages = assert_pure_jsonrpc(stdout)

    assert [m["id"] for m in messages] == [0, 1]  # handshake + the call, no error replies


def test_only_tools_capability_is_advertised():
    stdout, _ = run_session([])
    capabilities = json.loads(stdout[0])["result"]["capabilities"]
    assert "tools" in capabilities
    assert "prompts" not in capabilities
    assert "resources" not in capabilities


# --- stdio isolation ---------------------------------------------------------


def assert_pure_jsonrpc(stdout_lines: list[str]) -> list[dict]:
    assert stdout_lines, "server wrote nothing to stdout"
    messages = []
    for line in stdout_lines:
        message = json.loads(line)  # fails on any stray non-JSON line
        assert message["jsonrpc"] == "2.0"
        messages.append(message)
    return messages


def test_stdout_is_pure_jsonrpc_and_logs_go_to_stderr():
    stdout, stderr = run_session([
        tool_call(1, "get_customer_record", {"customer_id": "CUST-10001"}),
        tool_call(2, "trigger_refund", {"customer_id": "CUST-10001", "amount": 5, "reason": REASON}),
        tool_call(3, "trigger_refund", {"customer_id": "bad", "amount": -1, "reason": "x"}),
    ])

    messages = assert_pure_jsonrpc(stdout)
    assert sorted(m["id"] for m in messages) == [0, 1, 2, 3]

    logs = [json.loads(line) for line in stderr.splitlines()]
    assert {"server starting", "tool call", "refund processed"} <= {entry["msg"] for entry in logs}


def test_tool_crash_returns_generic_error_and_logs_details_to_stderr():
    stdout, stderr = run_session([tool_call(1, "crashing_tool", {})], script="tests/noisy_server.py")

    response = next(m for m in assert_pure_jsonrpc(stdout) if m.get("id") == 1)
    assert response["result"]["isError"] is True
    assert "hunter2" not in json.dumps(response)
    assert "hunter2" in stderr  # the traceback is kept for operators, on stderr only
    assert '"outcome": "crash"' in stderr


def test_concurrent_refunds_never_overspend_the_balance():
    # 20 refunds of 100 at once on a 500 balance, with a slow payment provider.
    calls = [tool_call(i, "trigger_refund", {"customer_id": "CUST-70001", "amount": 100, "reason": REASON}) for i in range(1, 21)]
    stdout, _ = run_session(calls, script="tests/slow_server.py")
    results = [m["result"] for m in assert_pure_jsonrpc(stdout) if m.get("id", 0) >= 1]

    assert len(results) == 20
    assert sum(not r["isError"] for r in results) == 5
    assert sum("exceeds refundable balance" in r["content"][0]["text"] for r in results if r["isError"]) == 15


def test_server_keeps_answering_when_host_never_reads_stderr():
    # If the host ignores stderr, the pipe fills up (~64 KB). Logging must not freeze the server.
    proc = start()  # stderr is a pipe that this test never reads
    n = 1500
    lines = [INITIALIZE, INITIALIZED] + [tool_call(i, "get_customer_record", {"customer_id": "CUST-10001"}) for i in range(1, n + 1)]
    writer = threading.Thread(target=lambda: (proc.stdin.write("".join(json.dumps(m) + "\n" for m in lines)), proc.stdin.flush()), daemon=True)
    writer.start()

    replies, deadline = 0, time.monotonic() + 20
    while replies < n + 1 and time.monotonic() < deadline:
        ready, _, _ = select.select([proc.stdout], [], [], 1)
        if ready and proc.stdout.readline():
            replies += 1
    proc.kill()
    proc.wait()
    for pipe in (proc.stdin, proc.stdout, proc.stderr):
        pipe.close()

    assert replies == n + 1


def test_stray_output_from_tool_code_never_reaches_stdout():
    stdout, stderr = run_session([tool_call(1, "noisy_tool", {})], script="tests/noisy_server.py")

    messages = assert_pure_jsonrpc(stdout)
    assert any(m.get("id") == 1 and m["result"]["isError"] is False for m in messages)
    for noise in [
        "stray print at startup",
        "stray print()",
        "stray sys.stdout.write without newline",
        "stray os.write to fd 1",
        "stray output from a child process",
    ]:
        assert noise in stderr
