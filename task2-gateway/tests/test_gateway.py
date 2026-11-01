"""End-to-end tests: client -> gateway process -> mock MCP server process."""

import gzip
import json
import time

import httpx
import jwt
import pytest

import auth
from conftest import McpClient, body, start_stack, stop_stack, token

ADMIN_CALL = ("admin_reset_key", {"key_id": "key-1"})


@pytest.fixture
def viewer(stack):
    return McpClient(stack.gateway, token("viewer", "bob")).initialize()


@pytest.fixture
def admin(stack):
    return McpClient(stack.gateway, token("admin", "alice")).initialize()


def error_code(response: httpx.Response):
    return body(response).get("error", {}).get("code")


# --- Requirement: tools/list is forwarded transparently ---------------------------------


def test_tools_list_is_forwarded_unchanged_for_viewer(viewer):
    r = viewer.post({"jsonrpc": "2.0", "id": 5, "method": "tools/list"})
    names = [t["name"] for t in body(r)["result"]["tools"]]
    assert r.status_code == 200
    assert names == ["get_status", "list_keys", "admin_reset_key", "admin_delete_user"]


# --- Requirement: admin_ tools need the admin role, enforced with -32001 ----------------------


@pytest.mark.parametrize("tool", ["admin_reset_key", "admin_delete_user"])
def test_viewer_admin_call_is_intercepted_with_32001(stack, viewer, tool):
    r = viewer.call(tool, {"key_id": "k", "user_id": "u"}, id_=42)
    assert r.status_code == 200
    assert body(r) == {"jsonrpc": "2.0", "id": 42, "error": {"code": -32001, "message": "Unauthorized Tool Call"}}
    assert stack.executed() == []  # never reached the downstream server


def test_viewer_can_call_normal_tools(stack, viewer):
    r = viewer.call("get_status")
    assert body(r)["result"]["isError"] is False
    assert stack.executed() == ["get_status"]


def test_admin_can_call_admin_tools(stack, admin):
    r = admin.call(*ADMIN_CALL)
    assert json.loads(body(r)["result"]["content"][0]["text"])["status"] == "rotated"
    assert stack.executed() == ["admin_reset_key"]


# --- Authentication ---------------------------------------------------------------------------------


def _jwt(claims: dict, secret=auth.SECRET, alg="HS256") -> str:
    base = {"sub": "eve", "role": "admin", "aud": auth.AUDIENCE, "exp": int(time.time()) + 600}
    return jwt.encode({**base, **claims}, secret, algorithm=alg)


@pytest.mark.parametrize(
    ("authorization", "message"),
    [
        (None, "Missing Authorization header"),
        ("Basic YWxpY2U6cHc=", "Authorization header must be 'Bearer <token>'"),
        ("Bearer", "Authorization header must be 'Bearer <token>'"),
        ("Bearer not.a.jwt", "Invalid token"),
        (f"Bearer {_jwt({'exp': int(time.time()) - 5})}", "Token has expired"),
        (f"Bearer {_jwt({}, secret='attacker-secret-attacker-secret-0000')}", "Invalid token"),
        (f"Bearer {jwt.encode({'sub': 'eve', 'role': 'admin', 'aud': 'mcp-gateway', 'exp': int(time.time()) + 600}, None, algorithm='none')}", "Invalid token"),
        (f"Bearer {_jwt({'aud': 'other-service'})}", "Invalid token"),
        (f"Bearer {_jwt({'role': 'superuser'})}", "Invalid token"),
        (f"Bearer {_jwt({'role': None})}", "Invalid token"),
    ],
)
def test_bad_credentials_get_401(stack, authorization, message):
    client = McpClient(stack.gateway, None)
    headers = {"Authorization": authorization} if authorization else {}
    r = client.post({"jsonrpc": "2.0", "id": 3, "method": "tools/list"}, **headers)

    assert r.status_code == 401
    assert r.headers["www-authenticate"].startswith("Bearer")
    assert r.json() == {"jsonrpc": "2.0", "id": 3, "error": {"code": -32000, "message": message}}


def test_jwt_without_exp_is_rejected(stack):
    no_exp = jwt.encode({"sub": "eve", "role": "admin", "aud": auth.AUDIENCE}, auth.SECRET, algorithm="HS256")
    r = McpClient(stack.gateway, no_exp).post({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    assert r.status_code == 401


@pytest.mark.parametrize("method", ["GET", "DELETE"])
def test_get_and_delete_also_need_a_token(stack, method):
    r = httpx.request(method, stack.gateway, headers={"Accept": "text/event-stream"})
    assert r.status_code == 401


def test_user_token_is_not_forwarded_and_identity_headers_cannot_be_spoofed(stack, viewer):
    viewer.call("get_status", **{})
    viewer.post({"jsonrpc": "2.0", "id": 9, "method": "tools/list"}, **{"X-Gateway-Role": "admin", "X-Gateway-User": "root"})
    seen = stack.received_headers()[-1]

    assert "authorization" not in seen
    assert seen["x-gateway-user"] == "bob"
    assert seen["x-gateway-role"] == "viewer"


# --- Bypass attempts (red team) ------------------------------------------------------------------------


def _call_bytes(name: str, id_=7) -> bytes:
    return json.dumps({"jsonrpc": "2.0", "id": id_, "method": "tools/call", "params": {"name": name, "arguments": {"key_id": "k"}}}).encode()


@pytest.mark.parametrize(
    ("label", "raw", "expected_code"),
    [
        ("batch with only an admin call", b"[" + _call_bytes("admin_reset_key") + b"]", -32001),
        ("batch hiding an admin call", b"[" + _call_bytes("get_status", 1) + b"," + _call_bytes("admin_reset_key", 2) + b"]", -32001),
        ("duplicate name, admin first", b'{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"admin_reset_key","name":"get_status"}}', -32700),
        ("duplicate name, admin last", b'{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"get_status","name":"admin_reset_key"}}', -32700),
        ("duplicate method", b'{"jsonrpc":"2.0","id":7,"method":"tools/list","method":"tools/call","params":{"name":"admin_reset_key"}}', -32700),
        ("duplicate params", b'{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"x"},"params":{"name":"admin_reset_key"}}', -32700),
        ("unicode escape", b'{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"\\u0061dmin_reset_key"}}', -32001),
        ("upper case", _call_bytes("ADMIN_reset_key"), -32001),
        ("fullwidth letters", _call_bytes("ａｄｍｉｎ_reset_key"), -32001),
        ("leading space", _call_bytes(" admin_reset_key"), -32001),
        ("zero-width space inside", _call_bytes("admin​_reset_key"), -32001),
        ("method in upper case", b'{"jsonrpc":"2.0","id":7,"method":"TOOLS/CALL","params":{"name":"admin_reset_key"}}', -32001),
        ("notification form (no id)", b'{"jsonrpc":"2.0","method":"tools/call","params":{"name":"admin_reset_key"}}', -32001),
        ("UTF-16 body", json.dumps({"jsonrpc": "2.0", "id": 7, "method": "tools/call", "params": {"name": "admin_reset_key"}}).encode("utf-16"), -32001),
        ("UTF-8 BOM", b"\xef\xbb\xbf" + _call_bytes("admin_reset_key"), -32001),
        ("trailing garbage", _call_bytes("admin_reset_key") + b" xyz", -32700),
        ("not JSON at all", b"tools/call admin_reset_key", -32700),
        ("empty batch", b"[]", -32600),
        ("batch of non-objects", b"[1, 2]", -32600),
        ("JSON string", b'"tools/call"', -32600),
    ],
)
def test_bypass_attempts_never_reach_downstream(stack, viewer, label, raw, expected_code):
    r = viewer.post(None, raw=raw)
    assert error_code(r) == expected_code, label
    assert stack.executed() == [], label


def test_spoofed_role_header_does_not_grant_admin(stack, viewer):
    r = viewer.post(None, raw=_call_bytes("admin_reset_key"), **{"X-Gateway-Role": "admin"})
    assert error_code(r) == -32001
    assert stack.executed() == []


def test_compressed_body_is_refused(stack, viewer):
    r = viewer.post(None, raw=gzip.compress(_call_bytes("admin_reset_key")), **{"Content-Encoding": "gzip"})
    assert r.status_code == 415
    assert stack.executed() == []


def test_oversized_body_is_refused(stack, viewer):
    big = b'{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{"pad":"' + b"x" * 1_100_000 + b'"}}'
    r = viewer.post(None, raw=big)
    assert r.status_code == 413


# --- Sessions --------------------------------------------------------------------------------------------


def test_session_belongs_to_the_user_who_opened_it(stack, admin):
    intruder = McpClient(stack.gateway, token("viewer", "mallory"))
    intruder.session_id = admin.session_id

    for r in (intruder.call("get_status"), httpx.get(stack.gateway, headers=intruder.headers())):
        assert r.status_code == 403
        assert r.json()["error"]["code"] == -32002
    assert stack.executed() == []


def test_same_user_with_a_new_token_keeps_their_session(stack, admin):
    admin.bearer = token("admin", "alice")  # e.g. after a token refresh
    assert body(admin.call("get_status"))["result"]["isError"] is False


def test_unknown_session_gets_404(stack):
    client = McpClient(stack.gateway, token("viewer", "bob"))
    client.session_id = "does-not-exist"
    r = client.call("get_status")
    assert r.status_code == 404


def test_delete_never_forwards_a_body(stack):
    client = McpClient(stack.gateway, token("viewer", "erin")).initialize()
    httpx.request("DELETE", stack.gateway, headers=client.headers(), content=_call_bytes("admin_reset_key"))
    assert stack.received_headers()[-1].get("content-length", "0") == "0"
    assert stack.executed() == []


def test_delete_ends_the_session(stack):
    client = McpClient(stack.gateway, token("viewer", "dave")).initialize()
    r = httpx.delete(stack.gateway, headers=client.headers())
    assert r.status_code == 200
    assert client.call("get_status").status_code == 404


# --- Streaming responses (SSE) -------------------------------------------------------------------------


def test_sse_responses_are_streamed_through(sse_stack):
    client = McpClient(sse_stack.gateway, token("viewer", "bob")).initialize()
    r = client.call("get_status")
    assert r.headers["content-type"].startswith("text/event-stream")
    assert body(r)["result"]["isError"] is False


def test_admin_block_also_applies_with_sse_downstream(sse_stack):
    client = McpClient(sse_stack.gateway, token("viewer", "bob")).initialize()
    r = client.call(*ADMIN_CALL)
    assert error_code(r) == -32001
    assert sse_stack.executed() == []


# --- Downstream failure ----------------------------------------------------------------------------------


def test_downstream_down_gives_502_without_internal_details():
    down = start_stack(downstream_url="http://127.0.0.1:9/mcp")  # nothing listens on port 9
    try:
        r = McpClient(down.gateway, token("viewer", "bob")).post({"jsonrpc": "2.0", "id": 4, "method": "tools/list"})
        assert r.status_code == 502
        assert r.json() == {"jsonrpc": "2.0", "id": 4, "error": {"code": -32003, "message": "Downstream MCP server unavailable"}}
    finally:
        stop_stack(down)
