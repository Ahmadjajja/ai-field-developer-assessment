"""The real server with a slow billing provider (20 ms per refund).

Real payment APIs take time. That widens the gap between "check the balance"
and "record the refund", which is exactly where a race would let concurrent
refunds overspend. Used by the concurrency test.
"""

import stdio_guard

stdio_guard.claim()

import time  # noqa: E402

import anyio  # noqa: E402

import data  # noqa: E402
import server  # noqa: E402

_real_send = data.send_to_billing_provider


def _slow_send(customer_id, amount):
    time.sleep(0.02)  # stands in for the network call to the payment provider
    return _real_send(customer_id, amount)


data.send_to_billing_provider = _slow_send

if __name__ == "__main__":
    try:
        anyio.run(server.serve_stdio)
    finally:
        server.log_handler.drain(timeout=2)
