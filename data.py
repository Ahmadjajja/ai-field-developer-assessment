"""Mock customer data (Task 1).

An in-memory stand-in for a real CRM and billing system. In production these
functions would call the company's existing APIs instead.

Three named customers are seeded. Any other well-formed ID (e.g. CUST-12345)
gets a generated record on first use, so every valid ID works for testing.

Money is kept as Decimal (exact cents), never float: 0.1 + 0.2 must be 0.3.
"""

import threading
import uuid
from datetime import datetime, timezone
from decimal import Decimal

CENT = Decimal("0.01")
DEFAULT_TOTAL_PAID = Decimal("500.00")


class RefundRefused(Exception):
    """The refund is valid input but not allowed (e.g. it exceeds the balance)."""


# The SDK runs tools in worker threads, so two refunds for the same customer can
# run at the same time. Everything that reads or changes balances holds this
# lock, so "check the balance, then refund" is one atomic step.
_lock = threading.Lock()

CUSTOMERS: dict[str, dict] = {
    "CUST-10001": {"customer_id": "CUST-10001", "name": "Jane Doe", "email": "jane@example.com", "plan": "Pro", "total_paid": Decimal("240.00"), "total_refunded": Decimal("0.00")},
    "CUST-10002": {"customer_id": "CUST-10002", "name": "Omar Khan", "email": "omar@example.com", "plan": "Basic", "total_paid": Decimal("60.00"), "total_refunded": Decimal("20.00")},
    "CUST-10003": {"customer_id": "CUST-10003", "name": "Li Wei", "email": "li@example.com", "plan": "Enterprise", "total_paid": Decimal("1200.00"), "total_refunded": Decimal("0.00")},
}

REFUNDS: list[dict] = []


def _customer(customer_id: str) -> dict:
    """Seeded or generated record. Caller must hold _lock."""
    if customer_id not in CUSTOMERS:
        number = customer_id.removeprefix("CUST-")
        CUSTOMERS[customer_id] = {
            "customer_id": customer_id,
            "name": f"Test Customer {number}",
            "email": f"customer{number}@example.com",
            "plan": "Basic",
            "total_paid": DEFAULT_TOTAL_PAID,
            "total_refunded": Decimal("0.00"),
        }
    return CUSTOMERS[customer_id]


def _public(customer: dict) -> dict:
    """JSON-friendly copy of a customer record, with the refundable balance."""
    return {
        **customer,
        "total_paid": float(customer["total_paid"]),
        "total_refunded": float(customer["total_refunded"]),
        "refundable_balance": float(customer["total_paid"] - customer["total_refunded"]),
    }


def get_customer(customer_id: str) -> dict:
    with _lock:
        return _public(_customer(customer_id))


def send_to_billing_provider(customer_id: str, amount: Decimal) -> str:
    """Mock of the payment provider call that actually moves the money. Returns its refund ID."""
    return f"RF-{uuid.uuid4().hex[:8].upper()}"


def refund(customer_id: str, amount: Decimal, reason: str, reason_code: str | None = None) -> dict:
    """Check the balance, pay out and record the refund as one atomic step."""
    with _lock:
        customer = _customer(customer_id)
        balance = customer["total_paid"] - customer["total_refunded"]
        if amount > balance:
            raise RefundRefused(f"Refund of {amount} exceeds refundable balance of {balance}")

        refund_id = send_to_billing_provider(customer_id, amount)
        customer["total_refunded"] += amount
        record = {
            "refund_id": refund_id,
            "customer_id": customer_id,
            "amount": float(amount),
            "reason": reason,
            "reason_code": reason_code,
            "status": "processed",
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        REFUNDS.append(record)
        return record
