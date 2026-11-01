"""Bearer token handling for the gateway.

Tokens are JWTs signed with HS256. The gateway trusts a token only if:
  - the signature matches our secret (nobody can forge "role": "admin"),
  - the algorithm is exactly HS256 (blocks "alg": "none" and algorithm-swap tricks),
  - it hasn't expired ("exp" is required),
  - it was issued for this gateway ("aud" must match),
  - it names a user ("sub") and a known role.

Payload example: {"sub": "alice", "role": "admin", "aud": "mcp-gateway", "exp": 1767225600}
"""

import os
import time
from dataclasses import dataclass

import jwt

ALGORITHM = "HS256"
AUDIENCE = "mcp-gateway"
ROLES = {"admin", "viewer"}

# The signing secret. Set GATEWAY_JWT_SECRET in any real deployment; the
# default only exists so the demo runs out of the box.
DEV_SECRET = "dev-only-secret-change-me-0123456789abcdef"
SECRET = os.environ.get("GATEWAY_JWT_SECRET", DEV_SECRET)


@dataclass(frozen=True)
class Identity:
    user: str
    role: str


class AuthError(Exception):
    """The request has no usable credentials. The message is safe to show the client."""


def bearer_token(authorization_header: str | None) -> str:
    """Pull the token out of an "Authorization: Bearer <token>" header."""
    if not authorization_header:
        raise AuthError("Missing Authorization header")
    scheme, _, token = authorization_header.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise AuthError("Authorization header must be 'Bearer <token>'")
    return token.strip()


def verify(token: str) -> Identity:
    try:
        claims = jwt.decode(
            token,
            SECRET,
            algorithms=[ALGORITHM],
            audience=AUDIENCE,
            options={"require": ["exp", "sub", "aud"]},
        )
    except jwt.ExpiredSignatureError:
        raise AuthError("Token has expired") from None
    except jwt.InvalidTokenError:
        # One message for every other failure (bad signature, wrong audience, garbage...),
        # so the client learns nothing about why a forged token was rejected.
        raise AuthError("Invalid token") from None

    role = claims.get("role")
    if role not in ROLES:
        raise AuthError("Invalid token")
    return Identity(user=str(claims["sub"]), role=role)


def create_token(user: str, role: str, ttl_seconds: int = 3600, secret: str = SECRET) -> str:
    """Issue a token. In production an identity provider does this, not the gateway."""
    if role not in ROLES:
        raise ValueError(f"role must be one of {sorted(ROLES)}")
    now = int(time.time())
    claims = {"sub": user, "role": role, "aud": AUDIENCE, "iat": now, "exp": now + ttl_seconds}
    return jwt.encode(claims, secret, algorithm=ALGORITHM)
