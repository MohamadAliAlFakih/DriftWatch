"""HMAC-SHA256 verification for incoming drift webhooks.

The platform signs the raw request body with WEBHOOK_HMAC_SECRET and sends the hex
digest in `X-DriftWatch-Signature`. We recompute and compare with hmac.compare_digest
(constant-time) to defeat timing-attack reasoning even though the realistic threat
model in this project is "wrong secret in .env", not active attack. (D-11, AGENT-04.)
"""

import hmac
import hashlib

from pydantic import SecretStr


# verify HMAC over raw body; return True iff signature matches secret
def verify_signature(*, body: bytes, signature_header: str | None, secret: SecretStr) -> bool:
    # missing header is a fast-fail — no need to compute the digest
    if not signature_header:
        return False
    # compute hex digest using the shared secret over the exact bytes received
    expected = hmac.new(
        secret.get_secret_value().encode("utf-8"),
        body,
        hashlib.sha256,
    ).hexdigest()
    # constant-time compare; both must be hex strings of equal length
    return hmac.compare_digest(expected, signature_header.strip())
