"""Verify a model URI still exists on the platform side (AGENT-06).

The platform exposes a registry-read endpoint. Plausible default URL:
GET /registry/models/{model_name}/versions/{model_version}.

We treat 200 as exists, 404 as gone, anything else as transient (return True
to avoid false-positive stale terminal states; the platform is the source of
truth for non-existence). D-14.
"""

import httpx

from app.config import Settings
from app.core.logging import get_logger

log = get_logger(__name__)


# return True iff the platform reports the model version still exists
async def model_uri_exists(
    client: httpx.AsyncClient,
    settings: Settings,
    model_name: str,
    model_version: int,
) -> bool:
    # build the read URL against the platform's registry surface
    url = f"{settings.platform_url}/registry/models/{model_name}/versions/{model_version}"
    try:
        # short timeout — this runs on every triage_node entry; do not block the graph
        resp = await client.get(url, timeout=5.0)
    except httpx.HTTPError as exc:
        # transient errors don't mean "stale" — be conservative and assume exists
        log.warning("registry_check_transient_failure", error=str(exc))
        return True
    # 404 is the only signal we treat as "gone" — drives the stale terminal branch
    if resp.status_code == 404:
        return False
    # 200 is the only positive signal — keep the existing path moving
    if resp.status_code == 200:
        return True
    # any other status: log and assume still-exists (don't false-flag stale)
    log.warning("registry_check_non_200_non_404", status=resp.status_code)
    return True
