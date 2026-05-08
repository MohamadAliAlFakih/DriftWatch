"""Registry service tests.

File summary:
- Tests registry promotion behavior without a live MLflow server.
- Uses a dummy MLflow client that records alias calls.
- Verifies model promotion uses the configured Production alias.
- Verifies fallback selection uses the latest non-Production version.
"""

from types import SimpleNamespace

from app.config import Settings
from app.services.registry_service import RegistryService


class DummyClient:
    """Minimal MLflow client double that records alias promotion calls."""

    def __init__(self) -> None:
        """Initialize the list of alias calls captured by the test."""
        self.alias_calls = []

    def set_registered_model_alias(self, name, alias, version):
        """Record a requested MLflow alias assignment."""
        self.alias_calls.append((name, alias, version))


def test_promote_uses_mlflow_alias_call() -> None:
    """Verify registry promotion sets the configured MLflow model alias."""
    settings = Settings(
        platform_database_url="sqlite:///:memory:",
        mlflow_tracking_uri="http://mlflow:5000",
        webhook_hmac_secret="secret",
        mlflow_model_alias="Production",
    )
    service = RegistryService.__new__(RegistryService)
    service.settings = settings
    service.client = DummyClient()

    service.promote_model_version("model", "2")

    assert service.client.alias_calls == [("model", "Production", "2")]


def test_fallback_model_uses_latest_non_production_version() -> None:
    """Verify Production is skipped when choosing a registry fallback model."""
    settings = Settings(
        platform_database_url="sqlite:///:memory:",
        mlflow_tracking_uri="http://mlflow:5000",
        webhook_hmac_secret="secret",
        mlflow_model_alias="Production",
    )
    service = RegistryService.__new__(RegistryService)
    service.settings = settings
    service.client = SimpleNamespace(
        search_model_versions=lambda _query: [
            SimpleNamespace(
                name="driftwatch-bank-marketing",
                version="1",
                current_stage="Archived",
                last_updated_timestamp=100,
                creation_timestamp=90,
                run_id=None,
                source="s1",
            ),
            SimpleNamespace(
                name="driftwatch-bank-marketing",
                version="2",
                current_stage="Production",
                last_updated_timestamp=200,
                creation_timestamp=190,
                run_id=None,
                source="s2",
            ),
            SimpleNamespace(
                name="driftwatch-bank-marketing",
                version="3",
                current_stage="Archived",
                last_updated_timestamp=300,
                creation_timestamp=290,
                run_id=None,
                source="s3",
            ),
        ]
    )
    service.get_current_production_model = lambda: SimpleNamespace(model_version="2")

    fallback = service.get_fallback_model()

    assert fallback is not None
    assert fallback.model_version == "3"
    assert fallback.model_uri == "models:/driftwatch-bank-marketing/3"
