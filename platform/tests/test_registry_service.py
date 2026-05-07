"""Registry service tests.

File summary:
- Tests registry promotion behavior without a live MLflow server.
- Uses a dummy MLflow client that records alias calls.
- Verifies model promotion uses the configured Production alias.
"""

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
        mlflow_tracking_uri="http://mlflow:5001",
        webhook_hmac_secret="secret",
        mlflow_model_alias="Production",
    )
    service = RegistryService.__new__(RegistryService)
    service.settings = settings
    service.client = DummyClient()

    service.promote_model_version("model", "2")

    assert service.client.alias_calls == [("model", "Production", "2")]
