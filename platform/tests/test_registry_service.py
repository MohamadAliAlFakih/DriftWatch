"""Registry service tests."""

from app.config import Settings
from app.services.registry_service import RegistryService


class DummyClient:
    def __init__(self) -> None:
        self.alias_calls = []

    def set_registered_model_alias(self, name, alias, version):
        self.alias_calls.append((name, alias, version))


def test_promote_uses_mlflow_alias_call() -> None:
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
