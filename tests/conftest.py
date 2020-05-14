import pytest
from starlette.config import environ
from starlette.testclient import TestClient

environ["API_KEY"] = "aaaaaaaa-bbbb-cccc-cccc-dddddddddddd"
environ["IS_DEBUG"] = "True"
environ["MODEL_FOLDER"] = "./tests/data/models"
environ["DATA_FOLDER"] = "./tests/data/cs-train/ts-data"
environ["LOG_FOLDER"] = "./tests/data/logs"

from app.main import get_app  # noqa: E402


@pytest.fixture()
def test_client():
    app = get_app()
    with TestClient(app) as test_client:
        yield test_client
