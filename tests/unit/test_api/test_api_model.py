import os

import app.core.config as app_cfg


def test_prediction(test_client) -> None:
    response = test_client.get(
        "/api/v1/model/predict?" + "country=all&year=2018&month=02&day=01",
        headers={"token": str(app_cfg.API_KEY)},
    )
    assert response.status_code == 200
    assert "y_pred" in response.json()
    assert response.json()["y_pred"] > 0


def test_training(test_client) -> None:
    response = test_client.post(
        "/api/v1/model/train", headers={"token": str(app_cfg.API_KEY)}
    )
    assert response.status_code == 200

    os.remove("./tests/data/models/prod-eire-1.joblib")
    os.remove("./tests/data/models/prod-france-1.joblib")
    os.remove("./tests/data/models/prod-all-2.joblib")
    assert not os.path.exists("./tests/data/models/prod-eire-1.joblib")
    assert not os.path.exists("./tests/data/models/prod-france-1.joblib")
    assert not os.path.exists("./tests/data/models/prod-all-2.joblib")
