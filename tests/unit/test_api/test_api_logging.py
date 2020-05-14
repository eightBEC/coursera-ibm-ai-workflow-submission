
import app.core.config as app_cfg


def test_fetch_training_log(test_client) -> None:
    response = test_client.get("/api/v1/logs/training/2020/05/01",
                               headers={"token": str(app_cfg.API_KEY)})
    assert response.status_code == 200
    assert response.json() == [['2', '(273, 7)', '3.3', '16.0']]


def test_fetch_prediction_log(test_client) -> None:
    response = test_client.get("/api/v1/logs/prediction/2020/05/01",
                               headers={"token": str(app_cfg.API_KEY)})
    assert response.status_code == 200
    assert response.json() == [['2', '3.3', '16.0']]


def test_fetch_invalid_log_type(test_client) -> None:
    response = test_client.get("/api/v1/logs/some_unknown_log/2020/05/01",
                               headers={"token": str(app_cfg.API_KEY)})
    assert response.status_code == 200
