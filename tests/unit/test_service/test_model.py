import os

import app.core.config as app_cfg
from app.services.model import ModelService


def test_predict():
    model_service = ModelService()
    result = model_service.predict("all", 2018, 1, 2)
    assert "y_pred" in result
    assert result["y_pred"] > 0


def test_train():
    assert not os.path.exists("./tests/data/models/prod-eire-1.joblib")
    assert not os.path.exists("./tests/data/models/prod-france-1.joblib")
    assert not os.path.exists("./tests/data/models/prod-all-2.joblib")

    model_service = ModelService()
    model_service.train()
    trained_models = os.listdir(app_cfg.MODEL_FOLDER)

    assert "prod-eire-1.joblib" in trained_models
    assert "prod-france-1.joblib" in trained_models
    assert "prod-all-2.joblib" in trained_models

    os.remove("./tests/data/models/prod-eire-1.joblib")
    os.remove("./tests/data/models/prod-france-1.joblib")
    os.remove("./tests/data/models/prod-all-2.joblib")
    assert not os.path.exists("./tests/data/models/prod-eire-1.joblib")
    assert not os.path.exists("./tests/data/models/prod-france-1.joblib")
    assert not os.path.exists("./tests/data/models/prod-all-2.joblib")
