from fastapi import APIRouter, Depends

from app.core import security
from app.services.model import ModelService

router = APIRouter()


@router.post("/train", name="train")
def post_train(authenticated: bool = Depends(security.validate_request)):
    model_service = ModelService()
    result = model_service.train()
    return result


@router.get("/list", name="list models")
def get_list(authenticated: bool = Depends(security.validate_request)):
    model_service = ModelService()
    result = model_service.list_models()
    return result


@router.get("/predict", name="predict")
def get_predict(
    country: str,
    year: int,
    month: int,
    day: int,
    authenticated: bool = Depends(security.validate_request),
):
    model_service = ModelService()
    prediction = model_service.predict(country, year, month, day)
    return prediction
