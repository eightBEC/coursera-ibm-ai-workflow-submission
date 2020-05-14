from fastapi import APIRouter, Depends

from app.core import security
from app.services.logger import get_log

router = APIRouter()


@router.get("/{log_type}/{year}/{month}/{day}", name="logs")
def get_logs(
    year: int,
    month: int,
    day: int,
    log_type: str,
    authenticated: bool = Depends(security.validate_request),
):
    try:
        if log_type == "training" or log_type == "prediction":
            return get_log(log_type, year, month, day)
        else:
            return "Expected log type to be either training or prediction."
    except ValueError as e:
        return e
