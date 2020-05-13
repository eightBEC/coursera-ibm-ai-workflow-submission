from fastapi import APIRouter, Depends

from app.core import security

router = APIRouter()


@router.get("logs/{log_id}", name="logs")
def get_logs(log_id, authenticated: bool = Depends(security.validate_request)):
    pass
