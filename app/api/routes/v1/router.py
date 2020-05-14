from fastapi import APIRouter

from app.api.routes.v1 import model, logger

api_router = APIRouter()
api_router.include_router(model.router, tags=["model"], prefix="/model")
api_router.include_router(logger.router, tags=["logs"], prefix="/logs")
