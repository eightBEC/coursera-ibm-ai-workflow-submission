from fastapi import APIRouter

from app.api.routes.v1 import model

api_router = APIRouter()
api_router.include_router(model.router, tags=["model"], prefix="/model")
