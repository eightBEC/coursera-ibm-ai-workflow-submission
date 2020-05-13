from fastapi import FastAPI

from app.api.routes.v1.router import api_router
from app.core.config import API_PREFIX, APP_NAME, APP_VERSION, IS_DEBUG


def get_app() -> FastAPI:
    app = FastAPI(title=APP_NAME, version=APP_VERSION, debug=IS_DEBUG)
    app.include_router(api_router, prefix=API_PREFIX + "/v1")
    return app


app = get_app()
