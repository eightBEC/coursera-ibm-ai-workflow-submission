import os

from starlette.config import Config
from starlette.datastructures import Secret

APP_VERSION = "0.0.1"
APP_NAME = "AI Workflow Capstone Submission"
API_PREFIX = "/api"

MODEL_FOLDER = os.getenv("MODEL_FOLDER", "./app/models")
DATA_FOLDER = os.getenv("DATA_FOLDER", "./data/cs-train/ts-data")
LOG_FOLDER = os.getenv("LOG_FOLDER", "./logs")

local_use = os.getenv("IS_LOCAL")
if local_use is not None and local_use.lower() == "true":
    config = Config(".env")
else:
    config = Config()

API_KEY: Secret = config("API_KEY", cast=Secret)
IS_DEBUG: bool = config("IS_DEBUG", default=False)
