import os
import datetime

from loguru import logger

from app.core.config import LOG_FOLDER
from app.core.messages import NO_LOG_FOUND, INVALID_DATE

logger.add(
    "./logs/training-{time:YYYY-MM-DD}.log",
    filter=lambda record: "training" in record["extra"],
    rotation="1 day",
)
logger.add(
    "./logs/prediction-{time:YYYY-MM-DD}.log",
    filter=lambda record: "prediction" in record["extra"],
    rotation="1 day",
)

training_logger = logger.bind(training=True)
prediction_logger = logger.bind(predidctio=True)


def _is_validate_date(year: int, month: int, day: int):
    try:
        datetime.datetime(year, month, day)
        return True
    except ValueError:
        return False


def _parse_logs(logs):
    if logs is None or logs == [] or type(logs) != list:
        return []

    return list(map(lambda x: x.split(";")[1:], logs))


def log_training(model_version, shape, runtime, metric):
    training_logger.info(";{};{};{};{}".format(model_version, shape, runtime, metric))


def log_prediction(model_version, prediction, runtime):
    prediction_logger.info(";{};{};{}".format(model_version, prediction, runtime))


def get_log(log_type: str, year: int, month: int, day: int):
    if not _is_validate_date(year, month, day):
        raise ValueError(INVALID_DATE.format(year, month, day))

    target_date = "{}-{}-{}".format(year, str(month).zfill(2), str(day).zfill(2))
    log_filename = "{}-{}.log".format(log_type, target_date)
    log_path = os.path.join(LOG_FOLDER, log_filename)
    if os.path.exists(log_path) and os.path.isfile(log_path):
        with open(log_path, "r") as f:
            return _parse_logs(f.read().splitlines())
    else:
        raise ValueError(NO_LOG_FOUND.format(log_type, year, month, day))
