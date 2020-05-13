from loguru import logger

logger.add(
    "train.log", filter=lambda record: "training" in record["extra"], rotation="1 day"
)
logger.add(
    "prediction.log",
    filter=lambda record: "prediction" in record["extra"],
    rotation="1 day",
)

training_logger = logger.bind(training=True)
prediction_logger = logger.bind(predidctio=True)


def log_training(model_version, shape, runtime, metric):
    training_logger.info("{};{};{};{}".format(model_version, shape, runtime, metric))


def log_prediction(model_version, prediction, runtime):
    prediction_logger.info("{};{};{}".format(model_version, prediction, runtime))
