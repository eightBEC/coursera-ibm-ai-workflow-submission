import os
import re
import time
from collections import defaultdict
from typing import Set

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.core import config as app_cfg
from app.core.messages import (
    INVALID_MODEL_NAME_PARAMS,
    MODEL_FOLDER_NOT_FOUND,
    MODEL_NOT_FOUND,
    MODEL_UNABLE_TO_EXTRACT_VERSION,
    NO_COUNTRY_GIVEN,
    NO_DATA_FOR_COUNTRY,
    DATE_NOT_IN_RANGE,
)
from app.services.logger import log_prediction, log_training

RANDOM_SEED = 11


class ModelService(object):
    def __init__(self):
        self.models = self._load_models()

    def _load_training_data(self):
        pass

    def _get_time_series_data(self):
        data_dir = app_cfg.DATA_FOLDER
        return {
            re.sub(r".csv", "", cf)[3:]: pd.read_csv(os.path.join(data_dir, cf))
            for cf in os.listdir(data_dir)
        }

    def _get_time_series_data_for_country(self, country):
        ts_data = self._get_time_series_data()
        if country in ts_data:
            ts_for_country = ts_data[country]
            x, y, dates = self._get_engineered_features(ts_for_country)
            dates = np.array([str(d) for d in dates])
            return {"X": x, "y": y, "dates": dates}
        else:
            raise ValueError(NO_DATA_FOR_COUNTRY.format(country))

    def _get_latest_model(self, models):
        if models is None or len(models) == 0:
            return None

        return max(models)

    def _get_models_by_country(self, country: str):
        if country is None:
            return set()
        else:
            model_folder_files = [f for f in os.listdir(app_cfg.MODEL_FOLDER)]
            return set(
                filter(
                    lambda x: x.endswith(".joblib")
                    and len(x.split("-")) == 3
                    and x.split("-")[1] == country,
                    model_folder_files,
                )
            )

    def _get_model_version(self, model_name):
        if model_name is None or len(model_name.split("-")) != 3:
            raise ValueError(MODEL_UNABLE_TO_EXTRACT_VERSION.format(model_name))

        model_version_ending = model_name.split("-")[2]

        return model_version_ending.split(".")[0]

    def _get_latest_model_for_country(self, country: str):
        if country is None:
            return None

        models = self._get_models_by_country(country)
        latest_model = self._get_latest_model(models)

        return latest_model

    def _get_countries(self) -> Set[str]:
        model_folder_files = [f for f in os.listdir(app_cfg.MODEL_FOLDER)]
        return set(
            map(
                lambda x: x.split("-")[1],
                filter(
                    lambda x: x.endswith(".joblib") and len(x.split("-")) == 3,
                    model_folder_files,
                ),
            )
        )

    def _load_model(self, model_file):
        logger.debug("Loading latest model.")
        if not os.path.exists(os.path.join(app_cfg.MODEL_FOLDER, model_file)):
            raise ValueError(MODEL_FOLDER_NOT_FOUND.format(app_cfg.MODEL_FOLDER))

        model = joblib.load(os.path.join(app_cfg.MODEL_FOLDER, model_file))

        logger.info("Loaded model: {}".format(model_file))
        return model

    def _load_models(self):
        models = {}

        for country in self._get_countries():
            model_name = self._get_latest_model_for_country(country)
            latest_model_instance = self._load_model(model_name)
            model_version = self._get_model_version(model_name)
            models[country] = {
                "model": latest_model_instance,
                "version": model_version,
            }

        return models

    def _get_model_by_country(self, country):
        models = self.models
        if models is None or len(models) == 0:
            models = self._load_models()

        if models is None or len(models) == 0:
            raise ValueError(MODEL_NOT_FOUND.format(country))

        if country is None or country not in models:
            raise ValueError(MODEL_NOT_FOUND.format(country))

        return models[country]

    def _get_engineered_features(self, df):

        # extract dates
        dates = df["date"].values.copy()
        dates = dates.astype("datetime64[D]")

        # engineer some features
        eng_features = defaultdict(list)
        previous = [7, 14, 28, 70]
        y = np.zeros(dates.size)
        for d, day in enumerate(dates):
            current = np.zeros(0)

            # use windows in time back from a specific date
            for num in previous:
                current = np.datetime64(day, "D")
                prev = current - np.timedelta64(num, "D")
                mask = np.in1d(dates, np.arange(prev, current, dtype="datetime64[D]"))
                eng_features["previous_{}".format(num)].append(df[mask]["revenue"].sum())

            # get get the target revenue
            plus_30 = current + np.timedelta64(30, "D")
            mask = np.in1d(dates, np.arange(current, plus_30, dtype="datetime64[D]"))
            y[d] = df[mask]["revenue"].sum()

            # attempt to capture monthly trend with previous years data (if present)
            start_date = current - np.timedelta64(365, "D")
            stop_date = plus_30 - np.timedelta64(365, "D")
            mask = np.in1d(dates, np.arange(start_date, stop_date, dtype="datetime64[D]"))
            eng_features["previous_year"].append(df[mask]["revenue"].sum())

            # add some non-revenue features
            minus_30 = current - np.timedelta64(30, "D")
            mask = np.in1d(dates, np.arange(minus_30, current, dtype="datetime64[D]"))
            eng_features["recent_invoices"].append(df[mask]["unique_invoices"].mean())
            eng_features["recent_views"].append(df[mask]["total_views"].mean())

        x = pd.DataFrame(eng_features)
        # combine features in to df and remove rows with all zeros
        x.fillna(0, inplace=True)
        mask = x.sum(axis=1) > 0
        x = x[mask]
        y = y[mask]
        dates = dates[mask]
        x.reset_index(drop=True, inplace=True)
        return (x, y, dates)

    def _get_train_test_split(self, x, y):
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.25, shuffle=True, random_state=RANDOM_SEED
        )
        return (x_train, x_test, y_train, y_test)

    def _train_model(self, x, y):
        param_grid_rf = {
            "rf__criterion": ["mse", "mae"],
            "rf__n_estimators": [10, 15, 20, 25],
        }

        pipe_rf = Pipeline(
            steps=[("scaler", StandardScaler()), ("rf", RandomForestRegressor())]
        )

        grid_estimator = GridSearchCV(
            pipe_rf, param_grid=param_grid_rf, cv=5, iid=False, n_jobs=-1
        )
        grid_estimator.fit(x, y)
        return grid_estimator

    def _evaluate_model(self, model, x_test, y_test):
        y_pred = model.predict(x_test)
        rmse = round(np.sqrt(mean_squared_error(y_test, y_pred)))
        return rmse

    def _get_new_model_version(self, country):
        latest_model = self._get_latest_model_for_country(country)
        if latest_model is None:
            return 1

        try:
            latest_model_version = int(self._get_model_version(latest_model))
            return latest_model_version + 1
        except ValueError as e:
            logger.error(e)

    def _get_model_path(self, country, model_version):
        if country is None or model_version is None:
            raise ValueError(INVALID_MODEL_NAME_PARAMS.format(country, model_version))
        return "{}-{}-{}.joblib".format("prod", country, model_version)

    def _train_for_country(self, df, country):
        start_time = time.time()

        # get engineered features
        x, y, __ = self._get_engineered_features(df)

        # train / test split
        x_train, x_test, y_train, y_test = self._get_train_test_split(x, y)

        # train model
        train_model = self._train_model(x_train, y_train)

        # evaluate model
        metrics = self._evaluate_model(train_model, x_test, y_test)

        # retrain with full data
        full_model = self._train_model(x, y)

        # get new model version
        model_version = self._get_new_model_version(country)

        # update training log
        runtime = time.time() - start_time

        log_training(model_version, df.shape, runtime, metrics)

        # persist model
        model_name = self._get_model_path(country, model_version)
        joblib.dump(full_model, os.path.join(app_cfg.MODEL_FOLDER, model_name))
        logger.info("Successfully trained model {}".format(model_name))

    def predict(self, country: str, year: int, month: int, day: int):
        logger.info("Predicting")
        start_time = time.time()

        if country is None:
            raise ValueError(NO_COUNTRY_GIVEN)

        model_data = self._get_model_by_country(country)
        model = model_data["model"]
        model_version = model_data["version"]

        country_data = self._get_time_series_data_for_country(country)
        target_date = "{}-{}-{}".format(year, str(month).zfill(2), str(day).zfill(2))

        if target_date not in country_data["dates"]:
            raise ValueError(
                DATE_NOT_IN_RANGE.format(
                    target_date, country_data["dates"][0], country_data["dates"][1]
                )
            )

        date_idx = np.where(country_data["dates"] == target_date)[0][0]
        payload = country_data["X"].iloc[[date_idx]]

        prediction = model.predict(payload)

        runtime = time.time() - start_time

        log_prediction(model_version, prediction, runtime)

        return {"y_pred": prediction[0]}

    def train(self):
        logger.info("Training all models")

        time_series_data = self._get_time_series_data()

        for country, df in time_series_data.items():
            self._train_for_country(df, country)

        logger.info("Finished training models")
        return "Finished model training"

    def list_models(self):
        logger.info("Listing models")

        model_info = {}

        for country in self._get_countries():
            latest_model = self._get_latest_model_for_country(country)
            model_info[country] = latest_model

        return model_info
