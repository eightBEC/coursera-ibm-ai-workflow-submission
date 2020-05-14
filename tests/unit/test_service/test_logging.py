import pytest
from app.services.logger import _is_validate_date, get_log, _parse_logs


def test_parse_logs():
    assert _parse_logs(None) == []
    assert _parse_logs([]) == []
    assert _parse_logs(["INFO:24 - ;2;(273, 7);3.3;16847.0"]) == [
        ["2", "(273, 7)", "3.3", "16847.0"]
    ]


def test_get_training_log():
    logs = get_log("training", 2020, 5, 1)
    assert logs == [["2", "(273, 7)", "3.3", "16.0"]]


def test_get_prediction_log():
    logs = get_log("prediction", 2020, 5, 1)
    assert logs == [["2", "3.3", "16.0"]]


def test_get_invalid_log_type():
    with pytest.raises(ValueError):
        get_log("invalid", 2020, 5, 1)

    with pytest.raises(ValueError):
        get_log("invalid", 2020, 5, 90)


def test_is_valid_date():
    assert _is_validate_date(2020, 3, 3)
    assert _is_validate_date(1900, 1, 12)


def test_is_invalid_date():
    assert not _is_validate_date(2000, 13, 1)
    assert not _is_validate_date(2000, 2, 30)
