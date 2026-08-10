import logging

from my_zero.utils.logger_config import SimFormatter, style_string


def make_record(level=logging.INFO, **extra):
    record = logging.LogRecord(
        "my_zero.train", level, __file__, 1, "Training %s", ("step",), None
    )
    for name, value in extra.items():
        setattr(record, name, value)
    return record


def test_sim_formatter_includes_standard_and_training_fields():
    output = SimFormatter(datefmt="%Y").format(make_record(iteration=2, time=1.234))

    assert "train" in output
    assert "INFO" in output
    assert "Iteration 2.00:" in output
    assert "Training step" in output
    assert "1.23 seconds" in output


def test_sim_formatter_colors_warning_messages():
    output = SimFormatter().format(make_record(logging.WARNING))

    assert "\x1b[93mWARNING" in output
    assert "\x1b[93mTraining step" in output


def test_style_string_does_not_modify_supplied_styles():
    styles = [1]

    assert style_string("text", style_spec=styles, underline=True) == (
        "\x1b[1;4mtext\x1b[0m"
    )
    assert styles == [1]
