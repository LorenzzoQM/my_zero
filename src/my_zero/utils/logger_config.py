import logging

ANSI_COLORS = {
    "GRAY": 90,
    "RED": 91,
    "GREEN": 92,
    "YELLOW": 93,
    "BLUE": 94,
    "MAGENTA": 95,
    "CYAN": 96,
    "WHITE": 97,
    "DARK_GRAY": 30,
    "DARK_RED": 31,
    "DARK_GREEN": 32,
    "DARK_YELLOW": 33,
    "DARK_BLUE": 34,
    "DARK_MAGENTA": 35,
    "DARK_CYAN": 36,
    "DARK_WHITE": 37,
}
LEVEL_COLORS = {
    logging.WARNING: "YELLOW",
    logging.ERROR: "RED",
    logging.CRITICAL: "RED",
}


def style_string(
    string,
    no_format=False,
    style_spec=None,
    color=None,
    background_color=None,
    bold=False,
    emph=False,
    underline=False,
):
    if no_format:
        return string

    styles = list(style_spec or ())
    if color:
        styles.append(ANSI_COLORS[color.upper()])
    if background_color:
        styles.append(ANSI_COLORS[background_color.upper()] + 10)
    styles.extend(
        code for enabled, code in ((bold, 1), (emph, 3), (underline, 4)) if enabled
    )
    if not styles:
        return string
    return f"\x1b[{';'.join(map(str, styles))}m{string}\x1b[0m"


class SimFormatter(logging.Formatter):
    """Format training logs with optional iteration and elapsed-time fields."""

    def format(self, record):
        color = LEVEL_COLORS.get(record.levelno)
        critical = record.levelno >= logging.CRITICAL
        shortname = record.name.partition(".")[2] or record.name

        fields = [
            style_string(
                self.formatTime(record, self.datefmt), color="GRAY", emph=True
            ),
            style_string(f"{shortname:<30}", color="GRAY"),
            style_string(f"{record.levelname:<10}", color=color, bold=critical),
        ]
        if hasattr(record, "iteration"):
            fields.append(
                style_string(
                    f"Iteration {record.iteration:.2f}:", color="GREEN", bold=True
                )
            )

        fields.append(style_string(record.getMessage(), color=color, bold=critical))
        if hasattr(record, "time"):
            fields.append(style_string(f"{record.time:.2f} seconds", color="DARK_CYAN"))

        message = " ".join(fields)
        if record.exc_info:
            message += "\n" + self.formatException(record.exc_info)
        if record.stack_info:
            message += "\n" + self.formatStack(record.stack_info)
        return message
