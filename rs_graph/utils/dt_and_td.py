#!/usr/bin/env python

from datetime import timedelta
from numbers import Number
from typing import Literal

timedelta_sizes = {
    "s": 1,
    "ms": 1e-3,
    "us": 1e-6,
    "ns": 1e-9,
    "m": 60,
    "h": 3600,
    "d": 3600 * 24,
    "w": 7 * 3600 * 24,
    "y": 52 * 7 * 3600 * 24,
}

tds2 = {
    "second": 1,
    "minute": 60,
    "hour": 60 * 60,
    "day": 60 * 60 * 24,
    "week": 7 * 60 * 60 * 24,
    "year": 52 * 7 * 60 * 60 * 24,
    "millisecond": 1e-3,
    "microsecond": 1e-6,
    "nanosecond": 1e-9,
}
tds2.update({k + "s": v for k, v in tds2.items()})
timedelta_sizes.update(tds2)
timedelta_sizes.update({k.upper(): v for k, v in timedelta_sizes.items()})


def parse_timedelta(  # noqa: C901
    s: str | float | timedelta,
    default: str | Literal[False] = "seconds",
) -> timedelta:
    """Parse timedelta string to number of seconds.

    Parameters
    ----------
    s : str, float, timedelta
        String to parse, or a float representing seconds, or a timedelta object.
    default: str or False, optional
        Unit of measure if s  does not specify one. Defaults to seconds.
        Set to False to require s to explicitly specify its own unit.

    Examples
    --------
    >>> from datetime import timedelta
    >>> from dask.utils import parse_timedelta
    >>> parse_timedelta('3s')
    3
    >>> parse_timedelta('3.5 seconds')
    3.5
    >>> parse_timedelta('300ms')
    0.3
    >>> parse_timedelta(timedelta(seconds=3))  # also supports timedeltas
    3

    Notes
    -----
    This function was copied from dask.utils.parse_timedelta.
    It was modified to add support for years and linted, formatted, and typed.
    """
    if isinstance(s, timedelta):
        return s
    if isinstance(s, Number):
        s_str = str(s)
    if isinstance(s, str):
        s_str = s

    # Should be a string now, parse
    s_str = s_str.replace(" ", "")
    if not s_str[0].isdigit():
        s_str = "1" + s_str

    for i in range(len(s_str) - 1, -1, -1):
        if not s_str[i].isalpha():
            break
    index = i + 1

    prefix = s_str[:index]
    suffix = s_str[index:] or default
    if suffix is False:
        raise ValueError(f"Missing time unit: {s_str}")
    if not isinstance(suffix, str):
        raise TypeError(f"default must be str or False, got {default!r}")

    n = float(prefix)

    try:
        multiplier = timedelta_sizes[suffix.lower()]
    except KeyError:
        valid_units = ", ".join(timedelta_sizes.keys())
        raise KeyError(f"Invalid time unit: {suffix}. Valid units are: {valid_units}") from None

    result = n * multiplier
    if int(result) == result:
        result = int(result)
    return timedelta(seconds=result)
