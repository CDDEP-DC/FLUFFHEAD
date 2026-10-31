"""Daily contact calendars: weekends, school terms and holidays, and seasonality.

Returns a [N_ROWS, days] array of multipliers in [0, 1], one row per engine
calendar row (household, workplace, school, gq, community, community at work,
import). The engine thins transmission events by these values.

School terms are approximate: districts differ, so states are grouped by when
schools usually start (early August in much of the South and West, late
August, or after Labor Day). Breaks: Thanksgiving Wednesday-Friday, winter
break, one spring week, summer. Seasonality is a daily transmissibility factor
from absolute humidity (weather.factor) when one is given, else a cosine in day
of year peaking at SEASON_PEAK_DOY.
"""
from __future__ import annotations

import datetime as dt

import numpy as np

from .engine import COMM, COMM_WORK, IMPORT, N_ROWS
from .network import GQ, HH, SCH, WP

WEEKEND_WORK = 0.25       # weekend workplace contact relative to weekdays
HOLIDAY_WORK = 0.25
WINTER_WEEK_WORK = 0.6    # Christmas to New Year
SEASON_PEAK_DOY = 15      # mid-January

EARLY_START = {"01", "04", "05", "12", "13", "15", "18", "21", "22", "28", "35", "40", "45", "47", "48", "49", "32"}
LATE_START = {"09", "10", "11", "23", "24", "25", "26", "33", "34", "36", "42", "44", "50", "51"}


def _nth_weekday(year, month, weekday, n):
    d = dt.date(year, month, 1)
    d += dt.timedelta(days=(weekday - d.weekday()) % 7)
    return d + dt.timedelta(weeks=n - 1)


def _last_weekday(year, month, weekday):
    d = dt.date(year, month + 1, 1) - dt.timedelta(days=1) if month < 12 else dt.date(year, 12, 31)
    return d - dt.timedelta(days=(d.weekday() - weekday) % 7)


def federal_holidays(year: int) -> set[dt.date]:
    return {dt.date(year, 1, 1), _nth_weekday(year, 1, 0, 3), _nth_weekday(year, 2, 0, 3),
            _last_weekday(year, 5, 0), dt.date(year, 6, 19), dt.date(year, 7, 4),
            _nth_weekday(year, 9, 0, 1), _nth_weekday(year, 10, 0, 2), dt.date(year, 11, 11),
            _nth_weekday(year, 11, 3, 4), dt.date(year, 12, 25)}


def school_open(day: dt.date, state: str) -> bool:
    if day.weekday() >= 5:
        return False
    y = day.year if day.month >= 7 else day.year - 1        # school year starting in y
    labor = _nth_weekday(y, 9, 0, 1)
    start = (dt.date(y, 8, 8) if state in EARLY_START else
             labor + dt.timedelta(days=1) if state in LATE_START else dt.date(y, 8, 24))
    end = dt.date(y + 1, 5, 25) if state in EARLY_START else dt.date(y + 1, 6, 15)
    if not start <= day <= end:
        return False
    thanks = _nth_weekday(y, 11, 3, 4)
    if thanks - dt.timedelta(days=1) <= day <= thanks + dt.timedelta(days=1):
        return False
    # winter break: Dec 21 through Jan 1 (weekends either side are already closed)
    if dt.date(y, 12, 21) <= day <= dt.date(y + 1, 1, 1):
        return False
    spring = _nth_weekday(y + 1, 3, 0, 3) if state in EARLY_START else _nth_weekday(y + 1, 4, 0, 1)
    if spring <= day < spring + dt.timedelta(days=5):
        return False
    return day not in school_holidays(day.year)


def school_holidays(year: int) -> set[dt.date]:
    """Holidays most districts close for (not Columbus or Veterans Day, which vary)."""
    return {_nth_weekday(year, 1, 0, 3), _nth_weekday(year, 2, 0, 3), _last_weekday(year, 5, 0),
            _nth_weekday(year, 9, 0, 1)}


def seasonality(day: dt.date, amplitude: float) -> float:
    """Relative transmissibility in (0, 1]: 1 at the seasonal peak."""
    doy = day.timetuple().tm_yday
    return (1 + amplitude * np.cos(2 * np.pi * (doy - SEASON_PEAK_DOY) / 365.25)) / (1 + amplitude)


def build(start: dt.date, days: int, state: str, season_amplitude: float = 0.2,
          season: np.ndarray | None = None) -> np.ndarray:
    """`season`, if given, is the daily seasonal factor in (0, 1] (else the cosine with `season_amplitude`)."""
    cal = np.ones((N_ROWS, days))
    for i in range(days):
        d = start + dt.timedelta(days=i)
        hol = d in federal_holidays(d.year)
        work = 1.0
        if d.weekday() >= 5:
            work = WEEKEND_WORK
        elif hol:
            work = HOLIDAY_WORK
        elif (d.month == 12 and d.day >= 24) or (d.month == 1 and d.day == 1):
            work = WINTER_WEEK_WORK
        cal[WP, i] = work
        cal[COMM_WORK, i] = work
        cal[SCH, i] = 1.0 if school_open(d, state) else 0.0
        f = season[i] if season is not None else seasonality(d, season_amplitude)
        for row in (HH, WP, SCH, GQ, COMM, COMM_WORK):
            cal[row, i] *= f
    cal[IMPORT] = 1.0
    return cal
