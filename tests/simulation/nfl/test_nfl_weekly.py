from datetime import datetime

from goalpost.simulation.nfl.weekly import week_for_date


def test_week_for_date_thursday_starts_week():
    thu = datetime(2024, 9, 12)  # Thursday
    assert week_for_date(thu) == week_for_date(thu)


def test_tuesday_belongs_to_preceding():
    tue = datetime(2024, 9, 10)  # Tuesday
    mon = datetime(2024, 9, 9)   # Monday (before)
    assert week_for_date(tue) == week_for_date(mon)


def test_wednesday_belongs_to_preceding():
    wed = datetime(2024, 9, 11)  # Wednesday
    mon = datetime(2024, 9, 9)   # Monday
    assert week_for_date(wed) == week_for_date(mon)
