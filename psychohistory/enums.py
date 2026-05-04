"""Enumerations for the Psychohistory Engine."""

from enum import Enum


class EventCategory(Enum):
    """Categories for historical events."""

    POLITICAL = "POLITICAL"
    ECONOMIC = "ECONOMIC"
    SOCIAL = "SOCIAL"
    MILITARY = "MILITARY"
    CULTURAL = "CULTURAL"
    NATURAL = "NATURAL"
