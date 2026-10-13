import math


def coalesce_zero_time(segments: list) -> list:
    merged = []
    for seg in segments:
        if seg.get("elapsed", 0) == 0 and merged:
            merged[-1]["yards"] = merged[-1].get("yards", 0) + seg.get("yards", 0)
        else:
            merged.append(seg)
    return merged


def validate_clock_progress(events: list, strict: bool = True):
    for i in range(1, len(events)):
        prev = events[i - 1]
        curr = events[i]
        if curr.get("remaining", 0) > prev.get("remaining", 0):
            if strict:
                raise ValueError("clock regression")
            else:
                curr["remaining"] = prev["remaining"]
