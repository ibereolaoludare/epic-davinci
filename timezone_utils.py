from datetime import datetime, timezone

def to_utc(dt: datetime) -> datetime:
    """Ensure a datetime object is timezone-aware and set to UTC."""
    if dt.tzinfo is None:
        # Assume naive is already UTC for safety, or prompt strict awareness.
        # In this pipeline, we enforce passing aware datetimes or explicit timestamps.
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

def is_weekend_gap(current_time: datetime, prev_time: datetime, expected_interval_hours: int) -> bool:
    """Determine if the gap between candles is due to a weekend."""
    diff = current_time - prev_time
    # If gap is significantly larger than the expected candle interval, it's likely a weekend
    if diff.total_seconds() / 3600 > expected_interval_hours * 1.5:
        return True
    return False

def get_session(dt: datetime) -> str:
    """Return the active forex session based on UTC time."""
    utc_dt = to_utc(dt)
    hour = utc_dt.hour
    
    # Simple static mapping (ignoring DST shifts for simplicity, but strictly UTC-based)
    # Tokyo: 00:00 - 09:00 UTC
    # London: 08:00 - 17:00 UTC
    # NY: 13:00 - 22:00 UTC
    if 0 <= hour < 8:
        return 'Tokyo'
    elif 8 <= hour < 13:
        return 'London'
    elif 13 <= hour < 17:
        return 'London/NY Overlap'
    elif 17 <= hour < 22:
        return 'NY'
    else:
        return 'Sydney/Off-hours'
