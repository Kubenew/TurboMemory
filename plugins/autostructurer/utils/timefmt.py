import datetime
def format_ts(seconds: float) -> str:
    return str(datetime.timedelta(seconds=int(max(0, seconds))))