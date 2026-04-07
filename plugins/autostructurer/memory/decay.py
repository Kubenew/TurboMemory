import math, time

def decay_factor(created_at_ts: float, half_life_days=90.0):
    age_days = max(0.0, (time.time() - created_at_ts) / 86400.0)
    return math.exp(-age_days / half_life_days)