"""TurboMemory v3.1 Confidence Decay Policy."""

import math
import time


def decay_confidence(
    conf: float, 
    age_seconds: int, 
    half_life_days: float
) -> float:
    """
    Apply exponential decay to confidence based on age.
    
    Args:
        conf: Current confidence (0-1)
        age_seconds: Age in seconds
        half_life_days: Half-life in days
        
    Returns:
        Decayed confidence
    """
    if half_life_days <= 0:
        return conf
    
    half_life_seconds = half_life_days * 86400.0
    decay_factor = 0.5 ** (age_seconds / half_life_seconds)
    return conf * decay_factor


def is_expired(
    created_at: int, 
    ttl_seconds: int
) -> bool:
    """Check if memory has expired based on TTL."""
    if ttl_seconds is None or ttl_seconds <= 0:
        return False
    
    import time
    now = int(time.time())
    return (now - created_at) >= ttl_seconds