"""TurboMemory v3.1 Decay Daemon."""

import time
import logging
from .decay import decay_confidence, is_expired

logger = logging.getLogger(__name__)


class DecayDaemon:
    """Background daemon for confidence decay and TTL expiration."""
    
    def __init__(self, store, config, interval_seconds: int = 3600):
        self.store = store
        self.config = config
        self.interval = interval_seconds
        self.running = False
    
    def tick(self) -> int:
        """Process one decay cycle."""
        cur = self.store.conn.cursor()
        
        # Get all memories with confidence > 0 and TTL
        cur.execute("""
            SELECT id, confidence, created_at, ttl_seconds 
            FROM memories 
            WHERE confidence > 0.01
        """)
        rows = cur.fetchall()
        
        now = int(time.time())
        updated = 0
        expired = 0
        
        for mem_id, conf, created_at, ttl_seconds in rows:
            # Check TTL expiration
            if is_expired(created_at, ttl_seconds):
                self.store.update_fields(mem_id, {"confidence": 0.0})
                expired += 1
                continue
            
            # Apply decay
            age = now - created_at
            new_conf = decay_confidence(
                conf, 
                age, 
                self.config.decay_half_life_days
            )
            
            # Only update if meaningful change
            if abs(new_conf - conf) > 0.01:
                self.store.update_fields(mem_id, {"confidence": float(new_conf)})
                updated += 1
        
        if updated or expired:
            logger.info(f"Decay tick: updated={updated}, expired={expired}")
        
        return updated + expired
    
    def run_forever(self):
        """Run decay loop."""
        self.running = True
        logger.info(f"Decay daemon started (interval={self.interval}s)")
        
        while self.running:
            try:
                self.tick()
            except Exception as e:
                logger.error(f"Decay daemon error: {e}")
            
            time.sleep(self.interval)
        
        logger.info("Decay daemon stopped")
    
    def stop(self):
        """Stop the daemon."""
        self.running = False