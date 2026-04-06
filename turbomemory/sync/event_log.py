"""Sync event log utilities."""

from typing import Dict, List, Any

class SyncEventLog:
    def __init__(self, event_log):
        self.event_log = event_log
    
    def export_for_sync(self, from_event_id: int = 0) -> List[Dict]:
        return self.event_log.export_events(from_event_id)
    
    def import_from_sync(self, events: List[Dict]) -> int:
        return self.event_log.import_events(events)
