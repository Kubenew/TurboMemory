"""TMF append-only event log for replication and audit."""

import struct
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Iterator, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import Enum


class EventType(Enum):
    """Event types for TMF log."""
    CHUNK_ADD = "chunk_add"
    CHUNK_UPDATE = "chunk_update"
    CHUNK_DELETE = "chunk_delete"
    TOPIC_CREATE = "topic_create"
    TOPIC_DELETE = "topic_delete"
    CONSOLIDATION_START = "consolidation_start"
    CONSOLIDATION_END = "consolidation_end"
    PRUNE = "prune"
    MERGE = "merge"
    VERIFICATION = "verification"
    IMPORT = "import"
    EXPORT = "export"
    SYNC_START = "sync_start"
    SYNC_END = "sync_end"


LOG_HEADER_FORMAT = "4sQ"  # magic(8), offset_count(8)
LOG_HEADER_SIZE = struct.calcsize(LOG_HEADER_FORMAT)
LOG_MAGIC = b"TMFY"


@dataclass
class LogEvent:
    """A single event in the log."""
    event_id: int = 0
    ts: str = ""
    event_type: str = ""
    topic: Optional[str] = None
    chunk_key: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    
    def to_bytes(self) -> bytes:
        """Serialize event to bytes."""
        data = {
            "ts": self.ts,
            "type": self.event_type,
            "topic": self.topic,
            "chunk": self.chunk_key,
            "details": self.details,
        }
        json_data = json.dumps(data, ensure_ascii=False).encode("utf-8")
        return json_data
    
    @classmethod
    def from_bytes(cls, event_id: int, data: bytes) -> "LogEvent":
        """Deserialize event from bytes."""
        obj = json.loads(data.decode("utf-8"))
        return cls(
            event_id=event_id,
            ts=obj.get("ts", ""),
            event_type=obj.get("type", ""),
            topic=obj.get("topic"),
            chunk_key=obj.get("chunk"),
            details=obj.get("details"),
        )


class TMFEventLog:
    """Append-only event log for TMF storage."""
    
    def __init__(self, log_path: str):
        self.log_path = Path(log_path)
        self._offset_map: Dict[int, int] = {}  # event_id -> file_offset
        self._next_id = 0
        self._load_index()
    
    def _load_index(self) -> None:
        """Load event ID index."""
        index_path = self.log_path.with_suffix(".idx")
        
        if index_path.exists():
            import json
            with open(index_path, "r") as f:
                data = json.load(f)
                self._offset_map = data.get("offsets", {})
                self._next_id = data.get("next_id", 0)
        else:
            self._next_id = self._get_next_id_from_log()
    
    def _save_index(self) -> None:
        """Save event ID index."""
        index_path = self.log_path.with_suffix(".idx")
        import json
        data = {
            "offsets": self._offset_map,
            "next_id": self._next_id,
        }
        with open(index_path, "w") as f:
            json.dump(data, f)
    
    def _get_next_id_from_log(self) -> int:
        """Scan log file to find next event ID."""
        if not self.log_path.exists():
            return 0
        
        event_id = 0
        offset = LOG_HEADER_SIZE
        
        with open(self.log_path, "rb") as f:
            while True:
                size_data = f.read(4)
                if not size_data:
                    break
                
                size = struct.unpack("I", size_data)[0]
                f.read(size)
                event_id += 1
        
        return event_id
    
    def _read_header(self) -> int:
        """Read header to get event count."""
        if not self.log_path.exists():
            return 0
        
        with open(self.log_path, "rb") as f:
            magic = f.read(8)
            if magic != LOG_MAGIC:
                return 0
            count = struct.unpack("Q", f.read(8))[0]
            return count
    
    def _write_header(self, count: int) -> None:
        """Write header with event count."""
        with open(self.log_path, "wb") as f:
            f.write(LOG_MAGIC)
            f.write(struct.pack("Q", count))
    
    def append(
        self,
        event_type: EventType,
        topic: Optional[str] = None,
        chunk_key: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Append an event. Returns event ID."""
        ts = datetime.now(timezone.utc).isoformat()
        
        event = LogEvent(
            event_id=self._next_id,
            ts=ts,
            event_type=event_type.value,
            topic=topic,
            chunk_key=chunk_key,
            details=details,
        )
        
        # Get current file position
        offset = self.log_path.stat().st_size if self.log_path.exists() else LOG_HEADER_SIZE
        
        # Write event size + data
        data = event.to_bytes()
        size = len(data)
        
        with open(self.log_path, "ab") as f:
            f.write(struct.pack("I", size))
            f.write(data)
        
        # Update index
        self._offset_map[self._next_id] = offset + 4  # +4 for size field
        self._next_id += 1
        self._save_index()
        
        return self._next_id - 1
    
    def get(self, event_id: int) -> Optional[LogEvent]:
        """Get an event by ID."""
        if event_id not in self._offset_map:
            return None
        
        offset = self._offset_map[event_id]
        
        with open(self.log_path, "rb") as f:
            f.seek(offset)
            size_data = f.read(4)
            if not size_data:
                return None
            
            size = struct.unpack("I", size_data)[0]
            data = f.read(size)
        
        return LogEvent.from_bytes(event_id, data)
    
    def get_range(self, start_id: int, end_id: int) -> List[LogEvent]:
        """Get a range of events."""
        events = []
        for event_id in range(start_id, min(end_id, self._next_id)):
            event = self.get(event_id)
            if event:
                events.append(event)
        return events
    
    def iterate_from(self, event_id: int = 0) -> Iterator[LogEvent]:
        """Iterate events from a given ID."""
        for eid in range(event_id, self._next_id):
            event = self.get(eid)
            if event:
                yield event
    
    def get_since(self, since_ts: str) -> List[LogEvent]:
        """Get all events since a timestamp."""
        events = []
        for event in self.iterate_from(0):
            if event.ts >= since_ts:
                events.append(event)
        return events
    
    def get_by_type(self, event_type: EventType) -> List[LogEvent]:
        """Get all events of a specific type."""
        events = []
        for event in self.iterate_from(0):
            if event.event_type == event_type.value:
                events.append(event)
        return events
    
    def get_by_topic(self, topic: str) -> List[LogEvent]:
        """Get all events for a topic."""
        events = []
        for event in self.iterate_from(0):
            if event.topic == topic:
                events.append(event)
        return events
    
    def get_count(self) -> int:
        """Get total event count."""
        return self._next_id
    
    def get_stats(self) -> Dict[str, Any]:
        """Get log statistics."""
        event_counts = {}
        
        for event in self.iterate_from(0):
            etype = event.event_type
            event_counts[etype] = event_counts.get(etype, 0) + 1
        
        return {
            "total_events": self._next_id,
            "event_counts": event_counts,
            "log_bytes": self.log_path.stat().st_size if self.log_path.exists() else 0,
        }
    
    def truncate(self, keep_from_id: int = 0) -> int:
        """Truncate log, keeping events from ID onwards. Returns count of removed events."""
        if keep_from_id >= self._next_id:
            return 0
        
        # This is a simplified version - real implementation would rebuild the file
        removed = self._next_id - keep_from_id
        
        # Reset index
        self._next_id = keep_from_id
        self._offset_map = {k: v for k, v in self._offset_map.items() if k >= keep_from_id}
        self._save_index()
        
        return removed
    
    def export_events(self, start_id: int = 0, end_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """Export events as JSON-serializable list."""
        end_id = end_id or self._next_id
        events = []
        
        for event_id in range(start_id, end_id):
            event = self.get(event_id)
            if event:
                events.append({
                    "event_id": event.event_id,
                    "ts": event.ts,
                    "event_type": event.event_type,
                    "topic": event.topic,
                    "chunk_key": event.chunk_key,
                    "details": event.details,
                })
        
        return events
    
    def import_events(self, events: List[Dict[str, Any]]) -> int:
        """Import events from JSON list. Returns count of imported events."""
        count = 0
        
        for evt in events:
            self.append(
                EventType(evt.get("event_type", "unknown")),
                topic=evt.get("topic"),
                chunk_key=evt.get("chunk_key"),
                details=evt.get("details"),
            )
            count += 1
        
        return count
    
    def close(self):
        """Close the log."""
        self._save_index()
