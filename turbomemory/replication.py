"""Log-based replication system for TurboMemory - Git-for-memory."""

import os
import json
import requests
import threading
from typing import Optional, Dict, Any, List, Callable
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timezone

from .formats.tmf_log import TMFEventLog, EventType


@dataclass
class SyncConfig:
    """Configuration for sync/replication."""
    remote_url: str = ""
    sync_interval: int = 60  # seconds
    conflict_policy: str = "last_write_wins"  # or "merge", "manual"
    auto_encrypt: bool = False
    compression: bool = True


class ReplicationLog:
    """Manages replication state and sync operations."""
    
    def __init__(self, log_path: str):
        self.log = TMFEventLog(log_path)
        self._sync_state: Dict[str, Any] = {}
        self._load_state()
    
    def _load_state(self) -> None:
        state_path = Path(self.log.log_path).with_suffix(".sync")
        if state_path.exists():
            with open(state_path, "r") as f:
                self._sync_state = json.load(f)
    
    def _save_state(self) -> None:
        state_path = Path(self.log.log_path).with_suffix(".sync")
        with open(state_path, "w") as f:
            json.dump(self._sync_state, f)
    
    def record_sync(self, remote: str, event_id: int, success: bool) -> None:
        """Record sync operation."""
        if remote not in self._sync_state:
            self._sync_state[remote] = {}
        
        self._sync_state[remote]["last_sync"] = datetime.now(timezone.utc).isoformat()
        self._sync_state[remote]["last_event_id"] = event_id
        self._sync_state[remote]["last_success"] = success
        self._save_state()
    
    def get_sync_state(self, remote: str) -> Dict[str, Any]:
        """Get sync state for remote."""
        return self._sync_state.get(remote, {})


class TurboSync:
    """Main sync engine - pulls missing events from remote."""
    
    def __init__(self, local_path: str, config: SyncConfig):
        self.local_path = Path(local_path)
        self.config = config
        self.log_path = self.local_path / "events.tmlog"
        self.repl_log = ReplicationLog(str(self.log_path))
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._callbacks: List[Callable] = []
    
    def register_callback(self, callback: Callable) -> None:
        """Register sync callback."""
        self._callbacks.append(callback)
    
    def pull(self, since_event_id: int = 0) -> Dict[str, Any]:
        """Pull missing events from remote."""
        if not self.config.remote_url:
            return {"error": "No remote URL configured"}
        
        try:
            # Fetch events from remote
            response = requests.get(
                f"{self.config.remote_url}/events",
                params={"since": since_event_id},
                timeout=30
            )
            
            if response.status_code != 200:
                return {"error": f"Remote returned {response.status_code}"}
            
            remote_events = response.json().get("events", [])
            
            # Apply events locally
            imported = 0
            for evt in remote_events:
                self.repl_log.log.import_events([evt])
                imported += 1
            
            return {
                "imported": imported,
                "latest_event_id": since_event_id + imported,
            }
        except Exception as e:
            return {"error": str(e)}
    
    def push(self) -> Dict[str, Any]:
        """Push local events to remote."""
        if not self.config.remote_url:
            return {"error": "No remote URL configured"}
        
        try:
            # Get local events
            events = self.repl_log.log.export_events()
            
            # Push to remote
            response = requests.post(
                f"{self.config.remote_url}/events",
                json={"events": events},
                timeout=30
            )
            
            if response.status_code != 200:
                return {"error": f"Remote returned {response.status_code}"}
            
            return {"pushed": len(events)}
        except Exception as e:
            return {"error": str(e)}
    
    def sync(self) -> Dict[str, Any]:
        """Full bidirectional sync."""
        # First pull remote changes
        state = self.repl_log.get_sync_state(self.config.remote_url)
        last_event = state.get("last_event_id", 0)
        
        pull_result = self.pull(since_event_id=last_event)
        
        # Then push local changes
        push_result = self.push()
        
        result = {
            "pull": pull_result,
            "push": push_result,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
        # Notify callbacks
        for cb in self._callbacks:
            try:
                cb(result)
            except Exception:
                pass
        
        return result
    
    def start_auto_sync(self) -> None:
        """Start automatic sync in background."""
        self._running = True
        
        def run_loop():
            while self._running:
                self.sync()
                import time
                time.sleep(self.config.sync_interval)
        
        self._thread = threading.Thread(target=run_loop, daemon=True)
        self._thread.start()
    
    def stop_auto_sync(self) -> None:
        """Stop automatic sync."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)


def create_sync(root: str, remote_url: str, **kwargs) -> TurboSync:
    """Create a new sync instance."""
    config = SyncConfig(remote_url=remote_url, **kwargs)
    return TurboSync(root, config)