"""Replication protocol for TurboMemory."""

import json
import asyncio
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path


class SyncAction(Enum):
    """Actions in replication protocol."""
    PUSH = "push"
    PULL = "pull"
    FULL_SYNC = "full_sync"
    INCREMENTAL = "incremental"


class SyncState(Enum):
    """Sync state machine."""
    IDLE = "idle"
    CONNECTING = "connecting"
    SYNCING = "syncing"
    ERROR = "error"
    COMPLETE = "complete"


@dataclass
class SyncRequest:
    """Sync request message."""
    action: SyncAction
    from_event_id: int = 0
    to_event_id: Optional[int] = None
    topics: Optional[List[str]] = None


@dataclass
class SyncResponse:
    """Sync response message."""
    success: bool
    event_count: int
    event_id_start: int
    event_id_end: int
    topics_synced: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


@dataclass
class NodeInfo:
    """Information about a sync node."""
    node_id: str
    host: str
    port: int
    last_event_id: int = 0
    last_sync: str = ""
    status: str = "online"
    capabilities: List[str] = field(default_factory=list)


@dataclass
class SyncConfig:
    """Configuration for replication."""
    node_id: str
    peer_nodes: List[str] = field(default_factory=list)
    sync_interval_sec: int = 300
    conflict_policy: str = "append"  # "append", "merge", "latest"
    compression: bool = True
    encryption: bool = False


class ReplicationProtocol:
    """Handles replication protocol between nodes."""
    
    def __init__(self, config: SyncConfig):
        self.config = config
        self.state = SyncState.IDLE
        self.last_error: Optional[str] = None
        
        self._event_log = None
        self._local_event_id = 0
    
    def set_event_log(self, event_log):
        """Set the event log to use for replication."""
        self._event_log = event_log
        if event_log:
            self._local_event_id = event_log.get_count()
    
    async def push_events(
        self,
        peer_url: str,
        from_event_id: int = 0,
    ) -> SyncResponse:
        """Push events to a peer node."""
        if not self._event_log:
            return SyncResponse(
                success=False,
                event_count=0,
                event_id_start=0,
                event_id_end=0,
                errors=["No event log configured"],
            )
        
        # Get events to push
        events = self._event_log.export_events(from_event_id)
        
        if not events:
            return SyncResponse(
                success=True,
                event_count=0,
                event_id_start=from_event_id,
                event_id_end=from_event_id,
            )
        
        # Send to peer (placeholder - implement HTTP)
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{peer_url}/sync/push",
                    json={"events": events},
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        return SyncResponse(
                            success=True,
                            event_count=len(events),
                            event_id_start=from_event_id,
                            event_id_end=from_event_id + len(events),
                            topics_synced=result.get("topics", []),
                        )
                    else:
                        return SyncResponse(
                            success=False,
                            event_count=0,
                            event_id_start=from_event_id,
                            event_id_end=from_event_id,
                            errors=[f"HTTP {resp.status}"],
                        )
        except ImportError:
            # aiohttp not installed, use sync fallback
            return await self._push_events_sync(peer_url, events, from_event_id)
        except Exception as e:
            return SyncResponse(
                success=False,
                event_count=0,
                event_id_start=from_event_id,
                event_id_end=from_event_id,
                errors=[str(e)],
            )
    
    async def _push_events_sync(
        self,
        peer_url: str,
        events: List[Dict],
        from_event_id: int,
    ) -> SyncResponse:
        """Synchronous fallback for pushing events."""
        import requests
        
        try:
            resp = requests.post(
                f"{peer_url}/sync/push",
                json={"events": events},
                timeout=30,
            )
            
            if resp.status_code == 200:
                result = resp.json()
                return SyncResponse(
                    success=True,
                    event_count=len(events),
                    event_id_start=from_event_id,
                    event_id_end=from_event_id + len(events),
                    topics_synced=result.get("topics", []),
                )
            else:
                return SyncResponse(
                    success=False,
                    event_count=0,
                    event_id_start=from_event_id,
                    event_id_end=from_event_id,
                    errors=[f"HTTP {resp.status_code}"],
                )
        except Exception as e:
            return SyncResponse(
                success=False,
                event_count=0,
                event_id_start=from_event_id,
                event_id_end=from_event_id,
                errors=[str(e)],
            )
    
    async def pull_events(
        self,
        peer_url: str,
        from_event_id: int = 0,
    ) -> SyncResponse:
        """Pull events from a peer node."""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{peer_url}/sync/pull",
                    params={"from_event_id": from_event_id},
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        events = data.get("events", [])
                        
                        # Import events
                        imported = self._event_log.import_events(events)
                        
                        return SyncResponse(
                            success=True,
                            event_count=imported,
                            event_id_start=from_event_id,
                            event_id_end=from_event_id + imported,
                        )
                    else:
                        return SyncResponse(
                            success=False,
                            event_count=0,
                            event_id_start=from_event_id,
                            event_id_end=from_event_id,
                            errors=[f"HTTP {resp.status}"],
                        )
        except ImportError:
            return await self._pull_events_sync(peer_url, from_event_id)
        except Exception as e:
            return SyncResponse(
                success=False,
                event_count=0,
                event_id_start=from_event_id,
                event_id_end=from_event_id,
                errors=[str(e)],
            )
    
    async def _pull_events_sync(self, peer_url: str, from_event_id: int) -> SyncResponse:
        """Synchronous fallback for pulling events."""
        import requests
        
        try:
            resp = requests.get(
                f"{peer_url}/sync/pull",
                params={"from_event_id": from_event_id},
                timeout=30,
            )
            
            if resp.status_code == 200:
                data = resp.json()
                events = data.get("events", [])
                imported = self._event_log.import_events(events)
                
                return SyncResponse(
                    success=True,
                    event_count=imported,
                    event_id_start=from_event_id,
                    event_id_end=from_event_id + imported,
                )
            else:
                return SyncResponse(
                    success=False,
                    event_count=0,
                    event_id_start=from_event_id,
                    event_id_end=from_event_id,
                    errors=[f"HTTP {resp.status_code}"],
                )
        except Exception as e:
            return SyncResponse(
                success=False,
                event_count=0,
                event_id_start=from_event_id,
                event_id_end=from_event_id,
                errors=[str(e)],
            )
    
    async def full_sync(self, peer_url: str) -> SyncResponse:
        """Perform full sync with a peer."""
        self.state = SyncState.SYNCING
        
        try:
            # First push our events
            push_result = await self.push_events(peer_url, from_event_id=0)
            
            # Then pull their events
            pull_result = await self.pull_events(peer_url, from_event_id=0)
            
            self.state = SyncState.COMPLETE
            
            return SyncResponse(
                success=push_result.success and pull_result.success,
                event_count=push_result.event_count + pull_result.event_count,
                event_id_start=0,
                event_id_end=self._local_event_id,
                topics_synced=push_result.topics_synced + pull_result.topics_synced,
                errors=push_result.errors + pull_result.errors,
            )
        
        except Exception as e:
            self.state = SyncState.ERROR
            self.last_error = str(e)
            return SyncResponse(
                success=False,
                event_count=0,
                event_id_start=0,
                event_id_end=0,
                errors=[str(e)],
            )
    
    async def sync_loop(self, peer_urls: List[str]):
        """Background sync loop."""
        while True:
            for peer_url in peer_urls:
                result = await self.full_sync(peer_url)
                if result.success:
                    print(f"Synced with {peer_url}: {result.event_count} events")
                else:
                    print(f"Sync failed with {peer_url}: {result.errors}")
            
            await asyncio.sleep(self.config.sync_interval_sec)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current sync status."""
        return {
            "node_id": self.config.node_id,
            "state": self.state.value,
            "last_event_id": self._local_event_id,
            "last_error": self.last_error,
            "peer_nodes": self.config.peer_nodes,
        }
