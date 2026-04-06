"""HTTP-based sync protocol for TurboMemory."""

import asyncio
import json
import hashlib
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

logger = logging.getLogger(__name__)


class SyncDirection(Enum):
    """Sync direction."""
    PUSH = "push"
    PULL = "pull"
    BIDIRECTIONAL = "bidirectional"


@dataclass
class SyncEndpoint:
    """Represents a sync peer endpoint."""
    url: str
    api_key: Optional[str] = None
    timeout_sec: int = 30
    retry_count: int = 3
    retry_delay_sec: float = 1.0


@dataclass
class SyncResult:
    """Result of a sync operation."""
    success: bool
    events_pushed: int = 0
    events_pulled: int = 0
    bytes_transferred: int = 0
    duration_ms: int = 0
    errors: List[str] = field(default_factory=list)
    peer_info: Optional[Dict] = None


class HTTPSyncClient:
    """HTTP-based sync client for peer-to-peer replication."""
    
    def __init__(
        self,
        local_node_id: str,
        endpoint: SyncEndpoint,
        compression: bool = True,
        encryption_key: Optional[str] = None,
    ):
        self.local_node_id = local_node_id
        self.endpoint = endpoint
        self.compression = compression
        self.encryption_key = encryption_key
        self._session: Optional[Any] = None
    
    async def _get_session(self) -> Any:
        """Get or create aiohttp session."""
        if self._session is None:
            import aiohttp
            headers = {}
            if self.endpoint.api_key:
                headers["X-API-Key"] = self.endpoint.api_key
            
            self._session = aiohttp.ClientSession(
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.endpoint.timeout_sec),
            )
        return self._session
    
    async def close(self):
        """Close the HTTP session."""
        if self._session:
            await self._session.close()
            self._session = None
    
    async def handshake(self) -> Dict[str, Any]:
        """Perform handshake with peer node."""
        try:
            session = await self._get_session()
            async with session.get(f"{self.endpoint.url}/sync/handshake") as resp:
                if resp.status == 200:
                    return await resp.json()
                else:
                    return {"error": f"HTTP {resp.status}"}
        except Exception as e:
            return {"error": str(e)}
    
    async def push_events(
        self,
        events: List[Dict],
        from_event_id: int = 0,
        topic_filter: Optional[List[str]] = None,
    ) -> SyncResult:
        """Push events to peer node."""
        start_time = datetime.now(timezone.utc)
        
        payload = {
            "node_id": self.local_node_id,
            "from_event_id": from_event_id,
            "events": events,
            "topic_filter": topic_filter or [],
            "timestamp": start_time.isoformat(),
        }
        
        if self.compression:
            import gzip
            payload_str = json.dumps(payload)
            compressed = gzip.compress(payload_str.encode())
            payload = {"compressed": True, "data": compressed.hex()}
        
        for attempt in range(self.endpoint.retry_count):
            try:
                session = await self._get_session()
                async with session.post(
                    f"{self.endpoint.url}/sync/push",
                    json=payload,
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        duration_ms = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
                        return SyncResult(
                            success=True,
                            events_pushed=len(events),
                            bytes_transferred=result.get("bytes_processed", 0),
                            duration_ms=duration_ms,
                            peer_info=result.get("peer_info"),
                        )
                    else:
                        error_msg = f"HTTP {resp.status}"
                        if attempt < self.endpoint.retry_count - 1:
                            await asyncio.sleep(self.endpoint.retry_delay_sec * (attempt + 1))
                            continue
                        return SyncResult(success=False, errors=[error_msg])
            except Exception as e:
                if attempt < self.endpoint.retry_count - 1:
                    await asyncio.sleep(self.endpoint.retry_delay_sec * (attempt + 1))
                    continue
                return SyncResult(success=False, errors=[str(e)])
        
        return SyncResult(success=False, errors=["Max retries exceeded"])
    
    async def pull_events(
        self,
        from_event_id: int = 0,
        to_event_id: Optional[int] = None,
        topic_filter: Optional[List[str]] = None,
    ) -> SyncResult:
        """Pull events from peer node."""
        start_time = datetime.now(timezone.utc)
        
        params = {
            "from_event_id": from_event_id,
            "node_id": self.local_node_id,
        }
        if to_event_id:
            params["to_event_id"] = to_event_id
        if topic_filter:
            params["topic_filter"] = ",".join(topic_filter)
        
        for attempt in range(self.endpoint.retry_count):
            try:
                session = await self._get_session()
                async with session.get(
                    f"{self.endpoint.url}/sync/pull",
                    params=params,
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        events = data.get("events", [])
                        duration_ms = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
                        return SyncResult(
                            success=True,
                            events_pulled=len(events),
                            bytes_transferred=data.get("bytes_sent", 0),
                            duration_ms=duration_ms,
                            peer_info=data.get("peer_info"),
                        )
                    else:
                        error_msg = f"HTTP {resp.status}"
                        if attempt < self.endpoint.retry_count - 1:
                            await asyncio.sleep(self.endpoint.retry_delay_sec * (attempt + 1))
                            continue
                        return SyncResult(success=False, errors=[error_msg])
            except Exception as e:
                if attempt < self.endpoint.retry_count - 1:
                    await asyncio.sleep(self.endpoint.retry_delay_sec * (attempt + 1))
                    continue
                return SyncResult(success=False, errors=[str(e)])
        
        return SyncResult(success=False, errors=["Max retries exceeded"])
    
    async def full_sync(
        self,
        from_event_id: int = 0,
        topic_filter: Optional[List[str]] = None,
    ) -> SyncResult:
        """Perform full bidirectional sync."""
        results: List[SyncResult] = []
        
        push_result = await self.push_events(
            events=[],  # Events would come from local event log
            from_event_id=from_event_id,
            topic_filter=topic_filter,
        )
        results.append(push_result)
        
        pull_result = await self.pull_events(
            from_event_id=from_event_id,
            topic_filter=topic_filter,
        )
        results.append(pull_result)
        
        return SyncResult(
            success=all(r.success for r in results),
            events_pushed=sum(r.events_pushed for r in results),
            events_pulled=sum(r.events_pulled for r in results),
            bytes_transferred=sum(r.bytes_transferred for r in results),
            duration_ms=sum(r.duration_ms for r in results),
            errors=[e for r in results for e in r.errors],
        )


class HTTPSyncServer:
    """HTTP server for handling sync requests."""
    
    def __init__(self, node_id: str, api_key: Optional[str] = None):
        self.node_id = node_id
        self.api_key = api_key
        self._event_log = None
        self._memory = None
    
    def set_backend(self, event_log, memory):
        """Set the backend for sync operations."""
        self._event_log = event_log
        self._memory = memory
    
    def _verify_api_key(self, headers: Dict) -> bool:
        """Verify API key from request headers."""
        if not self.api_key:
            return True
        provided_key = headers.get("X-API-Key")
        return provided_key == self.api_key
    
    async def handle_handshake(self, request: Dict) -> Dict:
        """Handle handshake request."""
        return {
            "node_id": self.node_id,
            "version": "0.8",
            "capabilities": ["push", "pull", "full_sync", "compression"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    
    async def handle_push(self, payload: Dict) -> Dict:
        """Handle push request."""
        if self._event_log is None:
            return {"error": "No event log configured", "success": False}
        
        events = payload.get("events", [])
        from_event_id = payload.get("from_event_id", 0)
        
        if not events:
            return {"success": True, "events_imported": 0, "bytes_processed": 0}
        
        imported = self._event_log.import_events(events)
        
        return {
            "success": True,
            "events_imported": imported,
            "bytes_processed": len(json.dumps(events)),
            "peer_info": {
                "node_id": payload.get("node_id"),
                "from_event_id": from_event_id,
            },
        }
    
    async def handle_pull(self, params: Dict) -> Dict:
        """Handle pull request."""
        if self._event_log is None:
            return {"error": "No event log configured", "success": False}
        
        from_event_id = int(params.get("from_event_id", 0))
        to_event_id = params.get("to_event_id")
        topic_filter = params.get("topic_filter", "").split(",") if params.get("topic_filter") else None
        
        events = self._event_log.export_events(from_event_id)
        
        if topic_filter:
            events = [e for e in events if e.get("topic") in topic_filter]
        
        if to_event_id:
            events = [e for e in events if e.get("event_id", 0) <= int(to_event_id)]
        
        return {
            "success": True,
            "events": events,
            "count": len(events),
            "bytes_sent": len(json.dumps(events)),
            "peer_info": {
                "node_id": params.get("node_id"),
                "from_event_id": from_event_id,
            },
        }


class SyncManager:
    """Manages multiple sync peers and coordination."""
    
    def __init__(
        self,
        node_id: str,
        event_log,
        memory,
        api_key: Optional[str] = None,
    ):
        self.node_id = node_id
        self._event_log = event_log
        self._memory = memory
        self.api_key = api_key
        self._clients: Dict[str, HTTPSyncClient] = {}
        self._server = HTTPSyncServer(node_id, api_key)
        self._server.set_backend(event_log, memory)
    
    def add_peer(self, peer_url: str, api_key: Optional[str] = None):
        """Add a peer for sync."""
        endpoint = SyncEndpoint(url=peer_url, api_key=api_key or self.api_key)
        client = HTTPSyncClient(self.node_id, endpoint)
        self._clients[peer_url] = client
    
    def remove_peer(self, peer_url: str):
        """Remove a peer."""
        if peer_url in self._clients:
            asyncio.create_task(self._clients[peer_url].close())
            del self._clients[peer_url]
    
    async def sync_with_peer(self, peer_url: str) -> SyncResult:
        """Sync with a specific peer."""
        if peer_url not in self._clients:
            return SyncResult(success=False, errors=["Peer not found"])
        
        client = self._clients[peer_url]
        
        from_event_id = 0
        if self._event_log:
            from_event_id = self._event_log.get_count()
        
        return await client.full_sync(from_event_id=from_event_id)
    
    async def sync_all_peers(self) -> Dict[str, SyncResult]:
        """Sync with all configured peers."""
        results = {}
        for peer_url in self._clients:
            results[peer_url] = await self.sync_with_peer(peer_url)
        return results
    
    async def start_background_sync(self, interval_sec: int = 300):
        """Start background sync loop."""
        while True:
            await self.sync_all_peers()
            await asyncio.sleep(interval_sec)
    
    async def close(self):
        """Close all client connections."""
        for client in self._clients.values():
            await client.close()
        self._clients.clear()