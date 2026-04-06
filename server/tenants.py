"""Multi-tenant management."""

from typing import Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone
import threading


@dataclass
class Tenant:
    """Represents a tenant in the system."""
    tenant_id: str
    created: str
    memory_root: str
    api_key: Optional[str] = None
    rate_limit: int = 1000
    enabled: bool = True
    metadata: Dict = field(default_factory=dict)


class TenantManager:
    """Manages multiple tenants."""
    
    def __init__(self):
        self._tenants: Dict[str, Tenant] = {}
        self._memories: Dict[str, any] = {}
        self._lock = threading.Lock()
    
    def create_tenant(
        self,
        tenant_id: str,
        memory_root: str,
        api_key: Optional[str] = None,
        **metadata,
    ) -> Tenant:
        """Create a new tenant."""
        with self._lock:
            if tenant_id in self._tenants:
                raise ValueError(f"Tenant {tenant_id} already exists")
            
            tenant = Tenant(
                tenant_id=tenant_id,
                created=datetime.now(timezone.utc).isoformat(),
                memory_root=memory_root,
                api_key=api_key,
                metadata=metadata,
            )
            
            self._tenants[tenant_id] = tenant
            return tenant
    
    def get_tenant(self, tenant_id: str) -> Optional[Tenant]:
        """Get a tenant by ID."""
        return self._tenants.get(tenant_id)
    
    def delete_tenant(self, tenant_id: str) -> bool:
        """Delete a tenant."""
        with self._lock:
            if tenant_id not in self._tenants:
                return False
            
            # Close memory if open
            if tenant_id in self._memories:
                self._memories[tenant_id].close()
                del self._memories[tenant_id]
            
            del self._tenants[tenant_id]
            return True
    
    def list_tenants(self) -> Dict[str, Tenant]:
        """List all tenants."""
        return dict(self._tenants)
    
    def get_memory(self, tenant_id: str):
        """Get or create memory instance for tenant."""
        with self._lock:
            if tenant_id not in self._memories:
                tenant = self._tenants.get(tenant_id)
                if not tenant:
                    raise ValueError(f"Tenant {tenant_id} not found")
                
                from turbomemory import TurboMemory
                self._memories[tenant_id] = TurboMemory(root=tenant.memory_root)
            
            return self._memories[tenant_id]
    
    def list_memories(self):
        """List all memory instances."""
        return list(self._memories.values())
    
    def close_all(self):
        """Close all memory instances."""
        for memory in self._memories.values():
            memory.close()
        self._memories.clear()
