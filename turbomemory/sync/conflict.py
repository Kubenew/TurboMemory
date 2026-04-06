"""Conflict resolution for replication."""

from typing import Dict, List, Any, Tuple
from enum import Enum


class ConflictPolicy(Enum):
    APPEND = "append"
    MERGE = "merge"
    LATEST = "latest"
    SOURCE_WINS = "source_wins"


class ConflictResolver:
    def __init__(self, policy: ConflictPolicy = ConflictPolicy.APPEND):
        self.policy = policy
    
    def resolve(
        self,
        local_event: Dict[str, Any],
        remote_event: Dict[str, Any],
    ) -> Tuple[bool, str]:
        """Resolve conflict between local and remote events.
        
        Returns: (use_local, reason)
        """
        if self.policy == ConflictPolicy.APPEND:
            # Always use local, skip remote
            return True, "append_policy"
        
        elif self.policy == ConflictPolicy.LATEST:
            # Use the more recent one
            local_ts = local_event.get("ts", "")
            remote_ts = remote_event.get("ts", "")
            
            if remote_ts > local_ts:
                return False, "remote_newer"
            return True, "local_newer"
        
        elif self.policy == ConflictPolicy.LATEST:
            # Simple merge - keep both if different, prefer local
            return True, "merge_policy"
        
        return True, "default"
