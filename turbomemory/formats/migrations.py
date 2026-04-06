"""Schema migration manager for TMF."""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass


MIGRATION_REGISTRY: Dict[str, Callable[["MigrationManager"], bool]] = {}


def register_migration(from_version: str, to_version: str):
    """Decorator to register a migration function."""
    def decorator(func: Callable[["MigrationManager"], bool]):
        key = f"{from_version}->{to_version}"
        MIGRATION_REGISTRY[key] = func
        return func
    return decorator


@dataclass
class MigrationResult:
    """Result of a migration operation."""
    success: bool
    from_version: str
    to_version: str
    messages: List[str]
    errors: List[str]


class MigrationManager:
    """Manages schema migrations for TMF storage."""
    
    def __init__(self, root: str):
        self.root = Path(root)
        self.meta_path = self.root / "tmmeta.json"
        self.current_version = self._read_stored_version()
    
    def _read_stored_version(self) -> str:
        """Read stored schema version."""
        if not self.meta_path.exists():
            return "0.0.0"
        
        with open(self.meta_path, "r") as f:
            data = json.load(f)
            return data.get("schema_version", "0.0.0")
    
    def _write_stored_version(self, version: str) -> None:
        """Write schema version to metadata."""
        if not self.meta_path.exists():
            return
        
        with open(self.meta_path, "r") as f:
            data = json.load(f)
        
        data["schema_version"] = version
        
        with open(self.meta_path, "w") as f:
            json.dump(data, f, indent=2)
    
    def get_pending_migrations(self, target_version: str) -> List[Tuple[str, str]]:
        """Get list of pending migrations."""
        pending = []
        
        current = self.current_version
        while current != target_version:
            next_version = self._get_next_version(current)
            if next_version is None:
                break
            
            key = f"{current}->{next_version}"
            if key not in MIGRATION_REGISTRY:
                break
            
            pending.append((current, next_version))
            current = next_version
        
        return pending
    
    def _get_next_version(self, current: str) -> Optional[str]:
        """Get the next version in the sequence."""
        parts = current.split(".")
        if len(parts) != 3:
            return None
        
        major, minor, patch = parts
        
        # Simple sequential version bump
        new_patch = int(patch) + 1
        return f"{major}.{minor}.{new_patch}"
    
    def migrate(self, target_version: str) -> MigrationResult:
        """Run migrations to target version."""
        messages = []
        errors = []
        
        pending = self.get_pending_migrations(target_version)
        
        if not pending:
            return MigrationResult(
                success=True,
                from_version=self.current_version,
                to_version=target_version,
                messages=["Already at target version"],
                errors=[],
            )
        
        for from_ver, to_ver in pending:
            key = f"{from_ver}->{to_ver}"
            
            if key not in MIGRATION_REGISTRY:
                errors.append(f"No migration registered for {key}")
                continue
            
            messages.append(f"Running migration {key}...")
            
            try:
                success = MIGRATION_REGISTRY[key](self)
                if not success:
                    errors.append(f"Migration {key} failed")
                    break
                
                self._write_stored_version(to_ver)
                self.current_version = to_ver
                messages.append(f"Migration {key} completed")
                
            except Exception as e:
                errors.append(f"Migration {key} error: {str(e)}")
                break
        
        return MigrationResult(
            success=len(errors) == 0,
            from_version=self.current_version,
            to_version=target_version,
            messages=messages,
            errors=errors,
        )
    
    def can_migrate(self, target_version: str) -> bool:
        """Check if migration to target version is possible."""
        pending = self.get_pending_migrations(target_version)
        return len(pending) > 0 or self.current_version == target_version


# Register migrations
@register_migration("0.0.0", "0.0.1")
def migrate_000_to_001(manager: MigrationManager) -> bool:
    """Initial schema - no migration needed."""
    manager.root.mkdir(parents=True, exist_ok=True)
    return True


@register_migration("0.0.1", "0.0.2")
def migrate_001_to_002(manager: MigrationManager) -> bool:
    """Add quality_score field."""
    # In a real implementation, would update SQLite schema
    return True


# Main migration entry point
def migrate_to_version(root: str, target_version: str = "1.0.0") -> MigrationResult:
    """Convenience function to migrate TMF storage."""
    manager = MigrationManager(root)
    return manager.migrate(target_version)


def get_migration_status(root: str) -> Dict[str, Any]:
    """Get current migration status."""
    manager = MigrationManager(root)
    
    return {
        "current_version": manager.current_version,
        "pending_migrations": len(manager.get_pending_migrations("1.0.0")),
    }
