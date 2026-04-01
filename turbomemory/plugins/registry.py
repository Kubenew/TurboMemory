"""Plugin registry for managing TurboMemory plugins."""

import importlib
import importlib.util
import os
import logging
from typing import Dict, Type, Optional, Any
from .base import Plugin, QualityScorer, EmbeddingProvider, StorageBackend, VerificationStrategy

logger = logging.getLogger(__name__)


class PluginRegistry:
    """Registry for loading and managing TurboMemory plugins."""

    _instance = None
    _plugins: Dict[str, Plugin] = {}
    _plugin_types: Dict[str, Type[Plugin]] = {
        "quality_scorer": QualityScorer,
        "embedding_provider": EmbeddingProvider,
        "storage_backend": StorageBackend,
        "verification_strategy": VerificationStrategy,
    }

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def register(cls, plugin: Plugin) -> None:
        """Register a plugin instance."""
        cls._plugins[plugin.name] = plugin
        plugin.initialize()
        logger.info(f"Registered plugin: {plugin.name} v{plugin.version}")

    @classmethod
    def unregister(cls, name: str) -> None:
        """Unregister a plugin."""
        if name in cls._plugins:
            cls._plugins[name].cleanup()
            del cls._plugins[name]
            logger.info(f"Unregistered plugin: {name}")

    @classmethod
    def get(cls, name: str) -> Optional[Plugin]:
        """Get a registered plugin by name."""
        return cls._plugins.get(name)

    @classmethod
    def get_by_type(cls, plugin_type: str) -> Optional[Plugin]:
        """Get the first plugin of a given type."""
        for plugin in cls._plugins.values():
            if isinstance(plugin, cls._plugin_types.get(plugin_type, Plugin)):
                return plugin
        return None

    @classmethod
    def list_plugins(cls) -> Dict[str, Dict[str, Any]]:
        """List all registered plugins."""
        return {
            name: {
                "version": plugin.version,
                "type": type(plugin).__name__,
                "description": plugin.description,
            }
            for name, plugin in cls._plugins.items()
        }

    @classmethod
    def load_from_module(cls, module_path: str) -> None:
        """Load a plugin from a Python module file."""
        if not os.path.exists(module_path):
            raise FileNotFoundError(f"Plugin module not found: {module_path}")

        spec = importlib.util.spec_from_file_location("plugin_module", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, type) and issubclass(attr, Plugin) and attr != Plugin:
                plugin = attr()
                cls.register(plugin)

    @classmethod
    def load_from_entry_point(cls, entry_point: str) -> None:
        """Load a plugin from an entry point string (module:attribute)."""
        module_path, attr_name = entry_point.split(":")
        module = importlib.import_module(module_path)
        plugin_class = getattr(module, attr_name)

        if isinstance(plugin_class, type) and issubclass(plugin_class, Plugin):
            plugin = plugin_class()
            cls.register(plugin)
        else:
            raise TypeError(f"{entry_point} is not a Plugin subclass")

    @classmethod
    def clear(cls) -> None:
        """Clear all registered plugins."""
        for plugin in cls._plugins.values():
            plugin.cleanup()
        cls._plugins.clear()
