import pytest
import os
import tempfile
from pathlib import Path


class TestReplication:
    @pytest.fixture
    def temp_root(self, tmp_path):
        return str(tmp_path / "replication_test")

    def test_replication_import(self):
        from turbomemory.replication import TurboSync, create_sync
        assert TurboSync is not None
        assert callable(create_sync)

    def test_create_sync_init(self, temp_root):
        from turbomemory.replication import create_sync
        sync = create_sync(temp_root, "http://localhost:8001")
        assert sync.root == temp_root

    def test_turbo_sync_basic(self, temp_root):
        from turbomemory.replication import TurboSync
        
        sync = TurboSync(
            local_root=temp_root,
            remote_url="http://localhost:8001"
        )
        
        assert sync.local_root == temp_root
        assert sync.remote_url == "http://localhost:8001"


class TestSyncModule:
    def test_sync_protocol_import(self):
        from turbomemory.sync import SyncNode
        assert SyncNode is not None

    def test_sync_node_init(self, temp_root):
        from turbomemory.sync import SyncNode
        
        node = SyncNode(temp_root)
        assert node.root == temp_root

    def test_http_sync_import(self):
        from turbomemory.sync.http_sync import HTTPSyncClient, HTTPSyncServer
        assert HTTPSyncClient is not None
        assert HTTPSyncServer is not None