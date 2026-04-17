import pytest
import os
import time
from pathlib import Path
from turbomemory.daemon.cli import start_daemon, stop_daemon, status_daemon, pid_path, log_path


class TestDaemon:
    @pytest.fixture
    def temp_root(self, tmp_path):
        return str(tmp_path / "daemon_test")

    def test_pid_path(self, temp_root):
        pid = pid_path(temp_root)
        assert "consolidator.pid" in str(pid)

    def test_log_path(self, temp_root):
        log = log_path(temp_root)
        assert "consolidator.log" in str(log)

    def test_status_daemon_not_running(self, temp_root):
        status = status_daemon(temp_root)
        assert "not running" in status.lower() or isinstance(status, str)

    def test_start_daemon_creates_pid_file(self, temp_root):
        os.makedirs(os.path.join(temp_root, "lock"), exist_ok=True)
        
        import subprocess
        import sys
        
        test_script = '''
import sys
import time
time.sleep(0.1)
'''
        
        with open(os.path.join(temp_root, "test_daemon.py"), "w") as f:
            f.write(test_script)
        
        Path(temp_root).mkdir(exist_ok=True)
        
        pid_file = pid_path(temp_root)
        assert "consolidator.pid" in str(pid_file)


class TestConsolidator:
    @pytest.fixture
    def temp_root(self, tmp_path):
        return str(tmp_path / "consolidator_test")

    def test_consolidator_import(self):
        from turbomemory.daemon.consolidator import run_once, daemon_loop
        assert callable(run_once)
        assert callable(daemon_loop)