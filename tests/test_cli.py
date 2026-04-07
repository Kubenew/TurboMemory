"""Tests for CLI commands."""

import pytest
import os
import json
import subprocess
import sys
import tempfile
import shutil


class TestCLI:
    """Test CLI commands."""
    
    @pytest.fixture
    def temp_dir(self):
        d = tempfile.mkdtemp()
        yield d
        shutil.rmtree(d, ignore_errors=True)
    
    def test_add_memory_cli(self, temp_dir):
        result = subprocess.run(
            [sys.executable, "-m", "turbomemory", "add_memory",
             "--topic", "test",
             "--text", "CLI test content",
             "--root", temp_dir],
            capture_output=True,
            text=True,
            timeout=60
        )
        assert result.returncode == 0 or "Added memory" in result.stdout or "chunk_id" in result.stdout
    
    def test_stats_cli(self, temp_dir):
        # First add some data
        subprocess.run(
            [sys.executable, "-m", "turbomemory", "add_memory",
             "--topic", "test", "--text", "Test content", "--root", temp_dir],
            capture_output=True,
            timeout=60
        )
        
        # Then get stats
        result = subprocess.run(
            [sys.executable, "-m", "turbomemory", "stats", "--root", temp_dir],
            capture_output=True,
            text=True,
            timeout=60
        )
        assert result.returncode == 0
    
    def test_query_cli(self, temp_dir):
        # First add data
        subprocess.run(
            [sys.executable, "-m", "turbomemory", "add_memory",
             "--topic", "test", "--text": "Test content", "--root", temp_dir],
            capture_output=True,
            timeout=60
        )
        
        # Query
        result = subprocess.run(
            [sys.executable, "-m", "turbomemory", "query",
             "--query", "test", "--root", temp_dir],
            capture_output=True,
            text=True,
            timeout=60
        )
        assert result.returncode == 0
    
    def test_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "turbomemory", "--help"],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode == 0
        assert "add_memory" in result.stdout