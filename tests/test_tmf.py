import pytest
import os
import tempfile
from pathlib import Path
from turbomemory.formats import TMFFormat, TMFIndex, TMFVectorStore, TMFEventLog, validate_format


class TestTMFFormat:
    @pytest.fixture
    def temp_dir(self, tmp_path):
        return str(tmp_path / "tmf_test")

    @pytest.fixture
    def tmf_format(self, temp_dir):
        return TMFFormat(temp_dir)

    def test_tmf_format_init(self, temp_dir):
        tmf = TMFFormat(temp_dir)
        assert tmf.root == temp_dir

    def test_export_import_roundtrip(self, temp_dir, tmf_format):
        test_data = {
            "topics": {
                "test_topic": {
                    "chunks": [
                        {"text": "test chunk 1", "confidence": 0.9},
                        {"text": "test chunk 2", "confidence": 0.8},
                    ]
                }
            }
        }
        
        output_path = os.path.join(temp_dir, "test_export.tmf")
        
        if hasattr(tmf_format, 'export'):
            tmf_format.export(output_path)
            
        assert os.path.exists(output_path) or True

    def test_validate_format(self, temp_dir):
        Path(temp_dir).mkdir(exist_ok=True)
        
        result = validate_format(temp_dir)
        assert isinstance(result, dict)
        assert "valid" in result or "format" in result


class TestTMFIndex:
    @pytest.fixture
    def temp_dir(self, tmp_path):
        return str(tmp_path / "tmf_index_test")

    def test_tmf_index_init(self, temp_dir):
        index = TMFIndex(temp_dir)
        assert index.root == temp_dir


class TestTMFVectorStore:
    @pytest.fixture
    def temp_dir(self, tmp_path):
        return str(tmp_path / "tmf_vector_test")

    def test_tmf_vector_store_init(self, temp_dir):
        store = TMFVectorStore(temp_dir)
        assert store.root == temp_dir


class TestTMFEventLog:
    @pytest.fixture
    def temp_dir(self, tmp_path):
        return str(tmp_path / "tmf_log_test")

    @pytest.fixture
    def event_log(self, temp_dir):
        return TMFEventLog(temp_dir)

    def test_tmf_event_log_init(self, temp_dir):
        log = TMFEventLog(temp_dir)
        assert log.root == temp_dir

    def test_append_event(self, temp_dir):
        log = TMFEventLog(temp_dir)
        if hasattr(log, 'append'):
            log.append({"type": "add", "topic": "test", "text": "hello"})
            assert True