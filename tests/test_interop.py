"""Tests for export/import functionality."""

import pytest
import os
import tempfile
import shutil

try:
    import pyarrow as pa
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False

try:
    import lance
    HAS_LANCE = True
except ImportError:
    HAS_LANCE = False


@pytest.mark.skipif(not HAS_PYARROW, reason="pyarrow not installed")
class TestParquetExport:
    """Test Parquet export/import."""
    
    @pytest.fixture
    def temp_dir(self):
        d = tempfile.mkdtemp()
        yield d
        shutil.rmtree(d, ignore_errors=True)
    
    def test_export_parquet_full(self, temp_dir):
        from turbomemory import TurboMemory
        from turbomemory.interop import export_to_parquet
        
        tm = TurboMemory(root=temp_dir, model_name="all-MiniLM-L6-v2")
        tm.add_memory("test", "Test document for export")
        
        output_path = os.path.join(temp_dir, "export.parquet")
        result = export_to_parquet(temp_dir, output_path, format="full")
        
        assert result["exported"] >= 1
        assert os.path.exists(output_path)
        
        tm.close()
    
    def test_export_parquet_quantized(self, temp_dir):
        from turbomemory import TurboMemory
        from turbomemory.interop import export_to_parquet
        
        tm = TurboMemory(root=temp_dir, model_name="all-MiniLM-L6-v2")
        tm.add_memory("test", "Test document for export")
        
        output_path = os.path.join(temp_dir, "export_quantized.parquet")
        result = export_to_parquet(temp_dir, output_path, format="quantized")
        
        assert result["exported"] >= 1
        assert os.path.exists(output_path)
        
        tm.close()


@pytest.mark.skipif(not HAS_LANCE, reason="lance not installed")
class TestLanceExport:
    """Test Lance export/import."""
    
    @pytest.fixture
    def temp_dir(self):
        d = tempfile.mkdtemp()
        yield d
        shutil.rmtree(d, ignore_errors=True)
    
    def test_export_lance_full(self, temp_dir):
        from turbomemory import TurboMemory
        from turbomemory.interop import export_to_lance
        
        tm = TurboMemory(root=temp_dir, model_name="all-MiniLM-L6-v2")
        tm.add_memory("test", "Test document for export")
        
        lance_path = os.path.join(temp_dir, "lance_dataset")
        result = export_to_lance(temp_dir, lance_path, format="full")
        
        assert result["exported"] >= 1
        
        tm.close()
    
    def test_export_lance_append(self, temp_dir):
        from turbomemory import TurboMemory
        from turbomemory.interop import export_to_lance
        
        tm = TurboMemory(root=temp_dir, model_name="all-MiniLM-L6-v2")
        tm.add_memory("test", "Test document 1")
        
        lance_path = os.path.join(temp_dir, "lance_dataset")
        result = export_to_lance(temp_dir, lance_path, format="full")
        
        tm.add_memory("test", "Test document 2")
        
        # Append mode
        result = export_to_lance(tm.root, lance_path, mode="append", format="full")
        
        assert result["exported"] >= 2
        
        tm.close()


@pytest.mark.skipif(not HAS_PYARROW, reason="pyarrow not installed")
class TestRoundTrip:
    """Test round-trip export/import."""
    
    @pytest.fixture
    def temp_dir(self):
        d = tempfile.mkdtemp()
        yield d
        shutil.rmtree(d, ignore_errors=True)
    
    def test_parquet_roundtrip_quality(self, temp_dir):
        """Test that vectors maintain similarity after round-trip."""
        from turbomemory import TurboMemory
        from turbomemory.interop import export_to_parquet, import_from_parquet
        from turbomemory.retrieval import cosine_similarity
        
        tm = TurboMemory(root=temp_dir, model_name="all-MiniLM-L6-v2")
        
        # Add original document
        tm.add_memory("test", "Python is a programming language")
        
        # Get original query result
        results_before = tm.query("programming", k=1)
        
        # Export to Parquet
        output_path = os.path.join(temp_dir, "test.parquet")
        export_to_parquet(temp_dir, output_path, format="full")
        
        # Import back
        import_from_parquet(temp_dir, output_path, topic="imported")
        
        # Query again
        results_after = tm.query("programming", k=1)
        
        # Check similarity preserved
        if results_before and results_after:
            score_before = results_before[0][0]
            score_after = results_after[0][0]
            # Allow small variance
            assert abs(score_before - score_after) < 0.1
        
        tm.close()
