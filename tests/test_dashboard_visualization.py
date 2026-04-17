import pytest


class TestDashboard:
    def test_dashboard_import(self):
        from turbomemory.dashboard import load_memory
        assert callable(load_memory)

    def test_dashboard_renders(self):
        import streamlit as st
        
        st.set_page_config = lambda **kwargs: None
        
        from turbomemory.dashboard import (
            render_overview,
            render_browse_topics,
            render_search,
            render_add_memory,
            render_consolidate,
            render_metrics,
        )
        
        assert callable(render_overview)
        assert callable(render_browse_topics)
        assert callable(render_search)
        assert callable(render_add_memory)
        assert callable(render_consolidate)
        assert callable(render_metrics)


class TestVisualization:
    def test_visualization_import(self):
        from turbomemory.visualization import RLVisualizer, MemoryAnalyzer, generate_report
        assert RLVisualizer is not None
        assert MemoryAnalyzer is not None
        assert callable(generate_report)

    def test_rl_visualizer_init(self):
        from turbomemory.visualization import RLVisualizer
        viz = RLVisualizer(output_dir="/tmp/test_viz")
        assert viz.output_dir == "/tmp/test_viz"

    def test_memory_analyzer_init(self):
        from turbomemory.visualization import MemoryAnalyzer
        analyzer = MemoryAnalyzer("/tmp/test_root")
        assert analyzer.root == "/tmp/test_root"