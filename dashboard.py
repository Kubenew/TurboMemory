"""TurboMemory Streamlit Dashboard."""

import streamlit as st
import json
import os
from turbomemory import TurboMemory


st.set_page_config(page_title="TurboMemory Dashboard", page_icon="🧠", layout="wide")

st.title("🧠 TurboMemory Dashboard")
st.caption("Browse, search, and manage your memory store")


@st.cache_resource
def load_memory(root: str) -> TurboMemory:
    return TurboMemory(root=root)


# Sidebar
with st.sidebar:
    st.header("Configuration")
    root_dir = st.text_input("Memory Root", value="turbomemory_data")
    
    if os.path.exists(root_dir):
        tm = load_memory(root_dir)
        st.success(f"Connected to {root_dir}")
    else:
        st.error(f"Directory not found: {root_dir}")
        st.stop()

    st.divider()
    
    action = st.selectbox(
        "Action",
        ["Overview", "Browse Topics", "Search", "Add Memory", "Consolidate", "Metrics"],
    )


def render_overview():
    st.header("Overview")
    
    stats = tm.stats()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Topics", stats.get("total_topics", 0))
    col2.metric("Chunks", stats.get("total_chunks", 0))
    col3.metric("Avg Quality", f"{stats.get('avg_quality', 0):.2f}")
    col4.metric("Storage", f"{stats.get('storage_bytes', 0) / 1024:.1f} KB")
    
    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Verified", stats.get("verified_chunks", 0))
    col6.metric("Expired", stats.get("expired_chunks", 0))
    col7.metric("Contradicted", stats.get("contradicted_chunks", 0))
    col8.metric("Consolidations", stats.get("consolidation_runs", 0))
    
    if stats.get("topic_health"):
        st.subheader("Topic Health")
        for topic, health in sorted(stats["topic_health"].items(), key=lambda x: x[1], reverse=True):
            bar = "🟩" * int(health * 10) + "🟥" * (10 - int(health * 10))
            st.write(f"**{topic}**: {bar} ({health:.2f})")


def render_browse():
    st.header("Browse Topics")
    
    topics_dir = os.path.join(root_dir, "topics")
    if not os.path.exists(topics_dir):
        st.warning("No topics found.")
        return
    
    topic_files = [f for f in os.listdir(topics_dir) if f.endswith(".tmem")]
    
    for tf in sorted(topic_files):
        topic_name = tf.replace(".tmem", "").replace("_", ".")
        with st.expander(f"📁 {topic_name}"):
            topic_data = tm.load_topic(topic_name)
            chunks = topic_data.get("chunks", [])
            
            st.write(f"**Chunks:** {len(chunks)}")
            st.write(f"**Updated:** {topic_data.get('updated', 'N/A')}")
            
            for chunk in chunks:
                with st.container(border=True):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(chunk.get("text", "")[:200])
                    with col2:
                        st.metric("Conf", f"{chunk.get('confidence', 0):.2f}")
                        st.metric("Quality", f"{chunk.get('quality_score', 0):.2f}")
                        st.metric("Stale", f"{chunk.get('staleness', 0):.2f}")


def render_search():
    st.header("Search Memory")
    
    query = st.text_input("Query", placeholder="Search your memories...")
    
    col1, col2 = st.columns(2)
    k = col1.slider("Results", 1, 20, 5)
    verify = col2.checkbox("Require Verification")
    
    if query:
        if verify:
            results = tm.verify_and_score(query, k=k)
            for score, topic, chunk, verif in results:
                with st.container(border=True):
                    st.write(chunk.get("text", ""))
                    st.caption(f"Topic: {topic} | Score: {score:.3f} | Verified: {verif.verified}")
        else:
            results = tm.query(query, k=k)
            for score, topic, chunk in results:
                with st.container(border=True):
                    st.write(chunk.get("text", ""))
                    st.caption(f"Topic: {topic} | Score: {score:.3f} | Quality: {chunk.get('quality_score', 0):.3f}")


def render_add_memory():
    st.header("Add Memory")
    
    topic = st.text_input("Topic", placeholder="e.g., python.tips")
    text = st.text_area("Memory Text", placeholder="Enter memory content...")
    
    col1, col2, col3 = st.columns(3)
    confidence = col1.slider("Confidence", 0.0, 1.0, 0.8)
    bits = col2.selectbox("Quantization Bits", [4, 6, 8], index=1)
    ttl = col3.number_input("TTL (days, 0=none)", 0.0, 365.0, 0.0)
    
    if st.button("Add Memory", type="primary"):
        if topic and text:
            ttl_val = ttl if ttl > 0 else None
            chunk_id = tm.add_memory(topic, text, confidence, bits, ttl_days=ttl_val)
            if chunk_id:
                st.success(f"Added! Chunk ID: {chunk_id}")
            else:
                st.warning("Memory was excluded by rules.")
        else:
            st.error("Please provide both topic and text.")


def render_consolidate():
    st.header("Consolidate")
    
    st.write("Run consolidation to merge duplicates, resolve contradictions, and prune stale memories.")
    
    if st.button("Run Consolidation", type="primary"):
        with st.spinner("Consolidating..."):
            from consolidator import run_once
            import argparse
            
            args = argparse.Namespace(
                threshold=0.93,
                min_entropy=0.10,
                staleness_prune=0.90,
                max_chunks=300,
                merge_threshold=0.85,
                make_absolute=True,
            )
            
            result = run_once(tm, args)
            st.success(f"Consolidation complete!")
            st.json(result)


def render_metrics():
    st.header("Metrics (JSON)")
    
    metrics = tm.get_metrics()
    st.json(metrics.to_dict())


if action == "Overview":
    render_overview()
elif action == "Browse Topics":
    render_browse()
elif action == "Search":
    render_search()
elif action == "Add Memory":
    render_add_memory()
elif action == "Consolidate":
    render_consolidate()
elif action == "Metrics":
    render_metrics()
