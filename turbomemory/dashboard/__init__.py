import streamlit as st
import json
import os
from turbomemory import TurboMemory


st.set_page_config(page_title='TurboMemory Dashboard', page_icon='🧠', layout='wide')

st.title('🧠 TurboMemory Dashboard')
st.caption('Browse, search, and manage your memory store')


@st.cache_resource
def load_memory(root: str) -> TurboMemory:
    return TurboMemory(root=root)


with st.sidebar:
    st.header('Configuration')
    root_dir = st.text_input('Memory Root', value='turbomemory_data')
    
    if os.path.exists(root_dir):
        tm = load_memory(root_dir)
        st.success(f'Connected to {root_dir}')
    else:
        st.error(f'Directory not found: {root_dir}')
        st.stop()

    st.divider()
    
    action = st.selectbox(
        'Action',
        ['Overview', 'Browse Topics', 'Search', 'Add Memory', 'Consolidate', 'Metrics'],
    )


def render_overview():
    st.header('Overview')
    
    stats = tm.stats()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Topics', stats.get('total_topics', 0))
    col2.metric('Chunks', stats.get('total_chunks', 0))
    col3.metric('Avg Quality', f'{stats.get(\"avg_quality\", 0):.2f}')
    col4.metric('Storage', f'{stats.get(\"storage_bytes\", 0) / 1024:.1f} KB')


def render_browse_topics():
    st.header('Browse Topics')
    
    topics = tm.list_topics()
    
    topic = st.selectbox('Select Topic', topics)
    
    if topic:
        data = tm.load_topic(topic)
        chunks = data.get('chunks', [])
        
        st.subheader(f'Topic: {topic}')
        st.write(f'Chunks: {len(chunks)}')
        
        for i, chunk in enumerate(chunks[:10]):
            with st.expander(f'Chunk {i+1}'):
                st.write(f'**Text:** {chunk.get(\"text\", \"\")[:200]}')
                st.write(f'**Confidence:** {chunk.get(\"confidence\", 0):.3f}')
                st.write(f'**Quality:** {chunk.get(\"quality_score\", 0):.3f}')


def render_search():
    st.header('Search Memory')
    
    query = st.text_input('Search Query')
    top_k = st.slider('Top K', 1, 20, 5)
    
    if st.button('Search'):
        results = tm.query(query, k=top_k)
        
        st.subheader(f'Results ({len(results)})')
        
        for score, topic, chunk in results:
            with st.expander(f'{topic} - {score:.3f}'):
                st.write(chunk.get('text', ''))


def render_add_memory():
    st.header('Add Memory')
    
    topic = st.text_input('Topic')
    text = st.text_area('Text')
    confidence = st.slider('Confidence', 0.0, 1.0, 0.8)
    
    if st.button('Add'):
        if topic and text:
            chunk_id = tm.add_memory(topic, text, confidence=confidence)
            st.success(f'Added chunk {chunk_id}')
        else:
            st.error('Topic and text are required.')


def render_consolidate():
    st.header('Consolidate')
    
    st.info('Run consolidation to merge similar memories, remove duplicates, and improve quality.')
    
    if st.button('Run Consolidation'):
        st.warning('Consolidation is not yet implemented in the dashboard.')
        # TODO: integrate consolidator


def render_metrics():
    st.header('Metrics')
    
    metrics = tm.get_metrics()
    st.json(metrics.to_dict())


if action == 'Overview':
    render_overview()
elif action == 'Browse Topics':
    render_browse_topics()
elif action == 'Search':
    render_search()
elif action == 'Add Memory':
    render_add_memory()
elif action == 'Consolidate':
    render_consolidate()
elif action == 'Metrics':
    render_metrics()