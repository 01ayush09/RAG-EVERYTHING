import streamlit as st
from src.utils import load_config
from src.retriever import Retriever
from src.rag_pipeline import RAGPipeline

st.set_page_config(page_title='My Local RAG', layout='wide')

@st.cache_data(show_spinner=False)
def init_pipeline(config):
    retriever = Retriever(config['documents_path'], config.get('embedding_model'))
    return RAGPipeline(retriever)

def main():
    st.title('My Local RAG — Compact Demo')
    st.write('Upload .txt files to data/docs or edit sample docs in data/docs/.')
    config = load_config()
    st.sidebar.header('Config')
    st.sidebar.write(config)

    pipeline = init_pipeline(config)

    q = st.text_input('Ask a question:')
    topk = st.sidebar.slider('Top K', min_value=1, max_value=5, value=config.get('top_k', 2))
    if st.button('Get Answer'):
        if not q.strip():
            st.warning('Type a question first.')
        else:
            res = pipeline.answer(q)
            st.subheader('Answer (generated stub)')
            st.write(res['answer'])
            st.subheader('Retrieved Context')
            st.markdown('---')
            for i, (txt, score) in enumerate(res['retrieved']):
                st.markdown(f'**Chunk {i+1} (score={score:.3f})**')
                st.write(txt[:800] + ('...' if len(txt) > 800 else ''))
                st.markdown('---')

if __name__ == '__main__':
    main()
