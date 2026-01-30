from .chunker import chunk_text
from .vectorstore import VectorStore
import glob, os

class Retriever:
    def __init__(self, docs_path, model_name):
        self.docs_path = docs_path
        self.store = VectorStore(model_name)
        self._prepare_store()

    def _read_txt_files(self):
        texts = []
        paths = glob.glob(os.path.join(self.docs_path, '*.txt'))
        for p in paths:
            with open(p, 'r', encoding='utf-8') as f:
                texts.append(f.read())
        return texts

    def _prepare_store(self):
        raw_texts = self._read_txt_files()
        chunks = []
        for t in raw_texts:
            chunks.extend(chunk_text(t))
        self.store.build(chunks)

    def retrieve(self, query, top_k=2):
        return self.store.search(query, top_k=top_k)
