from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class VectorStore:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.texts = []

    def build(self, texts):
        # texts: list of strings
        self.texts = texts
        vectors = self.model.encode(texts, show_progress_bar=False)
        self.embeddings = np.array(vectors)

    def add(self, texts):
        # append new texts
        new_vecs = self.model.encode(texts, show_progress_bar=False)
        if self.embeddings is None:
            self.embeddings = np.array(new_vecs)
        else:
            self.embeddings = np.vstack([self.embeddings, new_vecs])
        self.texts.extend(texts)

    def search(self, query, top_k=2):
        qv = self.model.encode([query], show_progress_bar=False)
        sims = cosine_similarity(qv, self.embeddings)[0]
        idxs = sims.argsort()[::-1][:top_k]
        return [(self.texts[i], float(sims[i])) for i in idxs]
