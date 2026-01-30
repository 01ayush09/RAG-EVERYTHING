import nltk
nltk.download('punkt', quiet=True)
from nltk.tokenize import word_tokenize

def chunk_text(text, size=200, overlap=40):
    words = word_tokenize(text)
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+size]
        chunks.append(' '.join(chunk))
        i += size - overlap
    return chunks
