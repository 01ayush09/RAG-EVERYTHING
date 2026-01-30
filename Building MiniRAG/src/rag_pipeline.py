class RAGPipeline:
    def __init__(self, retriever):
        self.retriever = retriever

    def _format_context(self, retrieved):
        lines = []
        for i, (text, score) in enumerate(retrieved):
            lines.append(f'-- Chunk {i+1} (score={score:.3f}) --\n{text}\n')
        return '\n'.join(lines)

    def answer(self, query):
        retrieved = self.retriever.retrieve(query)
        context = self._format_context(retrieved)
        # A minimal generation placeholder: In interviews, explain you would integrate an LLM here.
        generated = self._generate_stub_answer(query, retrieved)
        return {'query': query, 'context': context, 'answer': generated, 'retrieved': retrieved}

    def _generate_stub_answer(self, query, retrieved):
        # Very small, deterministic "answer" generator using retrieved text slices.
        combined = ' '.join([t for t, s in retrieved])
        summary = combined[:500].strip()
        return f"Based on retrieved documents, a short answer to: '{query}' -> {summary}..."
