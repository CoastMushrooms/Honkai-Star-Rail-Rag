"""Tests for the RAG pipeline."""

import os, sys, json, unittest
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from rag import flatten, load_docs, chunk_docs, VectorStore, _split_sentences


class TestFlatten(unittest.TestCase):
    def test_string(self):
        self.assertEqual(flatten("hello"), "hello")

    def test_nested_dict(self):
        result = flatten({"first_name": "Alice", "role": "Engineer"})
        self.assertIn("First Name: Alice", result)
        self.assertIn("Role: Engineer", result)

    def test_deeply_nested(self):
        result = flatten({"info": {"age": 30, "city": "NYC"}})
        self.assertIn("Age: 30", result)
        self.assertIn("City: NYC", result)


class TestSentenceSplitting(unittest.TestCase):
    def test_basic_sentences(self):
        text = "Hello world. This is a test. Another sentence here."
        sents = _split_sentences(text)
        self.assertEqual(len(sents), 3)

    def test_paragraph_split(self):
        text = "First paragraph.\n\nSecond paragraph."
        sents = _split_sentences(text)
        self.assertEqual(len(sents), 2)

    def test_question_marks(self):
        text = "What is this? It's a test! And this too."
        sents = _split_sentences(text)
        self.assertEqual(len(sents), 3)


class TestChunking(unittest.TestCase):
    def test_short_doc_single_chunk(self):
        docs = [{"id": "1", "title": "Short", "content": "Hello world."}]
        chunks = chunk_docs(docs, max_chars=500)
        self.assertEqual(len(chunks), 1)

    def test_long_doc_splits_on_sentences(self):
        sentences = ["This is sentence number %d." % i for i in range(50)]
        text = " ".join(sentences)
        docs = [{"id": "1", "title": "Long", "content": text}]
        chunks = chunk_docs(docs, max_chars=200)
        self.assertGreater(len(chunks), 1)
        # Each chunk should not wildly exceed max_chars
        for c in chunks:
            self.assertLessEqual(len(c["text"]), 500)  # generous allowance for sentence boundaries

    def test_preserves_metadata(self):
        docs = [{"id": "abc", "title": "My Doc", "content": "Some text."}]
        chunks = chunk_docs(docs)
        self.assertEqual(chunks[0]["title"], "My Doc")
        self.assertEqual(chunks[0]["source_id"], "abc")

    def test_overlap_preserves_context(self):
        sentences = ["Sentence %d is here." % i for i in range(20)]
        text = " ".join(sentences)
        docs = [{"id": "1", "title": "T", "content": text}]
        chunks = chunk_docs(docs, max_chars=100, overlap_sentences=2)
        # With overlap, later chunks should contain sentences from previous chunks
        if len(chunks) >= 2:
            last_words_chunk0 = chunks[0]["text"].split(".")[-2]
            self.assertIn(last_words_chunk0.strip().split()[-1], chunks[1]["text"])


class TestLoadDocs(unittest.TestCase):
    def test_loads_default(self):
        docs = load_docs()
        self.assertGreater(len(docs), 0)
        for doc in docs:
            self.assertIn("id", doc)
            self.assertIn("title", doc)
            self.assertIn("content", doc)
            self.assertIsInstance(doc["content"], str)


class TestVectorStore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chunks = [
            {"id": "1", "text": "Python is a programming language used for web development and AI.", "title": "Python", "source_id": "1"},
            {"id": "2", "text": "Cats are small furry animals that people keep as pets.", "title": "Cats", "source_id": "2"},
            {"id": "3", "text": "The Eiffel Tower is a landmark in Paris, France.", "title": "Paris", "source_id": "3"},
            {"id": "4", "text": "Machine learning models learn patterns from training data.", "title": "ML", "source_id": "4"},
            {"id": "5", "text": "Dogs are loyal animals often called man's best friend.", "title": "Dogs", "source_id": "5"},
        ]
        cls.store = VectorStore(cls.chunks)

    def test_index_size(self):
        self.assertEqual(self.store.index.ntotal, 5)

    def test_semantic_search_programming(self):
        results = self.store.search("How do I code in Python?", n_results=2)
        titles = [r["title"] for r in results]
        self.assertIn("Python", titles)

    def test_semantic_search_animals(self):
        results = self.store.search("Tell me about pets", n_results=2)
        titles = [r["title"] for r in results]
        self.assertTrue(any(t in titles for t in ["Cats", "Dogs"]))

    def test_semantic_not_keyword(self):
        results = self.store.search("artificial intelligence", n_results=2)
        titles = [r["title"] for r in results]
        self.assertTrue(any(t in titles for t in ["Python", "ML"]))

    def test_results_have_scores(self):
        results = self.store.search("anything")
        for r in results:
            self.assertIn("score", r)
            self.assertIsInstance(r["score"], float)

    def test_add_chunks(self):
        s = VectorStore()
        self.assertEqual(s.index.ntotal, 0)
        s.add_chunks([{"id": "x", "text": "test chunk", "title": "Test", "source_id": "x"}])
        self.assertEqual(s.index.ntotal, 1)

    def test_n_results_capped(self):
        results = self.store.search("hello", n_results=100)
        self.assertLessEqual(len(results), 5)

    def test_stats(self):
        s = self.store.stats()
        self.assertEqual(s["total_chunks"], 5)
        self.assertEqual(s["total_documents"], 5)
        self.assertIn("embedding_dim", s)


if __name__ == "__main__":
    unittest.main()
