import os
import unittest
import numpy as np
import database
import document_parser

class TestLocalRAG(unittest.TestCase):
    
    def setUp(self):
        # Initialize the DB to make sure schema is created
        database.init_db()
        
    def test_chunking(self):
        text = "This is a sentence. " * 50
        chunks = document_parser.chunk_text(text, chunk_size=200, chunk_overlap=50)
        self.assertTrue(len(chunks) > 1)
        for chunk in chunks:
            self.assertTrue(len(chunk) <= 250) # Approx including overlap
            
    def test_database_insert_and_search(self):
        # Insert a mock document
        doc_id = database.add_document("test_doc.txt", ".txt", "processing")
        self.assertIsNotNone(doc_id)
        
        # Add chunks with dummy embeddings
        # We use dimensions of 4 (e.g. nomic-embed-text uses 768, but we test mathematical similarity with 4)
        emb1 = [1.0, 0.0, 0.0, 0.0]
        emb2 = [0.0, 1.0, 0.0, 0.0]
        
        database.add_document_chunk(doc_id, "This is chunk one talking about apples.", emb1, page_num=1)
        database.add_document_chunk(doc_id, "This is chunk two talking about oranges.", emb2, page_num=2)
        
        # Update status
        database.update_document_status(doc_id, "completed")
        
        # Retrieve documents
        docs = database.get_all_documents()
        self.assertTrue(any(d["id"] == doc_id for d in docs))
        
        # Search using query that matches emb1
        results = database.similarity_search([1.0, 0.1, 0.0, 0.0], top_k=1, doc_ids=[doc_id])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["text"], "This is chunk one talking about apples.")
        self.assertTrue(results[0]["score"] > 0.9)
        
        # Clean up document
        database.delete_document(doc_id)
        
        # Verify chunks are gone
        results_after_delete = database.similarity_search([1.0, 0.1, 0.0, 0.0], top_k=5, doc_ids=[doc_id])
        self.assertEqual(len(results_after_delete), 0)

if __name__ == '__main__':
    unittest.main()
