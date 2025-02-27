"""
Tests for the DocumentProcessor class
"""

import unittest
import os
import tempfile
import shutil
import sys
import pytest

# Add parent directory to path to import module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from document_processor import DocumentProcessor


class TestDocumentProcessor(unittest.TestCase):
    """Test cases for DocumentProcessor class"""

    def setUp(self):
        """Set up test environment"""
        # Create temporary directory for test documents
        self.test_dir = tempfile.mkdtemp()
        self.vector_db_path = tempfile.mkdtemp()
        
        # Create a test document
        self.test_doc_path = os.path.join(self.test_dir, "test_doc.txt")
        with open(self.test_doc_path, "w") as f:
            f.write("This is a test document.\nIt contains multiple sentences.\nThis is for testing the document processor.")
            
    def tearDown(self):
        """Clean up test environment"""
        # Remove temporary directories
        shutil.rmtree(self.test_dir)
        shutil.rmtree(self.vector_db_path)

    def test_init(self):
        """Test DocumentProcessor initialization"""
        processor = DocumentProcessor(
            document_dir=self.test_dir,
            vector_db_path=self.vector_db_path,
            chunk_size=500,
            chunk_overlap=50
        )
        
        # Check if attributes are correctly set
        self.assertEqual(processor.document_dir, self.test_dir)
        self.assertEqual(processor.vector_db_path, self.vector_db_path)
        self.assertEqual(processor.chunk_size, 500)
        self.assertEqual(processor.chunk_overlap, 50)

    @pytest.mark.skipif(not os.environ.get("RUN_INTEGRATION_TESTS"), 
                       reason="Integration test - requires external dependencies")
    def test_load_documents(self):
        """Test loading documents from directory"""
        processor = DocumentProcessor(
            document_dir=self.test_dir,
            vector_db_path=self.vector_db_path
        )
        
        documents = processor.load_documents()
        
        # Check if document was loaded
        self.assertEqual(len(documents), 1)
        self.assertTrue(os.path.basename(self.test_doc_path) in documents[0].metadata["source"])
        self.assertIn("This is a test document.", documents[0].page_content)


if __name__ == "__main__":
    unittest.main()