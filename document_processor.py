"""
Document Processing Module

This module handles loading documents from a directory, splitting them into
manageable chunks, and converting them to vector embeddings for storage.
"""

from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles document loading, processing, and vectorization."""
    
    def __init__(self, document_dir, vector_db_path="./chroma_db", 
                 chunk_size=1000, chunk_overlap=100,
                 embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the document processor.
        
        Args:
            document_dir (str): Directory containing documents to process
            vector_db_path (str): Directory to store the vector database
            chunk_size (int): Size of document chunks in characters
            chunk_overlap (int): Overlap between chunks in characters
            embedding_model (str): Name of the embedding model to use
        """
        self.document_dir = document_dir
        self.vector_db_path = vector_db_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model
        
    def load_documents(self):
        """
        Load documents from the specified directory.
        
        Returns:
            list: List of loaded documents
        """
        logger.info(f"Loading documents from {self.document_dir}")
        try:
            # Create a directory loader that finds all text files
            loader = DirectoryLoader(
                self.document_dir, 
                glob="**/*.txt", 
                loader_cls=TextLoader
            )
            documents = loader.load()
            logger.info(f"Loaded {len(documents)} documents")
            return documents
        except Exception as e:
            logger.error(f"Error loading documents: {str(e)}")
            raise
    
    def split_documents(self, documents):
        """
        Split documents into manageable chunks.
        
        Args:
            documents (list): List of documents to split
            
        Returns:
            list: List of document chunks
        """
        logger.info(f"Splitting documents into chunks of size {self.chunk_size} with overlap {self.chunk_overlap}")
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size, 
                chunk_overlap=self.chunk_overlap
            )
            splits = text_splitter.split_documents(documents)
            logger.info(f"Created {len(splits)} document chunks")
            return splits
        except Exception as e:
            logger.error(f"Error splitting documents: {str(e)}")
            raise
    
    def create_vector_store(self, document_chunks):
        """
        Create a vector store from document chunks.
        
        Args:
            document_chunks (list): List of document chunks to vectorize
            
        Returns:
            Chroma: Vector store object
        """
        logger.info(f"Creating vector store using {self.embedding_model}")
        try:
            # Initialize the embedding model
            embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
            
            # Create and persist the vector store
            vectorstore = Chroma.from_documents(
                documents=document_chunks,
                embedding=embeddings,
                persist_directory=self.vector_db_path
            )
            vectorstore.persist()
            logger.info(f"Vector store created and persisted to {self.vector_db_path}")
            return vectorstore
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            raise
    
    def process(self):
        """
        Process documents: load, split, and vectorize.
        
        Returns:
            Chroma: Vector store object
        """
        # Check if vector store already exists
        if os.path.exists(self.vector_db_path) and os.listdir(self.vector_db_path):
            logger.info(f"Loading existing vector store from {self.vector_db_path}")
            embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
            return Chroma(persist_directory=self.vector_db_path, embedding_function=embeddings)
        
        # Otherwise create new vector store
        documents = self.load_documents()
        document_chunks = self.split_documents(documents)
        return self.create_vector_store(document_chunks)