"""
Retrieval Chain Module

This module creates the conversational retrieval chain that combines
document retrieval with the language model to answer user queries.
"""

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentChatbot:
    """Combines retrieval and language model for document question answering."""
    
    def __init__(self, llm, vectorstore, memory_size=5):
        """
        Initialize the document chatbot.
        
        Args:
            llm: LangChain language model
            vectorstore: Vector store for document retrieval
            memory_size (int): Number of conversation turns to remember
        """
        self.llm = llm
        self.vectorstore = vectorstore
        self.memory_size = memory_size
        self.chain = None
        
    def create_chain(self):
        """
        Create the conversational retrieval chain.
        
        Returns:
            ConversationalRetrievalChain: The complete chain
        """
        logger.info("Creating conversational retrieval chain")
        try:
            # Set up conversation memory
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer",
                input_key="question",
                k=self.memory_size
            )
            
            # Create the chain
            self.chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
                memory=memory,
                return_source_documents=True,  # Return source documents for attribution
                verbose=True  # Set to False in production
            )
            
            logger.info("Conversational retrieval chain created successfully")
            return self.chain
            
        except Exception as e:
            logger.error(f"Error creating chain: {str(e)}")
            raise
    
    def query(self, question):
        """
        Process a user query through the chain.
        
        Args:
            question (str): User question
            
        Returns:
            dict: Response containing answer and source documents
        """
        if not self.chain:
            logger.error("Chain not created. Call create_chain() first.")
            raise ValueError("Chain not created")
        
        logger.info(f"Processing query: {question}")
        try:
            result = self.chain({"question": question})
            return {
                "answer": result["answer"],
                "sources": [doc.metadata.get("source", "Unknown") for doc in result.get("source_documents", [])]
            }
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            error_message = f"I'm sorry, I encountered an error: {str(e)}"
            return {"answer": error_message, "sources": []}