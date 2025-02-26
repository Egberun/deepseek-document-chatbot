"""
Configuration Module

This module handles loading and managing configuration settings
for the chatbot application.
"""

import os
import json
import logging
from dataclasses import dataclass, field, asdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ChatbotConfig:
    """Configuration settings for the document chatbot."""
    
    # Document processing settings
    document_dir: str = "./documents"
    vector_db_path: str = "./chroma_db"
    chunk_size: int = 1000
    chunk_overlap: int = 100
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # LLM settings
    llm_model: str = "deepseek-ai/deepseek-llm-7b-chat"
    use_gpu: bool = True
    max_new_tokens: int = 512
    temperature: float = 0.7
    
    # Retrieval settings
    num_results: int = 3
    memory_size: int = 5
    
    # UI settings
    use_cli: bool = False
    ui_theme: str = "default"
    ui_title: str = "DeepSeek Document Chatbot"
    
    # Advanced settings
    log_level: str = "INFO"
    cache_dir: str = "./.cache"
    custom_prompts: dict = field(default_factory=dict)
    
    def save(self, filepath="config.json"):
        """
        Save configuration to a JSON file.
        
        Args:
            filepath (str): Path to save the configuration
        """
        try:
            with open(filepath, "w") as f:
                json.dump(asdict(self), f, indent=2)
            logger.info(f"Configuration saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")
    
    @classmethod
    def load(cls, filepath="config.json"):
        """
        Load configuration from a JSON file.
        
        Args:
            filepath (str): Path to load the configuration from
            
        Returns:
            ChatbotConfig: Loaded configuration
        """
        if not os.path.exists(filepath):
            logger.warning(f"Configuration file {filepath} not found, using defaults")
            return cls()
            
        try:
            with open(filepath, "r") as f:
                config_dict = json.load(f)
            logger.info(f"Configuration loaded from {filepath}")
            return cls(**config_dict)
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            logger.warning("Using default configuration")
            return cls()