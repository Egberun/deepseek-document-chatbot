"""
Helper utilities for the DeepSeek Document Chatbot
"""

import os
import logging
import json
from typing import Dict, Any, List, Optional


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """
    Set up logging configuration
    
    Args:
        log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        logging.Logger: Configured logger
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    # Configure logging
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    return logging.getLogger(__name__)


def ensure_directory(directory_path: str) -> None:
    """
    Ensure a directory exists
    
    Args:
        directory_path (str): Path to directory
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        
        
def load_json_file(filepath: str) -> Dict[str, Any]:
    """
    Load JSON file into dictionary
    
    Args:
        filepath (str): Path to JSON file
        
    Returns:
        Dict[str, Any]: Loaded JSON data
    """
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}
        
        
def save_json_file(data: Dict[str, Any], filepath: str) -> None:
    """
    Save dictionary to JSON file
    
    Args:
        data (Dict[str, Any]): Data to save
        filepath (str): Path to save JSON file
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
        
        
def format_sources(sources: List[str]) -> str:
    """
    Format a list of source documents into a readable string
    
    Args:
        sources (List[str]): List of source document paths
        
    Returns:
        str: Formatted sources string
    """
    if not sources:
        return ""
        
    formatted_sources = "\n\nSources:\n"
    for i, source in enumerate(sources, 1):
        # Get just the filename from the path
        filename = os.path.basename(source)
        formatted_sources += f"{i}. {filename}\n"
        
    return formatted_sources


def truncate_text(text: str, max_length: int = 100) -> str:
    """
    Truncate text to a maximum length
    
    Args:
        text (str): Text to truncate
        max_length (int): Maximum length
        
    Returns:
        str: Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length] + "..."