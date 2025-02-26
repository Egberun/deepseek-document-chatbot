"""
Monitoring Module

This module provides tools for monitoring chatbot performance and usage.
"""

import logging
import time
import threading
import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class QueryLog:
    """Log entry for a chatbot query."""
    
    query: str
    timestamp: str
    response_time: float
    token_count: int
    session_id: str
    success: bool
    error: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.metadata is None:
            self.metadata = {}

class ChatbotMonitor:
    """Monitors chatbot usage and performance."""
    
    def __init__(self, log_dir="./logs"):
        """
        Initialize the monitor.
        
        Args:
            log_dir (str): Directory for storing logs
        """
        self.log_dir = log_dir
        self.query_logs: List[QueryLog] = []
        self.start_time = time.time()
        self.query_count = 0
        self.error_count = 0
        self.total_response_time = 0
        self.total_token_count = 0
        self._lock = threading.Lock()
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Create a new log file for this session
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"chatbot_log_{self.session_id}.json")
        
        logger.info(f"Monitoring session started: {self.session_id}")
    
    def log_query(self, query: str, response_time: float, token_count: int, 
                 success: bool, error: Optional[str] = None, 
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Log a chatbot query.
        
        Args:
            query (str): User query
            response_time (float): Time taken to respond (seconds)
            token_count (int): Number of tokens generated
            success (bool): Whether the query was successful
            error (str, optional): Error message if unsuccessful
            metadata (Dict[str, Any], optional): Additional metadata
        """
        timestamp = datetime.now().isoformat()
        
        if metadata is None:
            metadata = {}
            
        log_entry = QueryLog(
            query=query,
            timestamp=timestamp,
            response_time=response_time,
            token_count=token_count,
            session_id=self.session_id,
            success=success,
            error=error,
            metadata=metadata
        )
        
        with self._lock:
            self.query_logs.append(log_entry)
            self.query_count += 1
            self.total_response_time += response_time
            self.total_token_count += token_count
            
            if not success:
                self.error_count += 1
                
            # Append to log file
            self._append_to_log_file(log_entry)
    
    def _append_to_log_file(self, log_entry: QueryLog):
        """
        Append a log entry to the log file.
        
        Args:
            log_entry (QueryLog): Log entry to append
        """
        try:
            with open(self.log_file, "a") as f:
                if os.path.getsize(self.log_file) == 0:
                    # New file, start JSON array
                    f.write("[\n")
                else:
                    # Existing file, add comma
                    f.seek(0, os.SEEK_END)
                    f.seek(f.tell() - 2, os.SEEK_SET)  # Move before final newline and ]
                    f.write(",\n")
                    
                json.dump(asdict(log_entry), f, indent=2)
                f.write("\n]")
                
        except Exception as e:
            logger.error(f"Error appending to log file: {str(e)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get current monitoring statistics.
        
        Returns:
            Dict[str, Any]: Current statistics
        """
        with self._lock:
            uptime = time.time() - self.start_time
            avg_response_time = self.total_response_time / max(1, self.query_count)
            avg_token_count = self.total_token_count / max(1, self.query_count)
            error_rate = self.error_count / max(1, self.query_count)
            
            return {
                "session_id": self.session_id,
                "uptime_seconds": uptime,
                "query_count": self.query_count,
                "error_count": self.error_count,
                "error_rate": error_rate,
                "avg_response_time": avg_response_time,
                "avg_token_count": avg_token_count
            }
    
    def get_recent_queries(self, count: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent queries.
        
        Args:
            count (int): Number of recent queries to retrieve
            
        Returns:
            List[Dict[str, Any]]: Recent queries
        """
        with self._lock:
            return [asdict(log) for log in self.query_logs[-count:]]
            
    def reset_stats(self):
        """Reset monitoring statistics."""
        with self._lock:
            self.start_time = time.time()
            self.query_count = 0
            self.error_count = 0
            self.total_response_time = 0
            self.total_token_count = 0
            
            # Start a new log file
            self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_file = os.path.join(self.log_dir, f"chatbot_log_{self.session_id}.json")
            
            logger.info(f"Monitoring stats reset, new session: {self.session_id}")