"""
Docker Adapter Module

This module provides adapters for connecting to models running in Docker containers.
It allows the chatbot to use models exposed via HTTP APIs instead of loading them locally.
"""

import json
import logging
import requests
from typing import Dict, Any, List, Optional, Union
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DockerLLMAdapter(LLM):
    """Adapter for connecting to LLMs running in Docker containers."""
    
    api_url: str
    api_key: Optional[str] = None
    model_name: str = "deepseek"
    max_tokens: int = 512
    temperature: float = 0.7
    timeout: int = 30
    request_headers: Dict[str, str] = None
    
    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "docker_adapter"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call the model API with the given prompt."""
        logger.info(f"Calling model API at {self.api_url}")
        
        try:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            if self.request_headers:
                headers.update(self.request_headers)
            
            payload = {
                "prompt": prompt,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "model": self.model_name
            }
            
            if stop:
                payload["stop"] = stop
                
            # Update with any additional kwargs
            payload.update(kwargs)
            
            logger.debug(f"Sending request: {json.dumps(payload, indent=2)}")
            
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            
            response.raise_for_status()
            result = response.json()
            
            logger.debug(f"Received response: {json.dumps(result, indent=2)}")
            
            # Handle different API response formats
            if "choices" in result and len(result["choices"]) > 0:
                # OpenAI-like format
                return result["choices"][0].get("text", "").strip()
            elif "generation" in result:
                # Text generation format
                return result["generation"].strip()
            elif "response" in result:
                # Simple response format
                return result["response"].strip()
            else:
                # Default to returning the whole response as string
                return str(result)
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling model API: {str(e)}")
            raise ValueError(f"Error calling model API: {str(e)}")
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get identifying parameters."""
        return {
            "api_url": self.api_url,
            "model_name": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }


# Factory function to create appropriate adapter based on API type
def create_docker_llm_adapter(
    api_url: str,
    api_type: str = "openai",
    api_key: Optional[str] = None,
    model_name: str = "deepseek",
    max_tokens: int = 512,
    temperature: float = 0.7
) -> DockerLLMAdapter:
    """
    Create an appropriate LLM adapter based on API type.
    
    Args:
        api_url (str): URL of the API endpoint
        api_type (str): Type of API (openai, simple, etc.)
        api_key (str, optional): API key if required
        model_name (str): Name of the model to use
        max_tokens (int): Maximum number of tokens to generate
        temperature (float): Temperature for text generation
        
    Returns:
        DockerLLMAdapter: Configured adapter
    """
    # Set custom headers based on API type
    request_headers = {}
    
    if api_type == "openai":
        request_headers["Content-Type"] = "application/json"
    elif api_type == "huggingface":
        request_headers["Content-Type"] = "application/json"
        
    return DockerLLMAdapter(
        api_url=api_url,
        api_key=api_key,
        model_name=model_name,
        max_tokens=max_tokens,
        temperature=temperature,
        request_headers=request_headers
    )