"""
DeepSeek LLM Integration Module

This module sets up and configures the DeepSeek language model for use
in a conversational context with document retrieval.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.llms import HuggingFacePipeline
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DeepSeekLLM:
    """Handles loading and configuration of the DeepSeek language model."""
    
    def __init__(self, model_name="deepseek-ai/deepseek-llm-7b-chat", 
                 use_gpu=True, max_new_tokens=512, temperature=0.7):
        """
        Initialize the DeepSeek LLM.
        
        Args:
            model_name (str): Name of the model to load
            use_gpu (bool): Whether to use GPU acceleration
            max_new_tokens (int): Maximum number of tokens to generate
            temperature (float): Temperature for text generation
        """
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.tokenizer = None
        self.model = None
        self.llm = None
        
    def load_model(self):
        """
        Load the DeepSeek model and tokenizer.
        
        Returns:
            HuggingFacePipeline: LangChain compatible LLM
        """
        logger.info(f"Loading DeepSeek model: {self.model_name}")
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Determine the device map
            device_map = "auto" if self.use_gpu else "cpu"
            
            # Load model with appropriate settings
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.use_gpu else torch.float32,
                device_map=device_map,
                trust_remote_code=True  # Some models may require this
            )
            
            # Create a text generation pipeline
            text_generation_pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Wrap the pipeline in a LangChain compatible format
            self.llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
            logger.info("DeepSeek model loaded successfully")
            return self.llm
            
        except Exception as e:
            logger.error(f"Error loading DeepSeek model: {str(e)}")
            raise
    
    def format_prompt(self, query, context=None):
        """
        Format a prompt for DeepSeek in the expected style.
        
        Args:
            query (str): User query
            context (str, optional): Document context to include
            
        Returns:
            str: Formatted prompt
        """
        # This may need to be adjusted based on DeepSeek's specific prompt format
        system_prompt = (
            "You are a helpful customer service AI assistant. "
            "Answer questions based on the provided information. "
            "If you don't know the answer, say that you don't know."
        )
        
        if context:
            prompt = f"<|im_start|>system\n{system_prompt}\n\nRelevant information:\n{context}<|im_end|>\n"
        else:
            prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            
        prompt += f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
        
        return prompt