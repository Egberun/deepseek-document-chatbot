"""
Prompt Templates Module

This module provides customizable prompt templates for different use cases.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class PromptTemplate:
    """Template for LLM prompts."""
    
    name: str
    system_prompt: str
    query_prefix: str = ""
    query_suffix: str = ""
    context_prefix: str = "Relevant information:\n"
    context_suffix: str = "\n"
    
    def format(self, query: str, context: Optional[str] = None) -> str:
        """
        Format a complete prompt.
        
        Args:
            query (str): User query
            context (str, optional): Retrieved context
            
        Returns:
            str: Formatted prompt
        """
        formatted_prompt = f"<|im_start|>system\n{self.system_prompt}"
        
        if context:
            formatted_prompt += f"\n\n{self.context_prefix}{context}{self.context_suffix}"
            
        formatted_prompt += f"<|im_end|>\n<|im_start|>user\n{self.query_prefix}{query}{self.query_suffix}<|im_end|>\n<|im_start|>assistant\n"
        
        return formatted_prompt

class PromptLibrary:
    """Library of prompt templates for different use cases."""
    
    def __init__(self):
        """Initialize prompt library with default templates."""
        self.templates: Dict[str, PromptTemplate] = {}
        self._initialize_defaults()
    
    def _initialize_defaults(self):
        """Initialize default prompt templates."""
        # Customer service template
        self.add_template(
            PromptTemplate(
                name="customer_service",
                system_prompt=(
                    "You are a helpful customer service AI assistant. "
                    "Answer questions based on the provided information. "
                    "Be concise, professional, and empathetic. "
                    "If you don't know the answer, say that you don't know and offer to escalate to a human agent."
                )
            )
        )
        
        # Technical support template
        self.add_template(
            PromptTemplate(
                name="technical_support",
                system_prompt=(
                    "You are a technical support AI assistant. "
                    "Provide clear, step-by-step solutions to technical problems based on the provided documentation. "
                    "Use technical terminology appropriately. "
                    "If the solution is not in the documentation, suggest troubleshooting steps and escalation paths."
                )
            )
        )
        
        # FAQ template
        self.add_template(
            PromptTemplate(
                name="faq",
                system_prompt=(
                    "You are an AI FAQ assistant. "
                    "Provide brief, direct answers to questions based on the provided FAQ information. "
                    "Keep responses concise and to the point. "
                    "If the question is not covered in the FAQs, politely state that and suggest related topics."
                )
            )
        )
    
    def add_template(self, template: PromptTemplate):
        """
        Add a template to the library.
        
        Args:
            template (PromptTemplate): Template to add
        """
        self.templates[template.name] = template
        logger.debug(f"Added template: {template.name}")
    
    def get_template(self, name: str) -> PromptTemplate:
        """
        Get a template by name.
        
        Args:
            name (str): Template name
            
        Returns:
            PromptTemplate: The requested template
            
        Raises:
            KeyError: If template doesn't exist
        """
        if name not in self.templates:
            logger.warning(f"Template '{name}' not found, using 'customer_service'")
            return self.templates["customer_service"]
            
        return self.templates[name]
    
    def list_templates(self) -> List[str]:
        """
        List all available templates.
        
        Returns:
            List[str]: List of template names
        """
        return list(self.templates.keys())