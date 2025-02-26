"""
Integration Example Script

This script demonstrates how to use all components together
to create a complete chatbot application.
"""

import logging
import argparse
import os
import time
from document_processor import DocumentProcessor
from llm_model import DeepSeekLLM
from retrieval_chain import DocumentChatbot
from config import ChatbotConfig
from monitoring import ChatbotMonitor
from prompt_templates import PromptLibrary, PromptTemplate

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="DeepSeek Document Chatbot Example")
    parser.add_argument("--config", type=str, default="config.json", help="Path to configuration file")
    parser.add_argument("--template", type=str, default="customer_service", help="Prompt template to use")
    args = parser.parse_args()
    
    # Load configuration
    config = ChatbotConfig.load(args.config)
    
    # Set up monitoring
    monitor = ChatbotMonitor()
    
    # Initialize document processor
    logger.info("Initializing document processor...")
    doc_processor = DocumentProcessor(
        document_dir=config.document_dir,
        vector_db_path=config.vector_db_path,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        embedding_model=config.embedding_model
    )
    
    # Process documents
    logger.info("Processing documents...")
    vectorstore = doc_processor.process()
    
    # Load DeepSeek LLM
    logger.info("Loading DeepSeek LLM...")
    deepseek_llm = DeepSeekLLM(
        model_name=config.llm_model,
        use_gpu=config.use_gpu,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature
    )
    llm = deepseek_llm.load_model()
    
    # Initialize prompt library
    prompt_library = PromptLibrary()
    
    # Add custom prompt template if specified in config
    if config.custom_prompts:
        for name, prompt_data in config.custom_prompts.items():
            prompt_library.add_template(
                PromptTemplate(
                    name=name,
                    system_prompt=prompt_data.get("system_prompt", ""),
                    query_prefix=prompt_data.get("query_prefix", ""),
                    query_suffix=prompt_data.get("query_suffix", ""),
                    context_prefix=prompt_data.get("context_prefix", "Relevant information:\n"),
                    context_suffix=prompt_data.get("context_suffix", "\n")
                )
            )
    
    # Get selected prompt template
    prompt_template = prompt_library.get_template(args.template)
    logger.info(f"Using prompt template: {prompt_template.name}")
    
    # Custom prompt handler for the LLM
    def prompt_handler(query, context=None):
        return prompt_template.format(query=query, context=context)
    
    # Override LLM's format_prompt function
    deepseek_llm.format_prompt = prompt_handler
    
    # Create chatbot
    logger.info("Creating document chatbot...")
    chatbot = DocumentChatbot(llm=llm, vectorstore=vectorstore, memory_size=config.memory_size)
    chain = chatbot.create_chain()
    
    # Example usage
    logger.info("\n=== DeepSeek Document Chatbot Example ===")
    print("\nWelcome to the DeepSeek Document Chatbot!")
    print("Type 'exit' or 'quit' to end the conversation.")
    print("Type 'stats' to see monitoring statistics.")
    
    try:
        while True:
            query = input("\nYou: ")
            
            if query.lower() in ["exit", "quit"]:
                print("\nThank you for using the chatbot. Goodbye!")
                break
                
            if query.lower() == "stats":
                stats = monitor.get_stats()
                print("\n=== Monitoring Statistics ===")
                print(f"Session ID: {stats['session_id']}")
                print(f"Uptime: {stats['uptime_seconds']:.2f} seconds")
                print(f"Queries: {stats['query_count']}")
                print(f"Errors: {stats['error_count']} ({stats['error_rate']*100:.2f}%)")
                print(f"Avg Response Time: {stats['avg_response_time']*1000:.2f} ms")
                print(f"Avg Tokens: {stats['avg_token_count']:.2f}")
                continue
            
            start_time = time.time()
            
            try:
                response = chatbot.query(query)
                
                end_time = time.time()
                response_time = end_time - start_time
                
                answer = response["answer"]
                sources = response.get("sources", [])
                
                # Log the query
                token_count = len(answer.split()) * 1.3  # Rough estimate
                monitor.log_query(
                    query=query,
                    response_time=response_time,
                    token_count=int(token_count),
                    success=True,
                    metadata={"sources": sources}
                )
                
                # Display response
                print(f"\nChatbot ({response_time*1000:.0f}ms): {answer}")
                
                if sources:
                    print("\nSources:")
                    for source in sources:
                        print(f"- {source}")
                        
            except Exception as e:
                logger.error(f"Error processing query: {str(e)}")
                error_message = f"I'm sorry, I encountered an error: {str(e)}"
                print(f"\nChatbot: {error_message}")
                
                # Log the error
                end_time = time.time()
                response_time = end_time - start_time
                monitor.log_query(
                    query=query,
                    response_time=response_time,
                    token_count=0,
                    success=False,
                    error=str(e)
                )
                
    except KeyboardInterrupt:
        print("\n\nThank you for using the chatbot. Goodbye!")

if __name__ == "__main__":
    main()