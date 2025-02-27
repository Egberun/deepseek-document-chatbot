"""
Example script demonstrating how to use the DeepSeek Document Chatbot
with a model running in a Docker container.
"""

import logging
import argparse
import os
from document_processor import DocumentProcessor
from docker_adapter import create_docker_llm_adapter
from retrieval_chain import DocumentChatbot
from prompt_templates import PromptLibrary

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main function for Docker example usage."""
    parser = argparse.ArgumentParser(description="DeepSeek Document Chatbot with Docker")
    parser.add_argument("--docs", type=str, default="./documents", help="Directory containing documents")
    parser.add_argument("--db", type=str, default="./chroma_db", help="Directory for vector database")
    parser.add_argument("--api-url", type=str, required=True, help="URL of the model API endpoint")
    parser.add_argument("--api-type", type=str, default="openai", help="Type of API (openai, simple, etc.)")
    parser.add_argument("--api-key", type=str, default=None, help="API key if required")
    parser.add_argument("--model-name", type=str, default="deepseek", help="Name of the model")
    parser.add_argument("--max-tokens", type=int, default=512, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for text generation")
    parser.add_argument("--prompt-template", type=str, default="customer_service", help="Prompt template to use")
    args = parser.parse_args()
    
    # Initialize document processor
    logger.info("Initializing document processor...")
    doc_processor = DocumentProcessor(
        document_dir=args.docs,
        vector_db_path=args.db
    )
    
    # Process documents
    logger.info("Processing documents...")
    vectorstore = doc_processor.process()
    
    # Initialize Docker LLM adapter
    logger.info(f"Connecting to model API at {args.api_url}...")
    llm = create_docker_llm_adapter(
        api_url=args.api_url,
        api_type=args.api_type,
        api_key=args.api_key,
        model_name=args.model_name,
        max_tokens=args.max_tokens,
        temperature=args.temperature
    )
    
    # Get prompt template
    prompt_lib = PromptLibrary()
    prompt_template = prompt_lib.get_template(args.prompt_template)
    logger.info(f"Using prompt template: {prompt_template.name}")
    
    # Create chatbot
    logger.info("Creating document chatbot...")
    chatbot = DocumentChatbot(llm=llm, vectorstore=vectorstore)
    chain = chatbot.create_chain()
    
    # Example usage
    logger.info("\n=== DeepSeek Document Chatbot with Docker ===")
    print("\nWelcome to the DeepSeek Document Chatbot!")
    print("Type 'exit' or 'quit' to end the conversation.")
    
    try:
        while True:
            query = input("\nYou: ")
            
            if query.lower() in ["exit", "quit"]:
                print("\nThank you for using the chatbot. Goodbye!")
                break
                
            try:
                response = chatbot.query(query)
                
                answer = response["answer"]
                sources = response.get("sources", [])
                
                # Display response
                print(f"\nChatbot: {answer}")
                
                if sources:
                    print("\nSources:")
                    for source in sources:
                        print(f"- {source}")
                        
            except Exception as e:
                logger.error(f"Error processing query: {str(e)}")
                print(f"\nChatbot: I'm sorry, I encountered an error: {str(e)}")
                
    except KeyboardInterrupt:
        print("\n\nThank you for using the chatbot. Goodbye!")

if __name__ == "__main__":
    main()