"""
Main Application Module

This module ties together all components into a complete chatbot application.
It handles the initialization, configuration, and execution of the chatbot.
"""

from document_processor import DocumentProcessor
from llm_model import DeepSeekLLM
from retrieval_chain import DocumentChatbot
import argparse
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="DeepSeek Document Chatbot")
    parser.add_argument("--docs", type=str, default="./documents", 
                        help="Directory containing documents")
    parser.add_argument("--db", type=str, default="./chroma_db", 
                        help="Directory for vector database")
    parser.add_argument("--model", type=str, default="deepseek-ai/deepseek-llm-7b-chat", 
                        help="DeepSeek model name")
    parser.add_argument("--chunk-size", type=int, default=1000, 
                        help="Document chunk size")
    parser.add_argument("--chunk-overlap", type=int, default=100, 
                        help="Document chunk overlap")
    parser.add_argument("--cpu-only", action="store_true", 
                        help="Use CPU only (no GPU)")
    parser.add_argument("--cli", action="store_true", 
                        help="Use command line interface (no web UI)")
    
    return parser.parse_args()

def initialize_components(args):
    """
    Initialize all components based on arguments.
    
    Args:
        args: Command line arguments
        
    Returns:
        tuple: (document_processor, llm, chatbot)
    """
    logger.info("Initializing components")
    
    # Ensure document directory exists
    if not os.path.exists(args.docs):
        logger.warning(f"Document directory {args.docs} does not exist. Creating it.")
        os.makedirs(args.docs)
    
    # Initialize document processor
    doc_processor = DocumentProcessor(
        document_dir=args.docs,
        vector_db_path=args.db,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    
    # Process documents and create/load vector store
    vectorstore = doc_processor.process()
    
    # Load DeepSeek LLM
    deepseek_llm = DeepSeekLLM(
        model_name=args.model,
        use_gpu=not args.cpu_only
    )
    llm = deepseek_llm.load_model()
    
    # Create document chatbot
    chatbot = DocumentChatbot(llm=llm, vectorstore=vectorstore)
    chain = chatbot.create_chain()
    
    logger.info("All components initialized successfully")
    return doc_processor, deepseek_llm, chatbot

def cli_interface(chatbot):
    """
    Run a command line interface for the chatbot.
    
    Args:
        chatbot: Document chatbot instance
    """
    logger.info("Starting command line interface")
    print("\n=== DeepSeek Document Chatbot ===")
    print("Type 'exit', 'quit', or Ctrl+C to end the conversation.\n")
    
    try:
        while True:
            query = input("\nYou: ")
            
            if query.lower() in ["exit", "quit"]:
                print("\nGoodbye!")
                break
                
            response = chatbot.query(query)
            print(f"\nChatbot: {response['answer']}")
            
            if response['sources']:
                print("\nSources:")
                for source in response['sources']:
                    print(f"- {source}")
                    
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
    except Exception as e:
        logger.error(f"Error in CLI interface: {str(e)}")
        print(f"\nAn error occurred: {str(e)}")

def web_interface(chatbot):
    """
    Launch a web interface for the chatbot.
    
    Args:
        chatbot: Document chatbot instance
    """
    import gradio as gr
    
    logger.info("Starting web interface")
    
    def respond(message, chat_history):
        """Process a message and update chat history."""
        response = chatbot.query(message)
        
        answer = response["answer"]
        sources = response.get("sources", [])
        
        if sources:
            source_text = "\n\nSources:\n" + "\n".join([f"- {source}" for source in sources])
            answer += source_text
            
        chat_history.append((message, answer))
        return "", chat_history
    
    with gr.Blocks(title="DeepSeek Document Chatbot") as demo:
        gr.Markdown("# Customer Service Document Chatbot")
        gr.Markdown("Ask questions about the documents in the knowledge base.")
        
        chatbot_ui = gr.Chatbot()
        msg = gr.Textbox(placeholder="Ask a question about the documents...")
        clear = gr.Button("Clear Conversation")
        
        msg.submit(respond, [msg, chatbot_ui], [msg, chatbot_ui])
        clear.click(lambda: None, None, chatbot_ui, queue=False)
    
    # Launch the web interface
    demo.launch(share=False)

def main():
    """Main entry point for the application."""
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Initialize components
        _, _, chatbot = initialize_components(args)
        
        # Start the appropriate interface
        if args.cli:
            cli_interface(chatbot)
        else:
            web_interface(chatbot)
            
    except Exception as e:
        logger.critical(f"Critical error in main: {str(e)}")
        print(f"Critical error: {str(e)}")
        return 1
        
    return 0

if __name__ == "__main__":
    exit(main())