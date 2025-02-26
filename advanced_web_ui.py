"""
Advanced Web UI Module

This module provides an enhanced web interface for the chatbot with
additional features like file upload and conversation export.
"""

import gradio as gr
import os
import json
import logging
from datetime import datetime
from document_processor import DocumentProcessor
from llm_model import DeepSeekLLM
from retrieval_chain import DocumentChatbot

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedWebUI:
    """Enhanced web interface for the document chatbot."""
    
    def __init__(self, chatbot, doc_processor, document_dir="./documents"):
        """
        Initialize the web UI.
        
        Args:
            chatbot: Document chatbot instance
            doc_processor: Document processor instance
            document_dir (str): Directory for documents
        """
        self.chatbot = chatbot
        self.doc_processor = doc_processor
        self.document_dir = document_dir
        self.conversation_history = []
        
    def respond(self, message, chat_history):
        """
        Process a message and update chat history.
        
        Args:
            message (str): User message
            chat_history (list): Current chat history
            
        Returns:
            tuple: (empty string, updated chat history)
        """
        if not message.strip():
            return "", chat_history
            
        try:
            response = self.chatbot.query(message)
            
            answer = response["answer"]
            sources = response.get("sources", [])
            
            # Format sources if available
            if sources:
                source_text = "\n\nSources:\n" + "\n".join([f"- {source}" for source in sources])
                answer += source_text
                
            # Update history
            chat_history.append((message, answer))
            self.conversation_history.append({"role": "user", "content": message})
            self.conversation_history.append({"role": "assistant", "content": answer})
            
            return "", chat_history
            
        except Exception as e:
            logger.error(f"Error in respond: {str(e)}")
            error_message = f"I'm sorry, I encountered an error: {str(e)}"
            chat_history.append((message, error_message))
            return "", chat_history
    
    def save_conversation(self, chat_history):
        """
        Save the conversation to a file.
        
        Args:
            chat_history (list): Current chat history
            
        Returns:
            str: Path to saved file
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"conversation_{timestamp}.json"
            
            # Create a conversations directory if it doesn't exist
            os.makedirs("conversations", exist_ok=True)
            filepath = os.path.join("conversations", filename)
            
            # Save the conversation
            with open(filepath, "w") as f:
                json.dump(self.conversation_history, f, indent=2)
                
            logger.info(f"Conversation saved to {filepath}")
            return f"Conversation saved to {filepath}"
            
        except Exception as e:
            logger.error(f"Error saving conversation: {str(e)}")
            return f"Error saving conversation: {str(e)}"
    
    def upload_files(self, files):
        """
        Handle file uploads.
        
        Args:
            files (list): List of uploaded files
            
        Returns:
            str: Status message
        """
        try:
            # Create documents directory if it doesn't exist
            os.makedirs(self.document_dir, exist_ok=True)
            
            # Save uploaded files
            for file in files:
                file_path = os.path.join(self.document_dir, os.path.basename(file.name))
                with open(file_path, "wb") as f:
                    f.write(file.read())
            
            # Reprocess documents
            self.doc_processor.process()
            
            logger.info(f"Uploaded and processed {len(files)} files")
            return f"Successfully uploaded and processed {len(files)} files"
            
        except Exception as e:
            logger.error(f"Error uploading files: {str(e)}")
            return f"Error uploading files: {str(e)}"
    
    def clear_conversation(self):
        """
        Clear the conversation history.
        
        Returns:
            list: Empty chat history
        """
        self.conversation_history = []
        return None
    
    def launch(self):
        """Launch the web interface."""
        with gr.Blocks(title="DeepSeek Document Chatbot") as demo:
            gr.Markdown("# DeepSeek Document Chatbot")
            gr.Markdown("Ask questions about your documents or upload new ones.")
            
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot_ui = gr.Chatbot(height=600)
                    
                    with gr.Row():
                        msg = gr.Textbox(
                            placeholder="Ask a question about the documents...",
                            show_label=False
                        )
                        submit_btn = gr.Button("Send", variant="primary")
                    
                    with gr.Row():
                        clear_btn = gr.Button("Clear Conversation")
                        save_btn = gr.Button("Save Conversation")
                
                with gr.Column(scale=1):
                    gr.Markdown("### Upload Documents")
                    file_output = gr.Textbox(label="Upload Status")
                    upload_button = gr.UploadButton(
                        "Click to Upload",
                        file_types=[".txt", ".md", ".pdf"],
                        file_count="multiple"
                    )
                    gr.Markdown("""
                    ### Instructions
                    
                    - Upload text documents to the knowledge base
                    - Ask questions about document content
                    - Save conversation for later reference
                    - Text files (.txt) are recommended for best results
                    """)
            
            # Set up event handlers
            msg.submit(self.respond, [msg, chatbot_ui], [msg, chatbot_ui])
            submit_btn.click(self.respond, [msg, chatbot_ui], [msg, chatbot_ui])
            clear_btn.click(self.clear_conversation, None, chatbot_ui)
            save_btn.click(self.save_conversation, [chatbot_ui], file_output)
            upload_button.upload(self.upload_files, upload_button, file_output)
        
        # Launch the web interface
        demo.launch(share=False)

# Entry point when run directly
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Advanced DeepSeek Document Chatbot UI")
    parser.add_argument("--docs", type=str, default="./documents", 
                    help="Directory containing documents")
    parser.add_argument("--db", type=str, default="./chroma_db", 
                    help="Directory for vector database")
    parser.add_argument("--model", type=str, default="deepseek-ai/deepseek-llm-7b-chat", 
                    help="DeepSeek model name")
    args = parser.parse_args()
    
    # Initialize components
    doc_processor = DocumentProcessor(
        document_dir=args.docs,
        vector_db_path=args.db
    )
    vectorstore = doc_processor.process()
    
    deepseek_llm = DeepSeekLLM(model_name=args.model)
    llm = deepseek_llm.load_model()
    
    chatbot = DocumentChatbot(llm=llm, vectorstore=vectorstore)
    chatbot.create_chain()
    
    # Launch advanced UI
    ui = AdvancedWebUI(chatbot, doc_processor, args.docs)
    ui.launch()