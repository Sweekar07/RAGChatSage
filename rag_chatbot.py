import os
import pandas as pd
import logging
from typing import List, Dict

# LangChain imports
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGChatbot:
    """RAG-based Chatbot implementation using LangChain and Gemini"""
    
    def __init__(self, google_api_key: str, data_path: str):
        """
        Initialize the RAG Chatbot
        
        Args:
            google_api_key: API key for Google Generative AI
            data_path: Path to the dataset file (CSV format)
        """
        self.google_api_key = google_api_key
        self.data_path = data_path
        self.vector_store = None
        self.retriever = None
        self.chain = None
        
        # Set API key
        os.environ["GOOGLE_API_KEY"] = google_api_key
        
        # Load data and initialize components
        self._load_data()
        self._setup_retrieval()
        self._setup_chain()
        
    def _load_data(self) -> None:
        """Load data from CSV file and prepare it for embedding"""
        logger.info(f"Loading data from {self.data_path}")
        try:
            # Load the CSV file
            df = pd.read_csv(self.data_path)

            columns = df.columns.tolist()
            
            # Create documents for vector store
            self.documents = []
            for _, row in df.iterrows():
                # Combine all fields into a single text document
                document_text = " ".join([f"{col}: {row[col]}" for col in columns])
                self.documents.append(document_text)
                
            logger.info(f"Loaded {len(self.documents)} documents from dataset")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def _setup_retrieval(self) -> None:
        """Create embeddings and vector store for retrieval"""
        logger.info("Setting up vector store and retriever")
        try:
            # Create embeddings using Gemini
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

            # Create vector store
            self.vector_store = FAISS.from_texts(
                texts=self.documents,
                embedding=embeddings
            )
            
            # Create retriever
            self.retriever = self.vector_store.as_retriever(
                search_kwargs={"k": 3}  # Retrieve top 3 most relevant documents
            )
            
            logger.info("Vector store and retriever successfully set up")
        except Exception as e:
            logger.error(f"Error setting up retrieval: {e}")
            raise
    
    def _setup_chain(self) -> None:
        """Set up the RAG chain using LangChain components"""
        logger.info("Setting up the RAG chain")
        try:
            # Initialize Gemini LLM
            llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite")
            
            # Create prompt template
            template = """
            Answer the question based ONLY on the following context:
            
            {context}
            
            Question: {question}
            
            If the answer cannot be found in the context, respond with "I don't have enough information to answer this question."
            
            Your answer should be comprehensive, accurate, and directly based on the information provided in the context.
            """
            
            prompt = ChatPromptTemplate.from_template(template)
            
            # Format documents function
            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)
            
            # Create the RAG chain
            self.chain = (
                {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
            
            logger.info("RAG chain successfully set up")
        except Exception as e:
            logger.error(f"Error setting up chain: {e}")
            raise
    
    def query(self, question: str) -> str:
        """
        Process a query through the RAG pipeline
        
        Args:
            question: User's question
            
        Returns:
            str: Response from the RAG pipeline
        """
        logger.info(f"Processing query: {question}")
        try:
            # Use the chain to get an answer
            response = self.chain.invoke(question)
            return response
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return f"Sorry, I encountered an error: {str(e)}"
    
    def save_conversation(self, conversations: List[Dict[str, str]], output_file: str) -> None:
        """
        Save a list of conversations to a file
        
        Args:
            conversations: List of conversation dictionaries with 'question' and 'answer' keys
            output_file: Path to save the conversations
        """
        logger.info(f"Saving conversations to {output_file}")
        try:
            with open(output_file, 'w') as f:
                for i, conv in enumerate(conversations, 1):
                    f.write(f"Conversation {i}:\n")
                    f.write(f"User: {conv['question']}\n")
                    f.write(f"Chatbot: {conv['answer']}\n\n")
            logger.info(f"Successfully saved conversations to {output_file}")
        except Exception as e:
            logger.error(f"Error saving conversations: {e}")
            raise


def main():
    """Main function to demonstrate the RAG Chatbot"""
    
    # You should replace this with your actual API key
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    
    # Sample data path (replace with your dataset path)
    DATA_PATH = "sample_data.csv"
    
    try:
        # Initialize the chatbot
        print("Initializing RAG Chatbot...")
        chatbot = RAGChatbot(google_api_key=GOOGLE_API_KEY, data_path=DATA_PATH)
        
        # Demonstrate with sample questions
        sample_questions = [
            "What is the capital of France?",
            "Tell me about machine learning algorithms",
            "How does RAG work?",
            "What are the benefits of using LangChain?",
            "What is Flask?"
        ]
        
        # Process queries and collect conversations
        conversations = []
        print("\nProcessing sample questions:")
        for question in sample_questions:
            print(f"\nQ: {question}")
            answer = chatbot.query(question)
            print(f"A: {answer}")
            conversations.append({"question": question, "answer": answer})
        
        # Save conversations
        chatbot.save_conversation(conversations, "sample_conversations.txt")
        print("\nSample conversations saved to 'sample_conversations.txt'")
        
        # Interactive mode
        print("\nEnter 'quit' to exit")
        while True:
            user_input = input("\nEnter your question: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                break
            
            answer = chatbot.query(user_input)
            print(f"Answer: {answer}")
            conversations.append({"question": user_input, "answer": answer})
        
        # Save all conversations at the end
        chatbot.save_conversation(conversations, "all_conversations.txt")
        print("All conversations saved to 'all_conversations.txt'")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()