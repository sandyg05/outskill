import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# For environment variables
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import Document, HumanMessage, SystemMessage

# Core Libraries
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()


@dataclass
class Config:
    """Configuration class for the RAG system"""

    # File paths
    csv_file_path: str = "Myntra_300_prod_catalogue.csv"
    vector_store_path: str = "myntra_vector_store"

    # Text splitting parameters
    chunk_size: int = 200
    chunk_overlap: int = 50
    separator: str = "\n"

    # Embedding configuration
    embedding_model: str = "models/text-embedding-004"

    # OpenRouter Models - Updated model names for OpenRouter (last updated: 2024)
    openrouter_models: Dict[str, str] = None
    default_model: str = (
        "openai/gpt-5-mini"  # Default model - using the requested GPT-5-mini model
    )
    temperature: float = 0.1
    max_tokens: int = 4096

    # Search configuration
    num_results: int = 10
    search_type: str = "similarity"  # or "mmr" for Maximum Marginal Relevance

    # API Keys (will be loaded from environment)
    google_api_key: Optional[str] = None
    openrouter_api_key: Optional[str] = None

    def __post_init__(self):
        """Initialize OpenRouter models dictionary"""
        if self.openrouter_models is None:
            self.openrouter_models = {
                # OpenAI models
                "gpt-4o": "openai/gpt-4o",
                "gpt-4o-mini": "openai/gpt-4o-mini",
                "gpt-5-mini": "openai/gpt-5-mini",
                "gpt-4-turbo": "openai/gpt-4-turbo",
                "gpt-3.5-turbo": "openai/gpt-3.5-turbo",
                "gpt-4-vision": "openai/gpt-4-vision-preview",
                "gpt-4-omni": "openai/gpt-4-omni",
                # Anthropic models
                "claude-3.5-sonnet": "anthropic/claude-3.5-sonnet",
                "claude-3.5-haiku": "anthropic/claude-3.5-haiku",
                "claude-3-opus": "anthropic/claude-3-opus-20240229",
                "claude-3-sonnet": "anthropic/claude-3-sonnet-20240229",
                "claude-3-haiku": "anthropic/claude-3-haiku-20240307",
                # Google models
                "gemini-pro": "google/gemini-pro",
                "gemini-pro-1.5": "google/gemini-pro-1.5",
                "gemini-1.5-flash": "google/gemini-1.5-flash",
                "gemini-1.5-pro": "google/gemini-1.5-pro",
                # Meta models
                "llama-3.1-405b": "meta-llama/llama-3.1-405b-instruct",
                "llama-3.1-70b": "meta-llama/llama-3.1-70b-instruct",
                "llama-3.1-8b": "meta-llama/llama-3.1-8b-instruct",
                # Mistral models
                "mixtral-8x7b": "mistralai/mixtral-8x7b-instruct",
                "mistral-large": "mistralai/mistral-large",
                "mistral-small": "mistralai/mistral-small",
                "mistral-tiny": "mistralai/mistral-tiny",
                # Other models
                "deepseek-chat": "deepseek/deepseek-chat",
                "qwen-2.5-72b": "qwen/qwen-2.5-72b-instruct",
                "qwen-2.5-7b": "qwen/qwen-2.5-7b-instruct",
                "claude-instant": "anthropic/claude-instant-1.2",
            }


class MyntraRAG:
    """Main RAG system for Myntra product search and Q&A"""

    def __init__(self, config: Config = None):
        """Initialize the RAG system with configuration"""
        self.config = config or Config()
        self._load_api_keys()
        self._initialize_components()
        self.vector_store = None
        self.current_model = self.config.default_model

    def _load_api_keys(self):
        """Load API keys from environment variables"""
        self.config.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.config.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

        # Validate required API keys
        if not self.config.google_api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        if not self.config.openrouter_api_key:
            print(
                "Warning: OPENROUTER_API_KEY not found. LLM features will be disabled."
            )

    def _initialize_components(self):
        """Initialize embeddings, LLMs, and other components"""
        # Initialize embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=self.config.embedding_model, google_api_key=self.config.google_api_key
        )

        # Initialize text splitter
        self.text_splitter = CharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separator=self.config.separator,
            length_function=len,
        )

        # Initialize OpenRouter LLM dictionary
        self.llms = {}

        if self.config.openrouter_api_key:
            # Create LLM instances for commonly used models
            # We'll create them on-demand to save resources
            self.openrouter_base_url = "https://openrouter.ai/api/v1"
            print("OpenRouter API configured successfully")
        else:
            print("No OpenRouter API key found. LLM features disabled.")

        # Initialize prompt template
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""{context}

---

Given the context above, answer the question as best as possible.

Question: {question}

Answer: """,
        )

    def get_llm(self, model_name: str = None) -> ChatOpenAI:
        """Get or create an LLM instance for the specified model"""
        if not self.config.openrouter_api_key:
            raise ValueError("OpenRouter API key not configured")

        model_name = model_name or self.current_model

        # Check if model_name is a shorthand and convert to full name
        if model_name in self.config.openrouter_models:
            model_name = self.config.openrouter_models[model_name]

        # Create LLM instance if not cached
        if model_name not in self.llms:
            self.llms[model_name] = ChatOpenAI(
                model=model_name,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                api_key=self.config.openrouter_api_key,
                base_url=self.openrouter_base_url,
                default_headers={
                    "HTTP-Referer": "https://github.com/myntra-rag",  # Optional
                    "X-Title": "Myntra RAG System",  # Optional
                },
            )
            print(f"Initialized LLM: {model_name}")

        return self.llms[model_name]

    def list_available_models(self) -> Dict[str, str]:
        """List all available models"""
        return self.config.openrouter_models

    def load_and_process_csv(self, file_path: str = None) -> List[Document]:
        """Load CSV file and convert to documents"""
        file_path = file_path or self.config.csv_file_path

        print(f"Loading CSV file from: {file_path}")

        # Load CSV
        try:
            df = pd.read_csv(file_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV file not found at: {file_path}")

        print(f"Loaded {len(df)} rows from CSV")

        # Convert DataFrame to text documents
        documents = []
        for idx, row in df.iterrows():
            # Create text representation of each row
            text_parts = []
            for col, value in row.items():
                if pd.notna(value):  # Skip NaN values
                    text_parts.append(f"{col}: {value}")

            text = "\n".join(text_parts)

            # Create document with metadata
            doc = Document(
                page_content=text,
                metadata={
                    "row_index": idx,
                    **row.to_dict(),  # Include all row data as metadata
                },
            )
            documents.append(doc)

        return documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        print(f"Splitting {len(documents)} documents into chunks...")

        chunks = []
        for doc in documents:
            # Split the document text
            split_texts = self.text_splitter.split_text(doc.page_content)

            # Create new documents for each chunk
            for i, chunk_text in enumerate(split_texts):
                chunk_doc = Document(
                    page_content=chunk_text,
                    metadata={
                        **doc.metadata,
                        "chunk_index": i,
                        "total_chunks": len(split_texts),
                    },
                )
                chunks.append(chunk_doc)

        print(f"Created {len(chunks)} chunks")
        return chunks

    def create_vector_store(self, documents: List[Document]) -> FAISS:
        """Create FAISS vector store from documents"""
        print("Creating vector store with embeddings...")

        # Create FAISS vector store
        vector_store = FAISS.from_documents(
            documents=documents, embedding=self.embeddings
        )

        print("Vector store created successfully")
        return vector_store

    def save_vector_store(self, vector_store: FAISS = None):
        """Save vector store to disk"""
        vs = vector_store or self.vector_store
        if vs is None:
            raise ValueError("No vector store to save")

        print(f"Saving vector store to: {self.config.vector_store_path}")
        vs.save_local(self.config.vector_store_path)
        print("Vector store saved successfully")

    def load_vector_store(self) -> FAISS:
        """Load vector store from disk"""
        print(f"Loading vector store from: {self.config.vector_store_path}")

        if not Path(self.config.vector_store_path).exists():
            raise FileNotFoundError(
                f"Vector store not found at: {self.config.vector_store_path}"
            )

        vector_store = FAISS.load_local(
            self.config.vector_store_path,
            self.embeddings,
            allow_dangerous_deserialization=True,
        )

        print("Vector store loaded successfully")
        return vector_store

    def ingest_data(self, file_path: str = None):
        """Complete ingestion pipeline: load CSV, process, and create vector store"""
        # Load and process CSV
        documents = self.load_and_process_csv(file_path)

        # Split documents into chunks
        chunks = self.split_documents(documents)

        # Create vector store
        self.vector_store = self.create_vector_store(chunks)

        # Save vector store
        self.save_vector_store()

        print(f"Ingestion complete! {len(chunks)} chunks indexed.")

    def search(self, query: str, k: int = None) -> List[Document]:
        """Search for relevant documents"""
        if self.vector_store is None:
            # Try to load from disk
            try:
                self.vector_store = self.load_vector_store()
            except FileNotFoundError:
                raise ValueError(
                    "No vector store available. Please run ingest_data() first."
                )

        k = k or self.config.num_results

        if self.config.search_type == "mmr":
            # Maximum Marginal Relevance search for diversity
            results = self.vector_store.max_marginal_relevance_search(
                query=query, k=k, fetch_k=k * 2
            )
        else:
            # Standard similarity search
            results = self.vector_store.similarity_search(query=query, k=k)

        return results

    def format_context(self, documents: List[Document]) -> str:
        """Format retrieved documents as context"""
        context_parts = []
        for i, doc in enumerate(documents, 1):
            context_parts.append(f"Text: {doc.page_content}")

        return "\n".join(context_parts)

    def query(
        self, question: str, model: str = None, verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Main query method - searches for relevant documents and generates answer

        Args:
            question: User's question
            model: Model to use (shorthand like 'gpt-4o' or full like 'openai/gpt-4o')
            verbose: Whether to print intermediate steps

        Returns:
            Dictionary with answer and metadata
        """
        # Search for relevant documents
        if verbose:
            print(f"Searching for relevant documents for: {question}")

        retrieved_docs = self.search(question)

        if verbose:
            print(f"Found {len(retrieved_docs)} relevant documents")

        # Format context
        context = self.format_context(retrieved_docs)

        # Get LLM
        model = model or self.current_model

        if not self.config.openrouter_api_key:
            raise ValueError(
                "No OpenRouter API key available. Please set OPENROUTER_API_KEY"
            )

        try:
            llm = self.get_llm(model)

            if verbose:
                print(f"Using model: {model}")

            # Generate answer
            prompt = self.prompt_template.format(context=context, question=question)

            response = llm.invoke(prompt)

            # Extract answer text
            if hasattr(response, "content"):
                answer = response.content
            else:
                answer = str(response)

            return {
                "question": question,
                "answer": answer,
                "sources": retrieved_docs,
                "model": model,
                "num_sources": len(retrieved_docs),
            }

        except Exception as e:
            print(f"Error with model {model}: {e}")
            raise

    def chat(self):
        """Interactive chat interface"""
        print("\n=== Myntra RAG Chat Interface (OpenRouter) ===")
        print(f"Current model: {self.current_model}")
        print("Type 'quit' to exit, 'help' for commands\n")

        while True:
            try:
                user_input = input("You: ").strip()

                if user_input.lower() == "quit":
                    print("Goodbye!")
                    break

                elif user_input.lower() == "help":
                    print("\nCommands:")
                    print("  quit - Exit the chat")
                    print("  help - Show this help message")
                    print("  models - List available models")
                    print("  model:<name> - Switch model (e.g., model:gpt-4o)")

                    # Group models by provider for the help display
                    providers = {
                        "OpenAI": [
                            k
                            for k in self.config.openrouter_models.keys()
                            if k.startswith("gpt")
                        ],
                        "Anthropic": [
                            k
                            for k in self.config.openrouter_models.keys()
                            if k.startswith("claude")
                        ],
                        "Google": [
                            k
                            for k in self.config.openrouter_models.keys()
                            if k.startswith("gemini")
                        ],
                        "Other": [
                            k
                            for k in self.config.openrouter_models.keys()
                            if not any(
                                k.startswith(p) for p in ["gpt", "claude", "gemini"]
                            )
                        ],
                    }

                    print("\nAvailable model shortcuts (examples):")
                    for provider, models in providers.items():
                        if models:
                            print(
                                f"  {provider}: {', '.join(sorted(models)[:3])}"
                                + (" ..." if len(models) > 3 else "")
                            )
                    print("\nUse 'models' command to see all available models")
                    print("\nJust type your question about products!\n")
                    continue

                elif user_input.lower() == "models":
                    print("\nAvailable models:")

                    # Group models by provider for better organization
                    providers = {
                        "OpenAI": [
                            k
                            for k in self.config.openrouter_models.keys()
                            if k.startswith("gpt")
                        ],
                        "Anthropic": [
                            k
                            for k in self.config.openrouter_models.keys()
                            if k.startswith("claude")
                        ],
                        "Google": [
                            k
                            for k in self.config.openrouter_models.keys()
                            if k.startswith("gemini")
                        ],
                        "Meta": [
                            k
                            for k in self.config.openrouter_models.keys()
                            if k.startswith("llama")
                        ],
                        "Mistral": [
                            k
                            for k in self.config.openrouter_models.keys()
                            if k.startswith("mistral") or k.startswith("mixtral")
                        ],
                        "Other": [
                            k
                            for k in self.config.openrouter_models.keys()
                            if not any(
                                k.startswith(p)
                                for p in [
                                    "gpt",
                                    "claude",
                                    "gemini",
                                    "llama",
                                    "mistral",
                                    "mixtral",
                                ]
                            )
                        ],
                    }

                    # Display models by provider
                    for provider, models in providers.items():
                        if models:
                            print(f"\n  {provider} Models:")
                            for short_name in sorted(models):
                                full_name = self.config.openrouter_models[short_name]
                                current = (
                                    " (current)"
                                    if full_name == self.current_model
                                    or short_name == self.current_model
                                    else ""
                                )
                                print(f"    {short_name}: {full_name}{current}")
                    print()
                    continue

                elif user_input.lower().startswith("model:"):
                    model_name = user_input.split(":", 1)[1].strip()

                    # Check if it's a valid model
                    if model_name in self.config.openrouter_models:
                        self.current_model = self.config.openrouter_models[model_name]
                        print(f"Switched to {model_name} ({self.current_model})\n")
                    elif model_name in self.config.openrouter_models.values():
                        self.current_model = model_name
                        print(f"Switched to {model_name}\n")
                    else:
                        print(f"Unknown model: {model_name}")
                        print("Use 'models' command to see available models\n")
                    continue

                # Process the query
                result = self.query(
                    question=user_input, model=self.current_model, verbose=False
                )

                print(
                    f"\nAssistant ({result['model'].split('/')[-1]}): {result['answer']}"
                )
                print(f"(Based on {result['num_sources']} sources)\n")

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}\n")


# Example usage and setup functions
def setup_environment():
    """Create .env file template if it doesn't exist"""
    env_path = Path(".env")
    if not env_path.exists():
        env_template = """# API Keys for Myntra RAG System
# Get your Google API key from: https://makersuite.google.com/app/apikey
GOOGLE_API_KEY=your_google_api_key_here

# Get your OpenRouter API key from: https://openrouter.ai/keys
OPENROUTER_API_KEY=your_openrouter_api_key_here
"""
        env_path.write_text(env_template)
        print(".env file created. Please add your API keys.")
        return False
    return True


def create_sample_csv():
    """Create a sample CSV file for testing"""
    sample_data = {
        "product_id": [
            "PROD001",
            "PROD002",
            "PROD003",
            "PROD004",
            "PROD005",
            "PROD006",
            "PROD007",
            "PROD008",
        ],
        "product_name": [
            "Men UV Protection Sports Shirt",
            "Women UV Shield Running Top",
            "Men Quick Dry UV Shirt",
            "Unisex UV Protection Hoodie",
            "Men Athletic UV Guard Tee",
            "Women UV Blocking Swim Shirt",
            "Kids UV Protection Rashguard",
            "Men UPF 50+ Fishing Shirt",
        ],
        "category": [
            "Men Shirts",
            "Women Activewear",
            "Men Shirts",
            "Unisex Outerwear",
            "Men T-Shirts",
            "Women Swimwear",
            "Kids Activewear",
            "Men Outdoor",
        ],
        "brand": [
            "Nike",
            "Adidas",
            "Puma",
            "Under Armour",
            "Reebok",
            "Speedo",
            "Decathlon",
            "Columbia",
        ],
        "price": [2499, 1999, 1799, 3299, 1599, 2799, 999, 3999],
        "color": ["Blue", "Pink", "Black", "Gray", "White", "Navy", "Red", "Khaki"],
        "size": [
            "M, L, XL",
            "S, M, L",
            "M, L, XL, XXL",
            "S, M, L, XL",
            "M, L, XL",
            "XS, S, M, L",
            "8-10, 10-12, 12-14",
            "M, L, XL, XXL",
        ],
        "features": [
            "UPF 50+ protection, Moisture-wicking, Anti-odor",
            "UPF 40+ protection, Breathable mesh panels, Reflective details",
            "UPF 50+ protection, Quick-dry fabric, Lightweight",
            "UPF 45+ protection, Thumbholes, Zippered pocket",
            "UPF 30+ protection, Stretchable fabric, Sweat-wicking",
            "UPF 50+ protection, Chlorine-resistant, Fast-drying",
            "UPF 50+ protection, Flatlock seams, 4-way stretch",
            "UPF 50+ protection, Vented back, Multiple pockets",
        ],
        "description": [
            "High-performance UV protection shirt for outdoor sports and activities",
            "Stylish and functional UV protection top for women runners",
            "Versatile UV protection shirt suitable for various outdoor activities",
            "Premium UV protection hoodie for all-weather outdoor adventures",
            "Basic UV protection t-shirt for everyday athletic activities",
            "Professional-grade UV protection swim shirt for water sports",
            "Comfortable UV protection rashguard designed specifically for kids",
            "Technical fishing shirt with maximum UV protection and utility features",
        ],
        "rating": [4.5, 4.3, 4.2, 4.7, 4.0, 4.6, 4.4, 4.8],
        "in_stock": [True, True, False, True, True, False, True, True],
    }

    df = pd.DataFrame(sample_data)
    df.to_csv("myntra_sample_catalog.csv", index=False)
    print("Sample CSV file created: myntra_sample_catalog.csv")


def test_openrouter_connection():
    """Test OpenRouter API connection with a simple query"""
    from dotenv import load_dotenv

    load_dotenv()

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("OPENROUTER_API_KEY not found in environment")
        return False

    try:
        # Test with a simple model
        llm = ChatOpenAI(
            model="openai/gpt-3.5-turbo",
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            max_tokens=50,
        )

        response = llm.invoke(
            "Say 'OpenRouter connected successfully!' and nothing else."
        )
        print(f"Connection test: {response.content}")
        return True
    except Exception as e:
        print(f"Connection test failed: {e}")
        return False


def main():
    """Main function to run the Myntra RAG system"""
    print("=== Myntra RAG System with OpenRouter ===\n")

    # Setup environment
    if not setup_environment():
        print("Please configure your API keys in the .env file and run again.")
        return

    # Load environment variables
    load_dotenv()

    # Test OpenRouter connection
    print("Testing OpenRouter connection...")
    if not test_openrouter_connection():
        print("\nFailed to connect to OpenRouter. Please check your API key.")
        print("Get your API key from: https://openrouter.ai/keys")
        return
    print()

    # Check for sample data
    if not Path("myntra_sample_catalog.csv").exists():
        print("Creating sample CSV file...")
        create_sample_csv()

    # Initialize configuration
    config = Config(
        csv_file_path="myntra_sample_catalog.csv",
        vector_store_path="myntra_vector_store",
        chunk_size=200,
        chunk_overlap=50,
        default_model="gpt-3.5-turbo",  # Start with a cheaper model
    )

    # Initialize RAG system
    try:
        rag = MyntraRAG(config)
    except ValueError as e:
        print(f"Error: {e}")
        print("\nPlease ensure you have set up the following environment variables:")
        print("  - GOOGLE_API_KEY: For embeddings")
        print("  - OPENROUTER_API_KEY: For LLM queries")
        return

    # Check if vector store exists
    if not Path(config.vector_store_path).exists():
        print("\nVector store not found. Ingesting data...")
        rag.ingest_data()
    else:
        print("\nVector store found. Loading...")
        rag.vector_store = rag.load_vector_store()

    # Example queries with different models
    print("\n=== Testing Different Models ===\n")

    test_question = "What UV protection shirts are available for men?"
    test_models = ["gpt-3.5-turbo", "claude-3-haiku", "mixtral-8x7b"]

    for model in test_models:
        if model in config.openrouter_models:
            try:
                print(f"\nTesting with {model}:")
                print(f"Question: {test_question}")
                result = rag.query(test_question, model=model, verbose=False)
                print(f"Answer: {result['answer'][:200]}...")  # Show first 200 chars
                print("-" * 50)
            except Exception as e:
                print(f"Error with {model}: {e}")

    # Start interactive chat
    print("\n\nStarting interactive chat mode...")
    print("You can switch models anytime using 'model:<name>' command")
    print("Example: model:gpt-4o or model:claude-3.5-sonnet\n")
    rag.chat()


if __name__ == "__main__":
    main()
