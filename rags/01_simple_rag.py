"""
Step 1: Basic Personal Document RAG
Simple PDF upload → Ask Questions → Get Answers

Requirements:
pip install llama-index
pip install openai
"""

import os

from dotenv import load_dotenv
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

load_dotenv()

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Configure LlamaIndex settings
Settings.llm = OpenAI(model="gpt-5-mini")
Settings.embed_model = OpenAIEmbedding()


def create_personal_rag():
    """Create a basic RAG system from documents"""

    # Step 1: Load documents from a folder
    print("📚 Loading documents...")
    documents = SimpleDirectoryReader(
        "/Users/ishandutta/Documents/code/outskill_agents/docs/resume"
    ).load_data()
    print(f"✅ Loaded {len(documents)} documents")

    # Step 2: Create vector index (automatically chunks and embeds)
    print("🔍 Creating search index...")
    index = VectorStoreIndex.from_documents(documents)
    print("✅ Index created successfully")

    # Step 3: Create query engine
    query_engine = index.as_query_engine()

    return query_engine


def ask_question(query_engine, question):
    """Ask a question and get an answer"""
    print(f"\n❓ Question: {question}")
    response = query_engine.query(question)
    print(f"🤖 Answer: {response}")
    return response


# Main execution
if __name__ == "__main__":
    # Create the RAG system
    rag_system = create_personal_rag()

    # Example questions
    questions = [
        "What are the key skills listed in the resume?",
        "What is the latest work experience of the candidate?",
        "What educational qualifications does the candidate have?",
    ]

    # Ask questions
    for question in questions:
        ask_question(rag_system, question)
        print("-" * 50)

"""
Usage Instructions:
1. Create a folder called 'documents' in your project directory
2. Add your PDF files to this folder
3. Set your OpenAI API key
4. Run the script

That's it! Your personal research assistant is ready.
"""
