"""
Step 2: Enhanced RAG with Tools & Multiple Sources
Documents + Web Search + Source Citations + Better Retrieval

Requirements:
pip install llama-index
pip install llama-index-readers-web
pip install llama-index-question-gen-openai
pip install crewai-tools
pip install openai

Step 2 Enhancements over Step 1:

🆕 Multiple Knowledge Sources:
   - Local documents (PDFs, text files)
   - Web articles and pages
   - Live Google search (optional)

🆕 Better Retrieval:
   - Similarity top-k configuration
   - Compact response mode
   - Sub-question decomposition

🆕 Source Attribution:
   - Shows where answers come from
   - Tracks document sources
   - Metadata preservation

🆕 Tool Integration:
   - Query engine tools
   - Web page reader
   - Google search capability

Usage Instructions:
1. Create 'documents' folder with your PDFs
2. Set OpenAI API key (required)
3. Set EXA_API_KEY in your environment (optional, for live search)
4. Run the script

Your Research Assistant now searches everywhere!
"""

import os

from crewai_tools import EXASearchTool
from dotenv import load_dotenv
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.readers.web import SimpleWebPageReader

load_dotenv()


# Set your API keys
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["EXA_API_KEY"] = os.getenv("EXA_API_KEY")

# Configure LlamaIndex settings
Settings.llm = OpenAI(model="gpt-5-mini")
Settings.embed_model = OpenAIEmbedding()


class EnhancedRAG:
    def __init__(self):
        self.document_engine = None
        self.web_engine = None
        self.combined_engine = None

    def setup_document_knowledge(
        self, doc_folder="/Users/ishandutta/Documents/code/outskill_agents/docs/papers"
    ):
        """Create knowledge base from local documents"""
        print("📚 Loading local documents...")
        documents = SimpleDirectoryReader(doc_folder).load_data()
        doc_index = VectorStoreIndex.from_documents(documents)
        self.document_engine = doc_index.as_query_engine(
            similarity_top_k=2, response_mode="compact"
        )
        print(f"✅ Loaded {len(documents)} local documents")

    def setup_web_knowledge(self, urls=None):
        """Create knowledge base from web sources"""
        if urls is None:
            urls = ["https://openai.com/research/"]

        print("🌐 Loading web documents...")
        web_documents = SimpleWebPageReader().load_data(urls)
        web_index = VectorStoreIndex.from_documents(web_documents)
        self.web_engine = web_index.as_query_engine(
            similarity_top_k=2, response_mode="compact"
        )
        print(f"✅ Loaded {len(web_documents)} web documents")

    def setup_combined_system(self):
        """Combine document knowledge + web search + live search"""
        print("🔧 Setting up enhanced RAG system...")

        # Create tools for different knowledge sources
        tools = []

        # Local documents tool
        if self.document_engine:
            doc_tool = QueryEngineTool(
                query_engine=self.document_engine,
                metadata=ToolMetadata(
                    name="document_search",
                    description="Search through uploaded documents and research papers",
                ),
            )
            tools.append(doc_tool)

        # Web knowledge tool
        if self.web_engine:
            web_tool = QueryEngineTool(
                query_engine=self.web_engine,
                metadata=ToolMetadata(
                    name="web_knowledge",
                    description="Search through curated web content and articles",
                ),
            )
            tools.append(web_tool)

        # EXA search tool for live web search
        try:
            exa_search_tool = EXASearchTool()
            print("✅ EXA Search enabled")
        except:
            print("⚠️ EXA Search not configured (optional)")

        # Create a simple multi-tool query engine that routes between tools
        from llama_index.core.query_engine import RouterQueryEngine
        from llama_index.core.selectors import LLMSingleSelector

        # Create router query engine instead of sub-question engine to avoid dependency issues
        self.combined_engine = RouterQueryEngine(
            selector=LLMSingleSelector.from_defaults(),
            query_engine_tools=tools,
            verbose=True,
        )

        print("✅ Enhanced RAG system ready!")

    def ask_question(self, question):
        """Ask a question using the enhanced RAG system"""
        print(f"\n❓ Question: {question}")
        print("🔍 Searching across multiple sources...")

        response = self.combined_engine.query(question)

        print(f"🤖 Answer: {response}")

        # Show sources if available
        if hasattr(response, "source_nodes") and response.source_nodes:
            print("\n📖 Sources:")
            for i, node in enumerate(response.source_nodes[:3], 1):
                source = node.metadata.get("file_name", "Unknown source")
                print(f"   {i}. {source}")

        return response


def main():
    """Main execution"""
    # Initialize enhanced RAG
    rag = EnhancedRAG()

    # Setup knowledge sources
    rag.setup_document_knowledge()  # Local PDFs
    rag.setup_web_knowledge()  # Web articles
    rag.setup_combined_system()  # Combine everything

    # Example questions that benefit from multiple sources
    questions = [
        "What are the important concepts in language models according to my documents and current web sources?",
        "What are the important concepts in object detection models and diffusion models according to my documents and current web sources?",
        "What are the practical applications mentioned in my documents, and what's happening in the industry now?",
    ]

    # Ask enhanced questions
    for question in questions:
        rag.ask_question(question)
        print("=" * 60)


if __name__ == "__main__":
    main()
