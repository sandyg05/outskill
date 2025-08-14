"""
Step 3: Zerodha Business Intelligence RAG with Multi-Step Reasoning
Advanced Retrieval + Query Decomposition + Answer Validation + Business Logic

Requirements:
pip install llama-index
pip install llama-index-readers-web
pip install crewai-tools
pip install pandas
pip install openai

Step 3 Advanced Features over Step 2:

🚀 Zerodha Business Intelligence Focus:
   - Brokerage and fintech data integration
   - Market trading intelligence gathering
   - Discount brokerage strategic analysis
   - Financial services insights

🚀 Multi-Step Reasoning:
   - Query decomposition and routing
   - Cross-source data correlation
   - Intelligent source selection
   - Context-aware responses

🚀 Validation & Quality Control:
   - Response confidence scoring
   - Missing information detection
   - Suggested follow-up questions
   - Risk factor identification

🚀 Conversation Memory:
   - Session history tracking
   - Executive summary generation
   - Context retention across queries
   - Strategic insight accumulation

🚀 Structured Data Integration:
   - Brokerage metrics analysis
   - Pandas-based data processing
   - Custom fintech business logic tools
   - Database-ready architecture

Usage Instructions:
1. Zerodha business documents are automatically loaded from docs/business
2. Set OpenAI API key (required)
3. Set EXA_API_KEY (optional, for market research)
4. Customize brokerage_data in analyze_brokerage_metrics()
5. Run the script

Your Zerodha Business Intelligence Assistant is ready!
"""

import json
import os
from typing import Dict, List

import pandas as pd
from crewai_tools import EXASearchTool
from dotenv import load_dotenv
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.query_engine import RouterQueryEngine, SubQuestionQueryEngine
from llama_index.core.response.pprint_utils import pprint_response
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import FunctionTool, QueryEngineTool, ToolMetadata
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.readers.web import SimpleWebPageReader

load_dotenv()


# Set your API keys
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["EXA_API_KEY"] = os.getenv("EXA_API_KEY")

# Configure LlamaIndex settings with more powerful model
Settings.llm = OpenAI(model="gpt-5-mini")
Settings.embed_model = OpenAIEmbedding()


class ZerodhaBusinessIntelligenceRAG:
    def __init__(self):
        self.document_engine = None
        self.brokerage_engine = None
        self.market_engine = None
        self.router_engine = None
        self.conversation_history = []

    def setup_document_knowledge(
        self,
        doc_folder="/Users/ishandutta/Documents/code/outskill_agents/docs/business",
    ):
        """Create knowledge base from Zerodha business documents"""
        print("📊 Loading Zerodha annual reports and business documents...")
        documents = SimpleDirectoryReader(doc_folder).load_data()

        # Enhanced indexing with metadata
        doc_index = VectorStoreIndex.from_documents(documents, show_progress=True)

        self.document_engine = doc_index.as_query_engine(
            similarity_top_k=5, response_mode="tree_summarize", verbose=True
        )
        print(f"✅ Loaded {len(documents)} Zerodha business documents")

    def setup_brokerage_data_tool(self):
        """Create tool for structured brokerage business data analysis"""

        def analyze_brokerage_metrics(query: str) -> str:
            """Analyze Zerodha brokerage business metrics from structured data"""
            # Sample Zerodha-style brokerage data (based on typical discount brokerage metrics)
            brokerage_data = {
                "Q1_2024": {
                    "active_clients": 6500000,
                    "revenue_crores": 850,
                    "brokerage_revenue": 320,
                    "mutual_fund_revenue": 180,
                    "other_revenue": 350,
                    "profit_margin": 0.42,
                    "avg_revenue_per_client": 1308,
                },
                "Q2_2024": {
                    "active_clients": 6800000,
                    "revenue_crores": 920,
                    "brokerage_revenue": 340,
                    "mutual_fund_revenue": 200,
                    "other_revenue": 380,
                    "profit_margin": 0.45,
                    "avg_revenue_per_client": 1353,
                },
                "Q3_2024": {
                    "active_clients": 7100000,
                    "revenue_crores": 980,
                    "brokerage_revenue": 365,
                    "mutual_fund_revenue": 215,
                    "other_revenue": 400,
                    "profit_margin": 0.43,
                    "avg_revenue_per_client": 1380,
                },
                "Q4_2024": {
                    "active_clients": 7500000,
                    "revenue_crores": 1050,
                    "brokerage_revenue": 385,
                    "mutual_fund_revenue": 235,
                    "other_revenue": 430,
                    "profit_margin": 0.46,
                    "avg_revenue_per_client": 1400,
                },
            }

            # Convert to DataFrame for analysis
            df = pd.DataFrame(brokerage_data).T

            # Perform analysis based on query
            if "client" in query.lower():
                total_clients = df["active_clients"].iloc[-1]
                client_growth = (
                    df["active_clients"].iloc[-1] / df["active_clients"].iloc[0] - 1
                ) * 100
                return f"Active clients Q4 2024: {total_clients:,}. Annual client growth: {client_growth:.1f}%. Zerodha continues strong user acquisition."
            elif "revenue" in query.lower() and "per client" in query.lower():
                avg_arpc = df["avg_revenue_per_client"].mean()
                arpc_trend = (
                    df["avg_revenue_per_client"].iloc[-1]
                    - df["avg_revenue_per_client"].iloc[0]
                )
                return f"Average revenue per client: ₹{avg_arpc:.0f}. ARPC increased by ₹{arpc_trend:.0f} over the year, showing improved monetization."
            elif "brokerage" in query.lower():
                total_brokerage = df["brokerage_revenue"].sum()
                brokerage_share = (
                    df["brokerage_revenue"] / df["revenue_crores"]
                ).mean()
                return f"Total brokerage revenue 2024: ₹{total_brokerage} crores. Brokerage represents {brokerage_share:.1%} of total revenue."
            elif "mutual fund" in query.lower() or "mf" in query.lower():
                total_mf = df["mutual_fund_revenue"].sum()
                mf_growth = (
                    df["mutual_fund_revenue"].iloc[-1]
                    / df["mutual_fund_revenue"].iloc[0]
                    - 1
                ) * 100
                return f"Mutual fund revenue 2024: ₹{total_mf} crores. MF revenue growth: {mf_growth:.1f}%. Strong diversification into wealth management."
            elif "profit" in query.lower() or "margin" in query.lower():
                avg_margin = df["profit_margin"].mean()
                total_revenue = df["revenue_crores"].sum()
                return f"Average profit margin: {avg_margin:.1%}. Total revenue: ₹{total_revenue} crores. Zerodha maintains industry-leading profitability."
            else:
                total_revenue = df["revenue_crores"].sum()
                total_clients = df["active_clients"].iloc[-1]
                avg_margin = df["profit_margin"].mean()
                return f"Zerodha 2024 Summary: ₹{total_revenue} crores revenue, {total_clients:,} active clients, {avg_margin:.1%} avg profit margin. Leading discount broker in India."

        # Create function tool
        brokerage_tool = FunctionTool.from_defaults(
            fn=analyze_brokerage_metrics,
            name="brokerage_analyzer",
            description="Analyze Zerodha brokerage business metrics including client base, revenue streams, and profitability",
        )

        return brokerage_tool

    def setup_market_intelligence(self):
        """Create market research and competitor analysis engine for fintech/brokerage"""
        print("🌐 Setting up fintech and brokerage market intelligence...")

        # Fintech and brokerage market research URLs
        market_urls = [
            "https://www.investopedia.com/articles/active-trading/020515/how-robinhood-makes-money.asp",
            "https://zerodha.com/z-connect/",
            "https://www.livemint.com/market/stock-market-news",
        ]

        try:
            market_documents = SimpleWebPageReader().load_data(market_urls)
            market_index = VectorStoreIndex.from_documents(market_documents)

            self.market_engine = market_index.as_query_engine(
                similarity_top_k=3, response_mode="compact"
            )
            print("✅ Fintech market intelligence ready")
        except:
            print("⚠️ Market intelligence setup failed (optional)")

    def setup_intelligent_routing(self):
        """Create intelligent query routing system for Zerodha business"""
        print("🧠 Setting up intelligent routing for Zerodha analysis...")

        # Create specialized query engines with descriptions
        query_engine_tools = []

        # Zerodha business documents engine
        if self.document_engine:
            doc_tool = QueryEngineTool(
                query_engine=self.document_engine,
                metadata=ToolMetadata(
                    name="zerodha_documents",
                    description="Search Zerodha annual reports, AMC documents, and internal business information",
                ),
            )
            query_engine_tools.append(doc_tool)

        # Market intelligence engine
        if self.market_engine:
            market_tool = QueryEngineTool(
                query_engine=self.market_engine,
                metadata=ToolMetadata(
                    name="fintech_market_intelligence",
                    description="Research fintech market trends, brokerage industry analysis, and competitive landscape",
                ),
            )
            query_engine_tools.append(market_tool)

        # EXA search for live data
        try:
            exa_search_tool = EXASearchTool()
            print("✅ Live search enabled for real-time market data")
        except:
            print("⚠️ Live search not available")

        # Create router query engine with LLM selector
        self.router_engine = RouterQueryEngine(
            selector=LLMSingleSelector.from_defaults(),
            query_engine_tools=query_engine_tools,
            verbose=True,
        )

        # Store brokerage tool separately for manual routing
        self.brokerage_tool = self.setup_brokerage_data_tool()

        print("✅ Zerodha intelligent routing system ready!")

    def validate_and_enhance_response(self, response, original_query):
        """Validate response quality and enhance with Zerodha-specific context"""

        validation_prompt = f"""
        Analyze this response for a Zerodha business intelligence query and return ONLY valid JSON:
        
        Original Query: {original_query}
        Response: {response}
        
        Consider Zerodha's position as India's largest discount broker and provide a JSON response with exactly this structure:
        {{
            "confidence_score": 8,
            "missing_info": "Specific missing information about discount brokerage business",
            "follow_ups": ["Question 1?", "Question 2?"],
            "risks": ["Risk 1", "Risk 2"]
        }}
        
        Rate confidence 1-10 based on:
        - Data completeness and accuracy
        - Relevance to Zerodha's business model
        - Actionability of insights
        - Source quality and reliability
        
        Return ONLY the JSON object, no other text.
        """

        validation_response = Settings.llm.complete(validation_prompt)

        try:
            # Clean the response - remove any markdown formatting or extra text
            response_text = str(validation_response).strip()
            if response_text.startswith("```json"):
                response_text = (
                    response_text.replace("```json", "").replace("```", "").strip()
                )
            elif response_text.startswith("```"):
                response_text = response_text.replace("```", "").strip()

            validation_data = json.loads(response_text)

            # Validate that required keys exist
            required_keys = ["confidence_score", "missing_info", "follow_ups", "risks"]
            if all(key in validation_data for key in required_keys):
                return validation_data
            else:
                print(f"⚠️ Validation response missing required keys: {validation_data}")
                raise ValueError("Missing required keys")

        except Exception as e:
            print(f"⚠️ JSON parsing failed: {e}")
            print(f"Raw response: {str(validation_response)[:200]}...")
            return {
                "confidence_score": 7,
                "missing_info": "Consider adding regulatory compliance context for Indian markets",
                "follow_ups": [
                    "How does this compare to other discount brokers?",
                    "What are the regulatory implications?",
                    "Impact on client acquisition costs?",
                ],
                "risks": [
                    "SEBI regulatory changes",
                    "Market volatility impact",
                    "Competition from traditional brokers",
                ],
            }

    def ask_zerodha_question(self, question: str, include_validation: bool = True):
        """Ask a Zerodha business intelligence question with advanced processing"""
        print(f"\n❓ Zerodha Business Query: {question}")
        print("🔍 Analyzing across Zerodha business intelligence sources...")

        # Store in conversation history
        self.conversation_history.append({"role": "user", "content": question})

        # Check if question is brokerage-related and use brokerage tool if needed
        brokerage_keywords = [
            "revenue",
            "profit",
            "client",
            "brokerage",
            "mutual fund",
            "mf",
            "margin",
            "arpc",
            "quarter",
            "q1",
            "q2",
            "q3",
            "q4",
            "metrics",
        ]
        is_brokerage_query = any(
            keyword in question.lower() for keyword in brokerage_keywords
        )

        if is_brokerage_query and hasattr(self, "brokerage_tool"):
            print("💰 Using Zerodha brokerage data analysis...")
            try:
                brokerage_response = self.brokerage_tool.call(question)
                # Combine with router engine response for comprehensive analysis
                router_response = self.router_engine.query(question)
                response_text = f"Brokerage Metrics Analysis: {brokerage_response}\n\nAdditional Context from Annual Reports: {router_response}"

                # Create a response-like object
                class CombinedResponse:
                    def __init__(self, text):
                        self.response = text
                        self.source_nodes = getattr(router_response, "source_nodes", [])

                    def __str__(self):
                        return self.response

                response = CombinedResponse(response_text)
            except Exception as e:
                print(f"⚠️ Brokerage tool error: {e}")
                response = self.router_engine.query(question)
        else:
            # Get response from router engine
            response = self.router_engine.query(question)

        print(f"\n🤖 Zerodha Analysis: {response}")

        # Show detailed source information
        if hasattr(response, "source_nodes") and response.source_nodes:
            print(f"\n📊 Data Sources Used:")
            for i, node in enumerate(response.source_nodes[:3], 1):
                source = node.metadata.get("file_name", "Zerodha Structured Data")
                confidence = node.score if hasattr(node, "score") else "N/A"
                print(f"   {i}. {source} (Relevance: {confidence})")

        # Validate and enhance response
        if include_validation:
            print(f"\n🔍 Response Validation:")
            validation = self.validate_and_enhance_response(str(response), question)

            print(
                f"   Confidence Score: {validation.get('confidence_score', 'N/A')}/10"
            )

            if validation.get("follow_ups"):
                print(f"   💡 Suggested Follow-ups:")
                for follow_up in validation["follow_ups"][:2]:
                    print(f"      • {follow_up}")

            if validation.get("risks"):
                print(f"   ⚠️ Considerations:")
                for risk in validation["risks"][:2]:
                    print(f"      • {risk}")

        # Store in conversation history
        self.conversation_history.append(
            {"role": "assistant", "content": str(response)}
        )

        return response

    def generate_executive_summary(self):
        """Generate executive summary from Zerodha analysis session"""
        if not self.conversation_history:
            return "No Zerodha analysis performed yet."

        summary_prompt = f"""
        Based on this Zerodha business intelligence session, create an executive summary:
        
        Conversation History: {self.conversation_history}
        
        Provide:
        1. Key Findings about Zerodha's business performance (3-4 bullet points)
        2. Strategic Recommendations for India's leading discount broker (2-3 actions)
        3. Priority Areas for Further Investigation in fintech/brokerage space
        
        Keep it concise and actionable for Zerodha's leadership team.
        """

        summary = Settings.llm.complete(summary_prompt)
        return str(summary)


def main():
    """Main execution with Zerodha business intelligence scenarios"""

    # Initialize Zerodha Business Intelligence RAG
    zerodha_rag = ZerodhaBusinessIntelligenceRAG()

    # Setup all knowledge sources
    zerodha_rag.setup_document_knowledge()  # Zerodha annual reports
    zerodha_rag.setup_market_intelligence()  # Fintech market data
    zerodha_rag.setup_intelligent_routing()  # Smart routing

    print("\n" + "=" * 60)
    print("🚀 ZERODHA BUSINESS INTELLIGENCE RAG SYSTEM READY")
    print("=" * 60)

    # Zerodha-specific business intelligence queries
    zerodha_questions = [
        "What are Zerodha's Q4 2024 financial performance and key revenue streams?",
        "How is Zerodha's client acquisition and revenue per client trending?",
        # "What strategic risks should Zerodha consider in the Indian discount brokerage market?",
        # "Analyze Zerodha's mutual fund business growth and its impact on revenue diversification",
        # "What are the key regulatory challenges mentioned in Zerodha's annual reports?",
        "How does Zerodha's profit margin compare to industry standards for discount brokers?",
    ]

    # Process each Zerodha business question
    for question in zerodha_questions:
        zerodha_rag.ask_zerodha_question(question)
        print("\n" + "-" * 60 + "\n")

    # Generate executive summary
    print("📋 ZERODHA EXECUTIVE SUMMARY")
    print("-" * 30)
    summary = zerodha_rag.generate_executive_summary()
    print(summary)


if __name__ == "__main__":
    main()
