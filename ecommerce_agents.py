import os
import asyncio
import requests
import streamlit as st
from typing import Annotated, List
from pydantic import Field
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from agent_framework import ChatAgent
from agent_framework.openai import OpenAIChatClient
from decouple import config

# ---------------------- STREAMLIT CONFIG ----------------------
st.set_page_config(page_title="E-commerce Multi-Agent Demo", layout="centered")
st.title("🛍️ E-commerce Multi-Agent Assistant")
st.caption("Powered by Microsoft Agent Framework + OpenAI + Pinecone")

VERBOSE = st.checkbox("Enable Verbose Debug", value=False)

# ---------------------- CONFIG ----------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or config("OPENAI_API_KEY")
PC_API_KEY = config("PINECONE_API_KEY")
MODEL_ID = "gpt-4o-mini"
INDEX_NAME = "ecommerce-products"

# ---------------------- PINECONE SETUP ----------------------
pc = Pinecone(api_key=PC_API_KEY)

# create index if not exists
if INDEX_NAME not in [i.name for i in pc.list_indexes().indexes]:
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(INDEX_NAME)

# ---------------------- SAFE EMBEDDER INIT ----------------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

embedder = load_embedder()

# ---------------------- PARAMETER ADAPTER ----------------------
def adapt_parameters(tool_fn):
    """Adapter to fix Microsoft Agent Framework parameter passing issues"""
    def wrapper(*args, **kwargs):
        # Debug what we're receiving
        if VERBOSE:
            print(f"[Parameter Adapter] Raw args: {args}, kwargs: {kwargs}")
        
        # Handle the case where framework passes 'args' and 'kwargs' as keyword arguments
        if 'args' in kwargs and len(kwargs) == 1:
            # If we get a string in 'args', use it as the query parameter
            if isinstance(kwargs['args'], str):
                return tool_fn(query=kwargs['args'])
            # If we get a list/tuple in 'args', unpack it as positional args
            elif isinstance(kwargs['args'], (list, tuple)):
                return tool_fn(*kwargs['args'])
        
        # Handle case with both 'args' and 'kwargs' 
        elif 'args' in kwargs and 'kwargs' in kwargs:
            query_param = kwargs['args']
            if isinstance(query_param, str):
                # For vector_search_products, we need to handle query and optionally top_k
                if tool_fn.__name__ == 'vector_search_products':
                    # Try to extract top_k from the kwargs string if present
                    additional_kwargs = {}
                    if kwargs.get('kwargs'):
                        try:
                            additional_kwargs = eval(kwargs['kwargs'])
                        except:
                            pass
                    return tool_fn(query=query_param, **additional_kwargs)
                else:
                    return tool_fn(query_param)
        
        # If we get normal parameters, pass them through
        elif 'query' in kwargs or (len(args) > 0 and isinstance(args[0], str)):
            return tool_fn(*args, **kwargs)
        
        # Fallback: try to extract a query from any string in args or kwargs
        else:
            all_values = list(args) + list(kwargs.values())
            for val in all_values:
                if isinstance(val, str):
                    return tool_fn(query=val)
            
            # Last resort: call with empty query
            return tool_fn(query="beauty products")
    
    # Preserve the original function name and docstring
    wrapper.__name__ = tool_fn.__name__
    wrapper.__doc__ = tool_fn.__doc__
    return wrapper

# ---------------------- TOOL WRAPPERS ----------------------
def debug_tool_wrapper(tool_fn, agent_name=""):
    """Wrap a tool function to print debug info"""
    def wrapped(*args, **kwargs):
        if VERBOSE:
            print(f"[{agent_name} Tool] Running {tool_fn.__name__} with args={args}, kwargs={kwargs}")
        try:
            res = tool_fn(*args, **kwargs)
            if VERBOSE:
                print(f"[{agent_name} Tool] Success: {len(res) if isinstance(res, list) else 'N/A'} results")
            return res
        except Exception as e:
            if VERBOSE:
                print(f"[{agent_name} Tool] Error: {str(e)}")
            return f"Error: {str(e)}"
    return wrapped

# ---------------------- TOOLS ----------------------
def query_product_api_by_id(product_id: Annotated[int, Field(description="Product ID to fetch from dummyjson")]) -> dict:
    """Fetch product details by ID from dummyjson API"""
    url = f"https://dummyjson.com/products/{product_id}"
    res = requests.get(url)
    res.raise_for_status()
    return res.json()

def search_product_api_by_text(query: Annotated[str, Field(description="Text query for products from dummyjson")]) -> dict:
    """Search products by text query from dummyjson API"""
    url = f"https://dummyjson.com/products/search"
    res = requests.get(url, params={"q": query})
    res.raise_for_status()
    data = res.json()
    return data["products"][:3] if "products" in data else []

def vector_search_products(query: Annotated[str, Field(description="Semantic product search query")], 
                          top_k: Annotated[int, Field(description="Number of results to return", default=3)] = 3) -> List[dict]:
    """Search products using semantic similarity in vector database"""
    try:
        # Encode the query
        q_emb = embedder.encode(query).tolist()
        
        # Query Pinecone
        results = index.query(vector=q_emb, top_k=top_k, include_metadata=True)
        
        # Format results
        formatted_results = []
        for match in results["matches"]:
            result_item = {
                "id": match["id"],
                "score": match["score"],
                "metadata": match["metadata"]
            }
            formatted_results.append(result_item)
        
        return formatted_results
    except Exception as e:
        return f"Vector search error: {str(e)}"

# ---------------------- AGENT WRAPPER ----------------------
async def run_with_debug(agent, query, thread, agent_name=""):
    if VERBOSE:
        print(f"[{agent_name}] Running agent with query: {query}")
    try:
        result = await agent.run(query, thread=thread)
        if VERBOSE:
            print(f"[{agent_name}] Result: {getattr(result, 'text', str(result))}")
        return result
    except Exception as e:
        st.error(f"Agent execution error: {str(e)}")
        return f"Error: {str(e)}"

# ---------------------- AGENTS ----------------------
openai_client = OpenAIChatClient(api_key=OPENAI_API_KEY, model_id=MODEL_ID)

# Create agents with properly wrapped and adapted tools
api_agent = ChatAgent(
    chat_client=openai_client,
    name="ProductAPIAgent",
    instructions="You retrieve accurate structured data (price, stock, brand, etc.) using dummyjson API. Use query_product_api_by_id for specific product IDs and search_product_api_by_text for text-based searches.",
    tools=[
        debug_tool_wrapper(adapt_parameters(query_product_api_by_id), agent_name="ProductAPIAgent"),
        debug_tool_wrapper(adapt_parameters(search_product_api_by_text), agent_name="ProductAPIAgent")
    ]
)

vector_agent = ChatAgent(
    chat_client=openai_client,
    name="VectorSearchAgent",
    instructions="You handle semantic similarity and recommendation queries using the vector_search_products tool. Always provide the query parameter as a string and optionally top_k for number of results.",
    tools=[debug_tool_wrapper(adapt_parameters(vector_search_products), agent_name="VectorSearchAgent")]
)

router_agent = ChatAgent(
    chat_client=openai_client,
    name="RouterAgent",
    instructions=(
        "You are the main orchestrator. Analyze the user query and decide which agent to use:\n"
        "- Use ProductAPIAgent for factual lookups, price checks, stock availability, or when user mentions specific product IDs\n"
        "- Use VectorSearchAgent for semantic searches, recommendations, similarity queries, or when user asks for 'similar to' or 'like' something\n"
        "- For beauty items, skincare, makeup, etc., use VectorSearchAgent for better semantic matching\n"
        "Always explain which agent you're using and why."
    ),
    tools=[api_agent.as_tool(), vector_agent.as_tool()]
)

# ---------------------- STREAMLIT UI ----------------------
user_query = st.text_input("Enter your query (e.g., 'Recommend phones similar to iPhone' or 'Show me beauty products'):")

if "thread" not in st.session_state:
    st.session_state.thread = router_agent.get_new_thread()

if st.button("Run Query"):
    if user_query.strip():
        with st.spinner("🤖 Agent thinking..."):
            async def process_query():
                return await run_with_debug(router_agent, user_query, st.session_state.thread, agent_name="RouterAgent")
            
            response = asyncio.run(process_query())
        
        st.success("Agent Response:")
        st.write(response)

        if VERBOSE:
            st.info("✅ Check terminal for detailed verbose logs (agent/tool execution)")
    else:
        st.warning("Please enter a query before submitting.")

# ---------------------- DIRECT TESTING ----------------------
with st.expander("🔧 Test Tools Directly"):
    st.subheader("Test Vector Search Directly")
    test_query = st.text_input("Test query:", "beauty products")
    if st.button("Test Vector Search"):
        with st.spinner("Testing..."):
            results = vector_search_products(test_query, top_k=3)
            st.write("Direct test results:", results)
    
    st.subheader("Test API Search Directly")
    api_query = st.text_input("API test query:", "phone")
    if st.button("Test API Search"):
        with st.spinner("Testing API..."):
            results = search_product_api_by_text(api_query)
            st.write("API test results:", results)