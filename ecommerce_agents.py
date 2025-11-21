import asyncio
import requests
import json
import re
import streamlit as st
from agent_framework.openai import OpenAIChatClient
from agent_framework import ChatMessage, TextContent, Role
from decouple import config
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

@st.cache_resource
def load_pinecone():
    pc = Pinecone(api_key=config("PINECONE_API_KEY"))
    return pc.Index("ecommerce-products")

index = load_pinecone()
@st.cache_resource
def load_st_model():
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    model.encode(["warmup"])  # Prevent meta tensor issues
    return model

model = load_st_model()
OPENAI_API_KEY = config("OPENAI_API_KEY")
model_id = "gpt-4o-mini"

router_agent = OpenAIChatClient(model_id=model_id).create_agent(
    instructions="""
    You are a Router Agent. Decide whether a user's query should go to:
    - 'API' if it is a direct request for specific product data.
    - 'Pinecone' if it requires semantic search.
    - 'Hybrid' if it requires both filtering and semantic understanding.
    ONLY respond with one of: API, Pinecone, Hybrid. Do not add any extra text.
    """,
    name="RouterAgent"
)

api_agent = OpenAIChatClient(model_id=model_id).create_agent(
    instructions="""
    You are an API Agent. User query is provided.
    Filter the provided products according to the query.
    Return a JSON list of matching product titles only.
    Example: ["Red Lipstick", "Powder Canister"]
    """,
    name="APIAgent"
)

pinecone_agent = OpenAIChatClient(model_id=model_id).create_agent(
    instructions="""
    You are a semantic product search agent.
    User query is provided.
    From the given products, return a JSON list of product titles that match semantically.
    Example: ["Red Lipstick", "Powder Canister"]
    """,
    name="PineconeAgent"
)

hybrid_agent = OpenAIChatClient(model_id=model_id).create_agent(
    instructions="""
    You are a product merging agent.
    User query is provided along with API and Pinecone results.
    Combine these lists into a single list of product titles, remove duplicates,
    keep the most relevant first, and return a JSON list only.
    Example: ["Red Lipstick", "Powder Canister"]
    """,
    name="HybridAgent"
)

def extract_titles_json(text: str):
    try:
        return json.loads(text)
    except Exception:
        titles = re.findall(r"'([^']+)'", text)
        titles += re.findall(r'"([^"]+)"', text)
        titles += re.findall(r"\d+\.\s*([^\n]+)", text)
        return [t.strip() for t in titles if t.strip()]

async def fetch_from_api(user_query: str) -> list:
    try:
        # Fetch all products
        response = requests.get("https://dummyjson.com/products")
        products = response.json().get("products", [])

        message = ChatMessage(
            role=Role.USER,
            contents=[TextContent(text=f"Query: {user_query}\nProducts: {products}")]
        )
        result = await api_agent.run(message)
        return extract_titles_json(result.text)

    except Exception as e:
        print(f"API Error: {e}")
        return []

async def search_pinecone(user_query: str) -> list:
    """
    Real Pinecone semantic search using the PineconeAgent for strict relevance filtering.
    - Pinecone retrieves top-k matches.
    - Agent strictly filters only truly relevant products.
    """
    try:
        # 1. Embed the user query
        query_emb = model.encode(user_query).tolist()

        # 2. Query Pinecone with top_k only (no filters)
        response = index.query(
            vector=query_emb,
            top_k=50,
            include_metadata=True
        )

        # 3. Prepare products for agent relevance filtering
        products = [
            {"title": item['metadata']['title'],
             "category": item['metadata'].get("category", ""),
             "brand": item['metadata'].get("brand", ""),
             "price": item['metadata'].get("price", 0),
             "rating": item['metadata'].get("rating", 0)}
            for item in response['matches']
        ]

        # 4. Use pinecone_agent to strictly filter for relevance
        relevance_message = ChatMessage(
            role=Role.USER,
            contents=[TextContent(text=f"""
            You are a strict product relevance agent.
            User query: "{user_query}"
            products: {products}

            Instructions:
            - Only include products that clearly match the user's query.
            - Ignore loosely related items (e.g., do not include nail polish when query is lipstick).
            - Only return the product titles in a JSON list.
            - Do not add any extra text.
            """)]
        )

        relevance_result = await pinecone_agent.run(relevance_message)
        try:
            filtered_titles = json.loads(relevance_result.text)
        except Exception:
            text = relevance_result.text
            filtered_titles = re.findall(r'"([^"]+)"', text)

        return filtered_titles

    except Exception as e:
        print(f"Pinecone Error: {e}")
        return []

async def hybrid_search(user_query: str) -> list:
    try:
        api_results = await fetch_from_api(user_query)
        pinecone_results = await search_pinecone(user_query)

        message = ChatMessage(
            role=Role.USER,
            contents=[TextContent(
                text=f"Query: {user_query}\nAPI Results: {api_results}\nPinecone Results: {pinecone_results}"
            )]
        )
        result = await hybrid_agent.run(message)
        return extract_titles_json(result.text)

    except Exception as e:
        print(f"Hybrid Error: {e}")
        return []

async def route_and_execute(user_query: str) -> list:
    message = ChatMessage(
        role=Role.USER,
        contents=[TextContent(text=user_query)]
    )

    result = await router_agent.run(message)
    route = result.text.strip()
    print(f"\nUser Query: {user_query}")
    print(f"Router Decision: {route}")

    if route == "API":
        return await fetch_from_api(user_query)
    elif route == "Pinecone":
        return await search_pinecone(user_query)
    elif route == "Hybrid":
        return await hybrid_search(user_query)
    else:
        return []

st.set_page_config(page_title="Microsoft E-Commerce Search", layout="wide")
st.title("🛍️ Microsoft E-Commerce Search Assistant")
st.markdown(
    "Enter a product query — the agents will decide whether to use the API or Pinecone or Hybrid, "
    "retrieve results, verify them, and display the best-matched items."
)

query = st.text_input("🔍 What are you looking for?", placeholder="e.g., iPhone, laptops, similar to Samsung, all products")

if st.button("Search"):
    if not query.strip():
        st.warning("Please enter a query before searching.")
    else:
        with st.spinner("🤖 Agents are collaborating..."):
            try:
                final_output = asyncio.run(route_and_execute(query))

                if isinstance(final_output, list) and len(final_output) > 0:
                    st.success(f"✅ Found {len(final_output)} products for query: '{query}'")
                    
                    # Fetch full product details from DummyJSON for display
                    response = requests.get("https://dummyjson.com/products")
                    all_products = response.json().get("products", [])

                    # Display each matched product
                    for title in final_output:
                        product_data = next((p for p in all_products if p["title"] == title), None)
                        if product_data:
                            with st.container():
                                cols = st.columns([1, 3])
                                with cols[0]:
                                    st.image(product_data.get("thumbnail", ""), use_container_width=True)
                                with cols[1]:
                                    st.subheader(product_data.get("title", "Unnamed Product"))
                                    st.markdown(f"**Brand:** {product_data.get('brand', 'Unknown')}")
                                    st.markdown(f"**Category:** {product_data.get('category', '-')}")
                                    st.markdown(f"**Price:** ${product_data.get('price', 0):.2f}")
                                    st.markdown(f"⭐ Rating: {product_data.get('rating', 0)} / 5")
                                st.divider()
                else:
                    st.warning("No products found.")
            except Exception as e:
                st.error(f"⚠️ Error during search: {e}")

st.markdown("---")
st.caption("Powered by Microsoft Agent Framework + Pinecone + DummyJSON API")