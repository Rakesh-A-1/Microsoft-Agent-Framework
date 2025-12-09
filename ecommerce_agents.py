import asyncio
import requests
import json
import re
import streamlit as st
from agent_framework.openai import OpenAIChatClient
from agent_framework import ChatMessage, TextContent, Role
from agent_framework.observability import setup_observability, get_meter, get_tracer
from decouple import config
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from conversation_storage import save_thread, restore_last_thread

setup_observability()

# Custom metrics and spans
meter = get_meter()
tracer = get_tracer()
search_counter = meter.create_counter(
    name="search_requests_total",
    description="Total number of product search requests",
    unit="1"
)

@st.cache_resource
def load_pinecone():
    pc = Pinecone(api_key=config("PINECONE_API_KEY"))
    return pc.Index("ecommerce-products")

index = load_pinecone()
@st.cache_resource
def load_st_model():
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    model.encode(["warmup"])
    return model

model = load_st_model()
OPENAI_API_KEY = config("OPENAI_API_KEY")
model_id = "gpt-4o-mini"

def traced_agent(agent_name, instructions):
    """Wrap agent.run with a tracer span automatically."""
    agent = OpenAIChatClient(model_id=model_id).create_agent(
        instructions=instructions,
        name=agent_name,
        temperature=0,
        seed=42
    )

    original_run = agent.run

    async def traced_run(message: ChatMessage, thread=None):
        with tracer.start_as_current_span(f"{agent_name}_run"):
            if thread is not None:
                return await original_run(message, thread=thread)
            return await original_run(message)

    agent.run = traced_run
    return agent

router_agent = traced_agent("RouterAgent", """
    Respond with: {"source": "API" | "Pinecone" | "Hybrid", "reason": "brief explanation"}
    Rules:
    - Use API when:
    - Query is generic full list.
    - Exact match for product, brand, or known category (e.g., beauty, electronics).
    - Contains numeric/logical filters (price, rating, stock).
    - Pinecone for semantic meaning queries only (e.g., "luxury skincare", "long lasting phones").
    - Hybrid only when BOTH semantic meaning AND numeric/logical filters exist.
    Output ONLY the JSON.
""")

api_agent = traced_agent("APIAgent", """
    You are an API Agent. User query is provided.
    Filter the provided products according to the query.
    Only keep products that EXACTLY match the user's requested product type and filters. 
    No loose matches, no semantic guessing, no "maybe relevant" items.  
    If it's not an exact match → delete it.
    Output must be ONLY valid JSON (a list of product objects).
    Do not return titles only. Do not return text or explanations.
    Each returned product must include all original fields, including thumbnail, brand, category, rating, price.
    Only include products that clearly match the user's query; remove anything that does not directly match.
    Example output:
    [
      {"title": "iPhone 9", "brand": "Apple", "price": 549},
      {"title": "iPhone X", "brand": "Apple", "price": 899}
    ]
""")

pinecone_agent = traced_agent("PineconeAgent", """
    You are a semantic product search agent.
    User query is provided.
    From the given products, return a JSON list of product that match semantically.
    Do not return titles only. Do not return text or explanations.
    Each returned product must include all original fields, including thumbnail, brand, category, rating, price.
    Only include products that clearly match the user's query; remove anything that does not directly match.
    Example output:
    [
      {"title": "iPhone 9", "brand": "Apple", "price": 549},
      {"title": "iPhone X", "brand": "Apple", "price": 899}
    ]
""")

hybrid_agent = traced_agent("HybridAgent", """
    You are a product merging agent.
    User query is provided along with API and Pinecone results.
    Combine these lists into a single list of product, remove duplicates,
    keep the most relevant first, and return a JSON list only.
    Do not return titles only. Do not return text or explanations.
    Each returned product must include all original fields, including thumbnail, brand, category, rating, price.
    Only include products that clearly match the user's query; remove anything that does not directly match.
    Example output:
    [
      {"title": "iPhone 9", "brand": "Apple", "price": 549},
      {"title": "iPhone X", "brand": "Apple", "price": 899}
    ]
""")

def extract_titles_json(text: str):
    try:
        return json.loads(text)
    except Exception:
        titles = re.findall(r"'([^']+)'", text)
        titles += re.findall(r'"([^"]+)"', text)
        titles += re.findall(r"\d+\.\s*([^\n]+)", text)
        return [t.strip() for t in titles if t.strip()]

async def fetch_from_api(user_query: str, thread) -> list:
    try:
        # Fetch all products
        response = requests.get("https://dummyjson.com/products")
        products = response.json().get("products", [])

        message = ChatMessage(
            role=Role.USER,
            contents=[TextContent(text=f"Query: {user_query}\nProducts: {products}")]
        )
        result = await api_agent.run(message, thread)
        return json.loads(result.text)
    except Exception as e:
        print(f"API Error: {e}")
        return []

async def search_pinecone(user_query: str, thread) -> list:
    """
    Real Pinecone semantic search using the PineconeAgent for strict relevance filtering.
    - Pinecone retrieves top-k matches.
    - Agent strictly filters only truly relevant products.
    - Returns full product objects.
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
             "rating": item['metadata'].get("rating", 0),
             "thumbnail": item['metadata'].get("thumbnail", "")}
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
            - Return FULL product objects in a JSON list.
            - Do not return titles only or any extra text.
            """)]
        )

        relevance_result = await pinecone_agent.run(relevance_message, thread)
        # Safely parse JSON
        try:
            return json.loads(relevance_result.text)
        except Exception:
            # fallback: wrap titles as objects
            titles = re.findall(r'"([^"]+)"', relevance_result.text)
            return [{"title": t} for t in titles]

    except Exception as e:
        print(f"Pinecone Error: {e}")
        return []

async def hybrid_search(user_query: str, thread) -> list:
    try:
        api_results = await fetch_from_api(user_query, thread)
        pinecone_results = await search_pinecone(user_query, thread)

        message = ChatMessage(
            role=Role.USER,
            contents=[TextContent(
                text=f"Query: {user_query}\nAPI Results: {api_results}\nPinecone Results: {pinecone_results}"
            )]
        )
        result = await hybrid_agent.run(message, thread)
        return extract_titles_json(result.text)
    except Exception as e:
        print(f"Hybrid Error: {e}")
        return []

async def route_and_execute(user_query: str, thread):
    search_counter.add(1, {"query": user_query})

    with tracer.start_as_current_span("route_and_execute"):
        message = ChatMessage(
            role=Role.USER,
            contents=[TextContent(text=user_query)]
        )

        # Pass the thread here — this ensures persistence
        result = await router_agent.run(message, thread=thread)

        try:
            route_json = json.loads(result.text.strip())
            route = route_json.get("source", "")
        except:
            route = result.text.strip()

        print(f"\nUser Query: {user_query}")
        print(f"Router Decision: {route}")

        if route == "API":
            search_results = await fetch_from_api(user_query, thread)
        elif route == "Pinecone":
            search_results = await search_pinecone(user_query, thread)
        elif route == "Hybrid":
            search_results = await hybrid_search(user_query, thread)
        else:
            search_results = []

        return search_results

st.set_page_config(page_title="Microsoft E-Commerce Search", layout="wide")
st.title("🛍️ Microsoft E-Commerce Search Assistant")
st.markdown(
    "Enter a product query — the agents will decide whether to use the API, Pinecone, or Hybrid, "
    "retrieve results, verify them, and display the best-matched items."
)

query = st.text_input("🔍 What are you looking for?", placeholder="e.g., iPhone, laptops, similar to Samsung, all products")
async def run_search(query):
    # Restore or create a new thread
    thread = await restore_last_thread(router_agent)

    # Run agent pipeline
    final_output = await route_and_execute(query, thread)

    # Save updated conversation state
    await save_thread(thread)

    return final_output
if st.button("Search"):
    if not query.strip():
        st.warning("Please enter a query before searching.")
    else:
        with st.spinner("🤖 Agents are collaborating..."):
            try:
                final_output = asyncio.run(run_search(query))
                if isinstance(final_output, list) and len(final_output) > 0:
                    st.success(f"✅ Found {len(final_output)} products for query: '{query}'")
                    for p in final_output:
                        with st.container():
                            cols = st.columns([1, 3])
                            with cols[0]:
                                st.image(p.get("thumbnail", ""), width='stretch')
                            with cols[1]:
                                st.subheader(p.get("title", "Unnamed Product"))
                                st.markdown(f"**Brand:** {p.get('brand', 'Unknown')}")
                                st.markdown(f"**Category:** {p.get('category', '-')}")
                                st.markdown(f"**Price:** ${p.get('price', 0)}")
                                st.markdown(f"⭐ Rating: {p.get('rating', 0)} / 5")
                            st.divider()
                else:
                    st.warning("No products found.")
            except Exception as e:
                st.error(f"⚠️ Error during search: {e}")

st.markdown("---")
st.caption("Powered by Microsoft Agent Framework + Pinecone + DummyJSON API")