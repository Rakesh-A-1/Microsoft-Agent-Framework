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

reformulator_agent = traced_agent("ReformulatorAgent", """
    You are a Search Query Refiner.
    Your goal is to rewrite the latest user input into a STANDALONE search query.
    
    Input:
    - Conversation History
    - Latest User Query
    
    Rules:
    1. Resolve references (e.g., "Cheaper ones" -> "Cheaper iPhones").
    2. If the user changes topic, output the new topic only.
    3. Output ONLY the raw query string. No JSON.
""")

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

response_agent = traced_agent("ResponseAgent", """
You are a Response Composer Agent.

Input:
- User query
- A verified list of products (JSON)

Your job:
- Write a natural, friendly response like ChatGPT
- Briefly introduce the result
- DO NOT modify product data
- DO NOT add or remove products
- DO NOT mention internal agents or logic
- End with a polite, non-repetitive follow-up question

Output rules:
- Plain text only
- No JSON
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
    history_entries = st.session_state.messages[-5:] 
    history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history_entries])

    with tracer.start_as_current_span("query_reformulation"):
        reform_message = ChatMessage(
            role=Role.USER, 
            contents=[TextContent(text=f"""
                Conversation History:
                {history_text}
                
                Latest User Query: "{user_query}"
                
                Output the standalone search query:
            """)]
        )
        
        reform_result = await reformulator_agent.run(reform_message)
        refined_query = reform_result.text.strip()
        
        print(f"Original: {user_query} - Refined: {refined_query}")

    search_counter.add(1, {"query": refined_query})

    with tracer.start_as_current_span("route_and_execute"):
        message = ChatMessage(
            role=Role.USER,
            contents=[TextContent(text=refined_query)]
        )

        result = await router_agent.run(message, thread=thread)

        try:
            route_json = json.loads(result.text.strip())
            route = route_json.get("source", "")
        except:
            route = result.text.strip()

        print(f"Router Decision: {route}")

        if route == "API":
            search_results = await fetch_from_api(refined_query, thread)
        elif route == "Pinecone":
            search_results = await search_pinecone(refined_query, thread)
        elif route == "Hybrid":
            search_results = await hybrid_search(refined_query, thread)
        else:
            search_results = []

        return search_results

st.set_page_config(
    page_title="E-commerce AI Agent",
    page_icon="🛍️",
    layout="wide"
)

if "thread" not in st.session_state:
    st.session_state.thread = None

st.title("🛍️ E-commerce AI Agent")
st.markdown("""
This agent can help you find products using:
- **Structured Data** (DummyJSON API)
- **Semantic Search** (Pinecone)
- **Hybrid Search**
""")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    st.info("Ensure your `.env` file is set up correctly.")
    if st.button("Reload Agent"):
        st.cache_resource.clear()
        st.success("Agent reloaded!")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render history (DO NOT ERASE)
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
prompt = st.chat_input("Ask about a product...")

if prompt:
    # Store user message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                results = asyncio.run(
                    route_and_execute(prompt, st.session_state.thread)
                )

                if results:
                    convo_message = ChatMessage(
                        role=Role.USER,
                        contents=[TextContent(text=f"""
    User query: {prompt}

    Products (use each product ONCE, no repetition):
    {json.dumps(results, indent=2)}

    Rules:
    - Mention each product only once
    - Include title, brand, category, price, rating
    - Do NOT repeat product names
    - Do NOT say "Found X products"
    - End with a friendly follow-up
    """)]
                    )

                    convo_result = asyncio.run(
                        response_agent.run(convo_message, st.session_state.thread)
                    )

                    # Show ONLY conversational response
                    st.markdown(convo_result.text)

                    # Save EXACT same text to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": convo_result.text
                    })

                else:
                    st.markdown("No products found.")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "No products found."
                    })

            except Exception as e:
                st.error(str(e))
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Error: {e}"
                })