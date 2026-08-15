"""
RetailMate AI — Customer-Facing AI Agent
Track 1: Build and Deploy a Customer-Facing AI Agent
Built with Streamlit + Groq (LLaMA 3.1) with real tool/function calling.
"""

import streamlit as st
from groq import Groq
import json
import os
from datetime import datetime, timedelta

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="RetailMate AI",
    page_icon="🛍️",
    layout="centered",
)

# ---------------------------------------------------------
# MOCK "BACKEND" DATA (simulates a real store's DB/API)
# ---------------------------------------------------------
ORDERS_DB = {
    "UT1001": {"item": "Denim Jacket - Blue, M", "status": "Shipped", "eta": "2 days"},
    "UT1002": {"item": "Running Shoes - White, 8", "status": "Processing", "eta": "4 days"},
    "UT1003": {"item": "Cotton Hoodie - Black, L", "status": "Delivered", "eta": "Delivered on 12 Aug"},
    "UT1004": {"item": "Formal Shirt - Grey, S", "status": "Out for Delivery", "eta": "Today"},
}

PRODUCTS_DB = [
    {"name": "Denim Jacket", "category": "Outerwear", "price": 2499, "stock": 14, "rating": 4.5},
    {"name": "Running Shoes", "category": "Footwear", "price": 3199, "stock": 0, "rating": 4.3},
    {"name": "Cotton Hoodie", "category": "Casual Wear", "price": 1799, "stock": 22, "rating": 4.6},
    {"name": "Formal Shirt", "category": "Formal Wear", "price": 1299, "stock": 8, "rating": 4.2},
    {"name": "Chino Trousers", "category": "Casual Wear", "price": 1899, "stock": 30, "rating": 4.4},
    {"name": "Leather Wallet", "category": "Accessories", "price": 899, "stock": 5, "rating": 4.7},
]

SUPPORT_TICKETS = []

# ---------------------------------------------------------
# TOOL FUNCTIONS (the "agent" part — real actions, not just chat)
# ---------------------------------------------------------
def check_order_status(order_id: str) -> str:
    order = ORDERS_DB.get(order_id.upper().strip())
    if not order:
        return json.dumps({"error": f"No order found with ID {order_id}. Please check the ID and try again."})
    return json.dumps(order)


def check_stock(product_name: str) -> str:
    matches = [p for p in PRODUCTS_DB if product_name.lower() in p["name"].lower()]
    if not matches:
        return json.dumps({"error": f"No product found matching '{product_name}'."})
    return json.dumps(matches)


def recommend_products(category: str = "", max_price: int = 999999) -> str:
    results = [
        p for p in PRODUCTS_DB
        if (category.lower() in p["category"].lower() if category else True)
        and p["price"] <= max_price
        and p["stock"] > 0
    ]
    results = sorted(results, key=lambda x: -x["rating"])[:3]
    if not results:
        return json.dumps({"error": "No matching products found in stock."})
    return json.dumps(results)


def create_support_ticket(issue_summary: str, order_id: str = "N/A") -> str:
    ticket_id = f"TCK{1000 + len(SUPPORT_TICKETS) + 1}"
    ticket = {
        "ticket_id": ticket_id,
        "order_id": order_id,
        "issue": issue_summary,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "status": "Open — a human agent will follow up within 24 hours",
    }
    SUPPORT_TICKETS.append(ticket)
    return json.dumps(ticket)


TOOL_FUNCTIONS = {
    "check_order_status": check_order_status,
    "check_stock": check_stock,
    "recommend_products": recommend_products,
    "create_support_ticket": create_support_ticket,
}

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "check_order_status",
            "description": "Check the delivery status of a customer's order using their order ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "The order ID, e.g. UT1001"}
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_stock",
            "description": "Check stock availability and price for a specific product by name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_name": {"type": "string", "description": "Name of the product, e.g. Denim Jacket"}
                },
                "required": ["product_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recommend_products",
            "description": "Recommend in-stock products, optionally filtered by category and max price.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "description": "Category filter, e.g. Casual Wear. Optional."},
                    "max_price": {"type": "integer", "description": "Maximum price in INR. Optional."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_support_ticket",
            "description": "Create a support ticket when the customer has an issue that needs human follow-up (e.g. damaged item, wrong item, refund request).",
            "parameters": {
                "type": "object",
                "properties": {
                    "issue_summary": {"type": "string", "description": "Short summary of the customer's issue"},
                    "order_id": {"type": "string", "description": "Related order ID, if any"},
                },
                "required": ["issue_summary"],
            },
        },
    },
]

SYSTEM_PROMPT = """You are RetailMate, the friendly AI shopping and support agent for "Urban Threads", 
an online clothing and accessories store. You help customers with:
- Checking order status (ask for order ID if not given, format like UT1001)
- Checking product stock and prices
- Recommending products based on their needs
- Creating support tickets for issues like damaged/wrong items or refunds

Always use the available tools to fetch real data instead of guessing. Be concise, warm, and helpful.
If a customer's request is outside these areas, politely say you can only help with Urban Threads shopping and support queries.
"""

# ---------------------------------------------------------
# GROQ CLIENT
# ---------------------------------------------------------
def get_client():
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        st.error("⚠️ GROQ_API_KEY not found. Add it in Render → Environment.")
        st.stop()
    return Groq(api_key=api_key)

def run_agent(client, messages):
    """Send messages to Groq with tool access, execute tool calls, and return final reply."""
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        tools=TOOLS_SCHEMA,
        tool_choice="auto",
        max_tokens=800,
    )
    msg = response.choices[0].message

    if msg.tool_calls:
        messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": msg.tool_calls})
        for tool_call in msg.tool_calls:
            fn_name = tool_call.function.name
            try:
                args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                args = {}
            fn = TOOL_FUNCTIONS.get(fn_name)
            result = fn(**args) if fn else json.dumps({"error": "Unknown tool"})
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": fn_name,
                "content": result,
            })
        # Get final natural-language reply after tool results
        follow_up = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            max_tokens=800,
        )
        final_msg = follow_up.choices[0].message.content
        messages.append({"role": "assistant", "content": final_msg})
        return final_msg
    else:
        messages.append({"role": "assistant", "content": msg.content})
        return msg.content


# ---------------------------------------------------------
# UI
# ---------------------------------------------------------
st.title("🛍️ RetailMate AI")
st.caption("Your AI shopping & support agent for **Urban Threads** — powered by Groq LLaMA 3.1")

with st.sidebar:
    st.header("Try quick actions")
    st.markdown("**Sample order IDs:** UT1001, UT1002, UT1003, UT1004")
    st.markdown("**Sample products:** Denim Jacket, Running Shoes, Cotton Hoodie, Formal Shirt, Chino Trousers, Leather Wallet")
    if st.button("📦 Check order UT1002"):
        st.session_state.setdefault("pending_prompt", "What's the status of order UT1002?")
    if st.button("🧥 Recommend casual wear under ₹2000"):
        st.session_state.setdefault("pending_prompt", "Recommend some casual wear under 2000 rupees")
    if st.button("🚫 Report a damaged item"):
        st.session_state.setdefault("pending_prompt", "My order UT1003 arrived damaged, please help")
    st.divider()
    st.caption("Built for Track 1: Build and Deploy a Customer-Facing AI Agent")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

# Render chat history (skip system message)
for m in st.session_state.messages:
    if m["role"] in ("user", "assistant") and m.get("content"):
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

prompt = st.chat_input("Ask about orders, products, or report an issue...")
if "pending_prompt" in st.session_state:
    prompt = st.session_state.pop("pending_prompt")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            client = get_client()
            reply = run_agent(client, st.session_state.messages)
            st.markdown(reply)
