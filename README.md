# RetailMate AI 🛍️

**Track 1 Submission — Build and Deploy a Customer-Facing AI Agent**

RetailMate AI is a customer-facing AI agent for a fictional online store, "Urban Threads."
Unlike a plain FAQ chatbot, it's a real **agent**: it uses tool/function calling to take
actions — checking live order status, checking product stock, recommending products, and
creating support tickets — instead of just generating text.

## Features (Tools the agent can call)
- `check_order_status(order_id)` — looks up real order status from the store DB
- `check_stock(product_name)` — checks price & stock availability
- `recommend_products(category, max_price)` — recommends best-rated in-stock items
- `create_support_ticket(issue_summary, order_id)` — files a support ticket for issues

## Tech Stack
- **Frontend/App:** Streamlit
- **LLM:** Groq (LLaMA 3.1 8B Instant) with native tool calling
- **Data:** Mock in-memory "database" (simulates a real retail backend/API)

## Run Locally
```bash
pip install -r requirements.txt
export GROQ_API_KEY="your_key_here"
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Push this folder to a GitHub repo.
2. Go to https://share.streamlit.io → "New app" → select the repo, branch, and `app.py`.
3. In **Settings → Secrets**, add:
   ```
   GROQ_API_KEY = "your_key_here"
   ```
4. Deploy. You'll get a public URL like `https://your-app.streamlit.app`.

## Sample Data (for testing)
- Order IDs: `UT1001`, `UT1002`, `UT1003`, `UT1004`
- Products: Denim Jacket, Running Shoes, Cotton Hoodie, Formal Shirt, Chino Trousers, Leather Wallet
