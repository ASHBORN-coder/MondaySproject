"""
Monday.com BI Agent - SIMPLIFIED VERSION
Clean flow: Sample → Analyze → Fetch → Polish → Generate Code → Execute
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from src.schema_manager import SchemaManager
from src.llm_query_analyzer import LLMQueryAnalyzer
from src.llm_code_executor import LLMCodeExecutor

load_dotenv()

# ==========================================
# DATA CLEANING (SIMPLE)
# ==========================================
def clean_currency(value):
    """Clean currency values"""
    if pd.isna(value) or value == '':
        return 0
    if isinstance(value, (int, float)):
        return value
    cleaned = str(value).replace('₹', '').replace(',', '').replace(' ', '').strip()
    try:
        return float(cleaned)
    except:
        return 0

def remove_header_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Removes rows that are just repeats of the header names (agnostic to which columns exist)"""
    if df.empty:
        return df
    mask = pd.Series([True] * len(df), index=df.index)
    for col in df.columns:
        mask &= (df[col].astype(str).str.strip() != str(col))
    return df[mask]

def extract_number(value):
    """Extract numeric value from strings like '500 units' or '1,234'"""
    if pd.isna(value) or value == '':
        return 0
    if isinstance(value, (int, float)):
        return value
    import re
    match = re.search(r'[\d,]+\.?\d*', str(value))
    if match:
        return float(match.group().replace(',', ''))
    return 0

def polish_deals_data(df: pd.DataFrame) -> pd.DataFrame:
    """Unified data polishing for deals - ensures consistency with orders data"""
    if df.empty:
        return df
    
    df = df.copy()
    df = df.dropna(how='all')
    
    # 1. Agnostic Header Removal (works with any column combination)
    df = remove_header_rows(df)
    
    # 2. Financial Cleaning
    if 'Masked Deal value' in df.columns:
        df['Masked Deal value'] = df['Masked Deal value'].apply(clean_currency)
    
    # 3. Sector Normalization (CRITICAL: Must match orders table exactly)
    if 'Sector/service' in df.columns:
        df['Sector/service'] = df['Sector/service'].fillna('Uncategorized').astype(str).str.strip().str.title()
    
    # 4. Date Standardization
    for col in ['Close Date (A)', 'Tentative Close Date', 'Created Date']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    return df

def polish_orders_data(df: pd.DataFrame) -> pd.DataFrame:
    """Unified data polishing for orders - ensures consistency with deals data"""
    if df.empty:
        return df
    
    df = df.copy()
    df = df.dropna(how='all')
    
    # 1. Agnostic Header Removal (works with any column combination)
    df = remove_header_rows(df)
    
    # 2. Financial Cleaning (Excl GST is our Source of Truth)
    money_cols = [
        'Amount in Rupees (Excl of GST) (Masked)',
        'Amount in Rupees (Incl of GST) (Masked)',
        'Billed Value in Rupees (Excl of GST.) (Masked)',
        'Billed Value in Rupees (Incl of GST.) (Masked)',
        'Collected Amount in Rupees (Incl of GST.) (Masked)',
        'Amount Receivable (Masked)'
    ]
    for col in money_cols:
        if col in df.columns:
            df[col] = df[col].apply(clean_currency)
    
    # 3. Quantity Cleaning (Use regex extractor for complex formats)
    for col in ['Quantity by Ops', 'Quantities as per PO', 'Quantity billed (till date)', 'Balance in quantity']:
        if col in df.columns:
            df[col] = df[col].apply(extract_number)
    
    # 4. Sector Normalization (CRITICAL: Must match deals table exactly)
    if 'Sector' in df.columns:
        df['Sector'] = df['Sector'].fillna('Uncategorized').astype(str).str.strip().str.title()
    
    # 5. Date Standardization
    for col in ['Last invoice date', 'Data Delivery Date', 'Date of PO/LOI']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    return df

# ==========================================
# STREAMLIT UI
# ==========================================
st.set_page_config(page_title="Monday.com BI Agent", page_icon="📊", layout="wide")
st.title("📊 Monday.com BI Agent - Simple Edition")
st.markdown("**Simple Flow:** Sample Data → LLM Analysis → Targeted Fetch → Polish → Generate Code → Execute")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    api_token = st.text_input("Monday.com API Token", 
                              value=os.getenv("MONDAY_API_TOKEN", ""),
                              type="password")
    gemini_key = st.text_input("Google Gemini API Key", 
                               value=os.getenv("GOOGLE_API_KEY", ""),
                               type="password")
    deals_id = st.text_input("Deals Board ID", 
                             value=os.getenv("MONDAY_DEALS_BOARD_ID", ""))
    orders_id = st.text_input("Work Orders Board ID", 
                              value=os.getenv("MONDAY_ORDERS_BOARD_ID", ""))
    
    st.markdown("---")
    st.markdown("### 🎯 Simple Flow")
    st.markdown("""
    1. Load 5 sample rows
    2. LLM analyzes query
    3. Fetch needed columns
    4. Polish data
    5. Generate pandas code
    6. Execute & return
    """)

# Session State
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! Ask me anything about your Deals and Work Orders data."}]

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "schema_manager" not in st.session_state:
    st.session_state.schema_manager = SchemaManager()

if "llm_analyzer" not in st.session_state:
    st.session_state.llm_analyzer = None

if "deals_schema" not in st.session_state:
    st.session_state.deals_schema = None

if "orders_schema" not in st.session_state:
    st.session_state.orders_schema = None

if "schemas_loaded" not in st.session_state:
    st.session_state.schemas_loaded = False

# Load schemas (one-time)
if api_token and deals_id and orders_id and not st.session_state.schemas_loaded:
    with st.spinner("📥 Loading sample data (5 rows each)..."):
        schema_mgr = st.session_state.schema_manager
        
        deals_schema = schema_mgr.fetch_board_schema(api_token, deals_id, "Deals", sample_size=5)
        orders_schema = schema_mgr.fetch_board_schema(api_token, orders_id, "Work Orders", sample_size=5)
        
        if deals_schema.get('success') and orders_schema.get('success'):
            st.session_state.deals_schema = deals_schema
            st.session_state.orders_schema = orders_schema
            st.session_state.schemas_loaded = True
            
            with st.sidebar:
                st.success(f"✅ Loaded {len(deals_schema['columns'])} Deals columns, {len(orders_schema['columns'])} Orders columns")

# Render chat
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Main Query Loop
if prompt := st.chat_input(placeholder="e.g., What is our total revenue?"):
    
    if not api_token or not gemini_key:
        st.error("⚠️ Please provide API keys in the sidebar.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        
        # Initialize LLM analyzer
        if st.session_state.llm_analyzer is None and gemini_key:
            st.session_state.llm_analyzer = LLMQueryAnalyzer(gemini_key)
        
        # Step 1: LLM Analysis
        st.markdown("### 🔍 Analyzing Query")
        
        deals_sample = st.session_state.deals_schema.get('sample_data') if st.session_state.deals_schema else None
        orders_sample = st.session_state.orders_schema.get('sample_data') if st.session_state.orders_schema else None
        
        with st.status("🧠 LLM analyzing query with sample data...", expanded=False) as status:
            llm_analysis = st.session_state.llm_analyzer.analyze_query_with_context(
                user_query=prompt,
                chat_history=st.session_state.chat_history,
                deals_sample=deals_sample,
                orders_sample=orders_sample
            )
            status.update(label="✅ Analysis complete", state="complete", expanded=False)
        
        with st.expander("📋 LLM Analysis", expanded=False):
            st.write(f"**Intent:** {llm_analysis.get('intent', 'general')}")
            st.write(f"**Boards needed:** {llm_analysis.get('boards_needed', [])}")
            st.write(f"**Deals columns:** {llm_analysis.get('deals_columns', [])}")
            st.write(f"**Orders columns:** {llm_analysis.get('orders_columns', [])}")
        
        # Step 2: Fetch Targeted Data
        st.markdown("### 📡 Fetching Data")
        
        schema_mgr = st.session_state.schema_manager
        deals_cols = llm_analysis.get('deals_columns', [])
        orders_cols = llm_analysis.get('orders_columns', [])
        
        # Safety: fetch all if LLM returned empty
        if "Deals" in llm_analysis.get('boards_needed', []) and not deals_cols:
            deals_cols = st.session_state.deals_schema['columns']
        
        if "Work Orders" in llm_analysis.get('boards_needed', []) and not orders_cols:
            orders_cols = st.session_state.orders_schema['columns']
        
        with st.status("Fetching from Monday.com...", expanded=False) as fetch_status:
            # Fetch deals
            if deals_cols:
                raw_deals, _, _ = schema_mgr.fetch_targeted_data(api_token, deals_id, deals_cols, "Deals")
                st.write(f"✅ Fetched {len(raw_deals)} deals with {len(deals_cols)} columns")
            else:
                raw_deals = pd.DataFrame()
            
            # Fetch orders
            if orders_cols:
                raw_orders, _, _ = schema_mgr.fetch_targeted_data(api_token, orders_id, orders_cols, "Work Orders")
                st.write(f"✅ Fetched {len(raw_orders)} orders with {len(orders_cols)} columns")
            else:
                raw_orders = pd.DataFrame()
            
            fetch_status.update(label="✅ Data fetched", state="complete", expanded=False)
        
        # Step 3: Polish Data
        st.markdown("### 🧹 Polishing Data")
        
        clean_deals = polish_deals_data(raw_deals)
        clean_orders = polish_orders_data(raw_orders)
        
        st.write(f"✅ Cleaned {len(clean_deals)} deals, {len(clean_orders)} orders")
        
        # Step 4: Generate & Execute Code
        st.markdown("### 🤖 Generating Analysis Code")
        
        code_executor = LLMCodeExecutor(gemini_key)
        
        with st.status("🧠 Generating pandas code...", expanded=False) as code_status:
            analysis_result = code_executor.analyze_with_generated_code(
                query=prompt,
                deals_df=clean_deals,
                orders_df=clean_orders,
                classification={'intent': llm_analysis.get('intent', 'general')}
            )
            
            if analysis_result["success"]:
                code_status.update(label="✅ Code executed successfully", state="complete", expanded=False)
            else:
                code_status.update(label="⚠️ Execution had errors", state="error", expanded=True)
        
        # Show generated code
        with st.expander("🔧 Generated Code", expanded=False):
            st.code(analysis_result["generated_code"], language="python")
            
            if analysis_result["execution_result"]["output"]:
                st.markdown("**Output:**")
                st.text(analysis_result["execution_result"]["output"])
            
            if analysis_result["execution_result"]["error"]:
                st.error(analysis_result["execution_result"]["error"])
        
        # Step 5: Display Results
        st.markdown("---")
        st.markdown("### 💡 Results")
        
        final_answer = analysis_result["analysis_summary"]
        st.markdown(final_answer)
        
        # Save to chat
        st.session_state.messages.append({"role": "assistant", "content": final_answer})
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        st.session_state.chat_history.append({"role": "assistant", "content": final_answer[:500]})
        
        # Keep last 10 messages
        if len(st.session_state.chat_history) > 10:
            st.session_state.chat_history = st.session_state.chat_history[-10:]
