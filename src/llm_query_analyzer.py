"""
LLM Query Analyzer Module
Uses LLM to intelligently analyze user queries with full context (sample data + chat history)
and determine exactly what columns to fetch from both boards.
"""

import pandas as pd
import json
from typing import Dict, List, Any, Optional
from langchain_google_genai import ChatGoogleGenerativeAI


class LLMQueryAnalyzer:
    """
    Hybrid intelligent query analyzer that uses LLM to understand user intent
    with full context including sample data and conversation history.
    """
    
    def __init__(self, gemini_api_key: str):
        """Initialize LLM for query analysis"""
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            temperature=0,
            max_output_tokens=2048,
            api_key=gemini_api_key
        )
    
    def analyze_query_with_context(
        self,
        user_query: str,
        chat_history: List[Dict[str, str]],
        deals_sample: Optional[pd.DataFrame] = None,
        orders_sample: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Analyze user query with full context to determine required data fetching strategy.
        
        Args:
            user_query: Current user question
            chat_history: List of previous messages [{"role": "user/assistant", "content": "..."}]
            deals_sample: Sample DataFrame from Deals board (5 rows)
            orders_sample: Sample DataFrame from Work Orders board (5 rows)
        
        Returns:
            {
                "intent": "revenue_analysis",
                "boards_needed": ["Deals", "Work Orders"],
                "deals_columns": ["Item Name", "Masked Deal value"],
                "orders_columns": ["Item Name", "Collected Amount"],
                "reasoning": "Need both potential and actual revenue",
                "context_aware_notes": "Continuing from previous revenue discussion"
            }
        """
        
        # Build context-rich prompt
        prompt = self._build_analysis_prompt(
            user_query, 
            chat_history, 
            deals_sample, 
            orders_sample
        )
        
        # Get LLM analysis
        try:
            response = self.llm.invoke(prompt)
            analysis = self._parse_llm_response(response.content)
            return analysis
        except Exception as e:
            # Fallback to conservative approach
            return self._fallback_analysis(user_query, deals_sample, orders_sample)
    
    def _get_schema_definition(self) -> str:
        """Get the exact same schema definition used in code executor"""
        return """
## DATA SCHEMA (Column Names and Definitions)

### DEALS DataFrame (deals_df) - Schema:
- 'Deal Name': Unique identifier for each deal (string)
- 'Owner code': Sales person/owner assigned to the deal (string)
- 'Client Code': Customer/company identifier (string)
- 'Deal Status': Current state of the deal (string values: Open, Won, Lost, etc.)
- 'Close Date (A)': Actual closing date of the deal (datetime)
- 'Closure Probability': Likelihood of deal closure (string: High, Medium, Low)
- 'Masked Deal value': Total monetary value of the deal in rupees (numeric)
- 'Tentative Close Date': Expected closing date (datetime)
- 'Deal Stage': Current stage in sales pipeline (string)
- 'Product deal': Product or service being sold (string)
- 'Sector/service': Industry sector or service category (string)
- 'Created Date': When the deal was created (datetime)

### ORDERS DataFrame (orders_df) - Schema:
- 'Deal name masked': Reference to the parent deal (string)
- 'Customer Name Code': Customer identifier (string)
- 'Serial #': Work order serial number (string)
- 'Nature of Work': Type of work being performed (string)
- 'Last executed month of recurring project': For recurring projects (datetime)
- 'Execution Status': Current execution state (string: Completed, In Progress, etc.)
- 'Data Delivery Date': When data was delivered (datetime)
- 'Date of PO/LOI': Purchase order or letter of intent date (datetime)
- 'Document Type': Type of contract document (string)
- 'Probable Start Date': Expected start date (datetime)
- 'Probable End Date': Expected end date (datetime)
- 'BD/KAM Personnel code': Business development/key account manager (string)
- 'Sector': Industry sector (string)
- 'Type of Work': Category of work (string)
- 'Last invoice date': Most recent invoice date (datetime)
- 'latest invoice no.': Most recent invoice number (string)
- 'Amount in Rupees (Excl of GST) (Masked)': Order amount excluding GST (numeric)
- 'Amount in Rupees (Incl of GST) (Masked)': Order amount including GST (numeric)
- 'Billed Value in Rupees (Excl of GST.) (Masked)': Amount already billed excluding GST (numeric)
- 'Billed Value in Rupees (Incl of GST.) (Masked)': Amount already billed including GST (numeric)
- 'Collected Amount in Rupees (Incl of GST.) (Masked)': Amount collected including GST (numeric)
- 'Amount to be billed in Rs. (Exl. of GST) (Masked)': Pending billing amount excluding GST (numeric)
- 'Amount to be billed in Rs. (Incl. of GST) (Masked)': Pending billing amount including GST (numeric)
- 'Amount Receivable (Masked)': Outstanding amount to be received (numeric)
- 'AR Priority account': Accounts receivable priority flag (string)
- 'Quantity by Ops': Quantity recorded by operations (numeric)
- 'Quantities as per PO': Quantity as per purchase order (numeric)
- 'Quantity billed (till date)': Cumulative billed quantity (numeric)
- 'Balance in quantity': Remaining quantity (numeric)
- 'Invoice Status': Current invoice state (string)
- 'Expected Billing Month': When billing is expected (datetime)
- 'Actual Billing Month': When billing actually occurred (datetime)
- 'Actual Collection Month': When payment was collected (datetime)
- 'WO Status (billed)': Work order billing status (string)
- 'Collection status': Payment collection state (string)
- 'Collection Date': Date of payment collection (datetime)
- 'Billing Status': Current billing state (string)
"""
    
    def _build_analysis_prompt(
        self,
        user_query: str,
        chat_history: List[Dict[str, str]],
        deals_sample: Optional[pd.DataFrame],
        orders_sample: Optional[pd.DataFrame]
    ) -> str:
        """Build comprehensive prompt for LLM analysis"""
        
        prompt_parts = []
        
        # System instruction
        prompt_parts.append("""You are an expert data analyst helping determine what data to fetch from Monday.com boards.
Your job is to analyze the user's query and determine EXACTLY which columns are needed from which boards.

**Your Task:**
1. Understand user intent from query + chat history
2. Identify which boards are needed (Deals, Work Orders, or both)
3. Select MINIMAL required columns from each board
4. Provide reasoning for your decisions

**Important Guidelines:**
- Be PRECISE with column names (use exact names from sample data)
- Fetch ONLY what's needed (minimize data transfer)
- Consider chat context (user might be continuing previous discussion)
- If query mentions "revenue/money/total", you likely need BOTH boards
- If query mentions "deals/pipeline/sales", focus on Deals board
- If query mentions "orders/delivery/execution", focus on Work Orders board
""")
        
        # Chat history context
        if chat_history:
            prompt_parts.append("\n**Recent Conversation History:**")
            # Last 5 messages for context
            recent_history = chat_history[-5:] if len(chat_history) > 5 else chat_history
            for msg in recent_history:
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')[:200]  # Truncate long messages
                prompt_parts.append(f"{role.upper()}: {content}")
        
        # Add schema definition (same as code executor)
        prompt_parts.append(self._get_schema_definition())
        
        # Add sample data
        if deals_sample is not None and not deals_sample.empty:
            prompt_parts.append("\n**Deals Sample Data (first 3 rows):**")
            prompt_parts.append(deals_sample.head(3).to_string(index=False))
        
        if orders_sample is not None and not orders_sample.empty:
            prompt_parts.append("\n**Work Orders Sample Data (first 3 rows):**")
            prompt_parts.append(orders_sample.head(3).to_string(index=False))
        
        # Current user query
        prompt_parts.append(f"\n**Current User Query:**\n{user_query}")
        
        # Output format instruction
        prompt_parts.append("""
**Output Format (JSON only, no explanation):**
{
    "intent": "revenue_analysis|deals_pipeline|operations|sector_analysis|general",
    "boards_needed": ["Deals", "Work Orders"],
    "deals_columns": ["exact column names from Deals board"],
    "orders_columns": ["exact column names from Work Orders board"],
    "reasoning": "Brief explanation of why these columns are needed",
    "context_aware_notes": "How this relates to previous conversation (if applicable)"
}

**Rules:**
- Use EXACT column names from sample data
- Include "Item Name" if you need to identify specific items
- For revenue queries, include value/amount columns from both boards
- For sector queries, include sector columns + relevant metrics
- For status queries, include status + related columns
- Keep column lists MINIMAL (only what's truly needed)
""")
        
        return "\n".join(prompt_parts)
    
    def _parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        """Parse LLM JSON response"""
        try:
            # Extract JSON from response (handle markdown code blocks)
            response_text = response_text.strip()
            if response_text.startswith("```json"):
                response_text = response_text.split("```json")[1].split("```")[0]
            elif response_text.startswith("```"):
                response_text = response_text.split("```")[1].split("```")[0]
            
            analysis = json.loads(response_text)
            
            # Validate required fields
            required_fields = ["intent", "boards_needed", "deals_columns", "orders_columns", "reasoning"]
            for field in required_fields:
                if field not in analysis:
                    raise ValueError(f"Missing required field: {field}")
            
            # Ensure lists
            if not isinstance(analysis["boards_needed"], list):
                analysis["boards_needed"] = [analysis["boards_needed"]]
            if not isinstance(analysis["deals_columns"], list):
                analysis["deals_columns"] = []
            if not isinstance(analysis["orders_columns"], list):
                analysis["orders_columns"] = []
            
            return analysis
            
        except Exception as e:
            # If parsing fails, return conservative fallback
            return {
                "intent": "general",
                "boards_needed": ["Deals", "Work Orders"],
                "deals_columns": [],
                "orders_columns": [],
                "reasoning": f"LLM response parsing failed: {str(e)}. Using fallback.",
                "context_aware_notes": "",
                "parse_error": True
            }
    
    def _fallback_analysis(
        self,
        user_query: str,
        deals_sample: Optional[pd.DataFrame],
        orders_sample: Optional[pd.DataFrame]
    ) -> Dict[str, Any]:
        """Conservative fallback when LLM analysis fails"""
        
        query_lower = user_query.lower()
        
        # Determine boards needed
        boards_needed = []
        deals_cols = []
        orders_cols = []
        
        # Revenue queries need both boards
        if any(kw in query_lower for kw in ['revenue', 'total', 'money', 'income', 'collected', 'billed']):
            boards_needed = ["Deals", "Work Orders"]
            if deals_sample is not None:
                deals_cols = list(deals_sample.columns)
            if orders_sample is not None:
                orders_cols = list(orders_sample.columns)
        else:
            # Default: fetch both boards with all columns
            boards_needed = ["Deals", "Work Orders"]
            if deals_sample is not None:
                deals_cols = list(deals_sample.columns)
            if orders_sample is not None:
                orders_cols = list(orders_sample.columns)
        
        return {
            "intent": "general",
            "boards_needed": boards_needed,
            "deals_columns": deals_cols,
            "orders_columns": orders_cols,
            "reasoning": "Fallback: LLM analysis unavailable, fetching all columns from both boards",
            "context_aware_notes": "",
            "is_fallback": True
        }
    
    def format_analysis_summary(self, analysis: Dict[str, Any]) -> str:
        """Format analysis results for display"""
        
        summary_parts = []
        
        summary_parts.append(f"**Intent**: {analysis.get('intent', 'unknown')}")
        summary_parts.append(f"**Boards**: {', '.join(analysis.get('boards_needed', []))}")
        
        deals_cols = analysis.get('deals_columns', [])
        if deals_cols:
            summary_parts.append(f"**Deals Columns** ({len(deals_cols)}): {', '.join(deals_cols[:5])}")
            if len(deals_cols) > 5:
                summary_parts.append(f"  ... and {len(deals_cols) - 5} more")
        
        orders_cols = analysis.get('orders_columns', [])
        if orders_cols:
            summary_parts.append(f"**Orders Columns** ({len(orders_cols)}): {', '.join(orders_cols[:5])}")
            if len(orders_cols) > 5:
                summary_parts.append(f"  ... and {len(orders_cols) - 5} more")
        
        summary_parts.append(f"**Reasoning**: {analysis.get('reasoning', 'N/A')}")
        
        context_notes = analysis.get('context_aware_notes', '')
        if context_notes:
            summary_parts.append(f"**Context**: {context_notes}")
        
        return "\n".join(summary_parts)
