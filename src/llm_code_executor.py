"""
LLM Code Generator & Executor
Generates pandas code directly and executes it for reliable analysis
"""

import pandas as pd
import numpy as np
import json
from typing import Dict, List, Any, Optional, Tuple
from langchain_google_genai import ChatGoogleGenerativeAI
import io
import sys
import traceback
from datetime import datetime

class LLMCodeExecutor:
    """LLM that generates and executes pandas code directly"""
    
    def __init__(self, gemini_key: str):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0,
            max_output_tokens=8192  # Increased from 4096 to prevent code cutoff
        )
        self.llm.google_api_key = gemini_key
    
    def analyze_with_generated_code(
        self,
        query: str,
        deals_df: pd.DataFrame,
        orders_df: pd.DataFrame,
        classification: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate pandas code from LLM and execute it
        
        Returns:
            Dict with: code, execution_result, analysis_summary, error (if any)
        """
        
        # Step 1: Generate pandas code based on query
        pandas_code = self._generate_pandas_code(query, deals_df, orders_df, classification)
        
        # Step 2: Execute the generated code
        execution_result = self._execute_pandas_code(pandas_code, deals_df, orders_df)
        
        # Step 3: Generate human-readable summary
        summary = self._generate_analysis_summary(
            query, 
            execution_result, 
            classification
        )
        
        return {
            "generated_code": pandas_code,
            "execution_result": execution_result,
            "analysis_summary": summary,
            "success": execution_result.get("error") is None
        }
    
    def _generate_pandas_code(
        self,
        query: str,
        deals_df: pd.DataFrame,
        orders_df: pd.DataFrame,
        classification: Dict[str, Any]
    ) -> str:
        """Generate pandas code to answer the query"""
        
        # Build detailed column information
        deals_col_info = self._get_detailed_column_info(deals_df, "deals_df")
        orders_col_info = self._get_detailed_column_info(orders_df, "orders_df")
        
        # Build code generation prompt
        prompt = f"""
You are a pandas expert. Generate Python code to analyze business data.

## User Query
"{query}"

## Available DataFrames with COMPLETE Column Information

### deals_df ({len(deals_df)} rows)
{deals_col_info}

Sample data (first 3 rows):
{deals_df.head(3).to_string() if not deals_df.empty else "Empty DataFrame"}

### orders_df ({len(orders_df)} rows)
{orders_col_info}

Sample data (first 3 rows):
{orders_df.head(3).to_string() if not orders_df.empty else "Empty DataFrame"}

## Query Classification
- Intent: {classification.get('intent', 'general')}
- Boards needed: {classification.get('boards_to_query', [])}

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

## CRITICAL INSTRUCTIONS

1. **ALWAYS check if columns exist before using them**
2. **ALWAYS define variables before using them**
3. **ALWAYS handle empty DataFrames**
4. **Use try-except for safety**
5. **Print clear results**

## Code Requirements

- DataFrames are ALREADY CLEANED (currency values are already numeric)
- deals_df and orders_df are ready to use
- Check if DataFrame is empty before calculations
- Check if columns exist before accessing them
- Initialize all variables with default values first
- Use .get() method for safe dictionary access

## CODE GENERATION GUIDELINES

1. **Column Usage**: Use the EXACT column names from the schema above (spacing and capitalization matter)
2. **Safety First**: Always check if columns exist before accessing them
3. **Initialize Variables**: Define all variables with default values before calculations
4. **Handle Empty Data**: Check if DataFrames are empty before operations
5. **Error Handling**: Use try-except blocks for calculations that might fail
6. **Clear Output**: Print results with proper formatting and currency symbols (₹)

## CRITICAL BUSINESS LOGIC RULES

### Status Filtering:
- For REVENUE queries: Consider 'Deal Status' column
  - Include: 'Won' deals (confirmed revenue)
  - Consider: 'Open' deals (potential revenue) - clarify in output
  - Exclude: 'Lost', 'Cancelled' deals
- Always specify which statuses are included in your calculation

### Date Range Filtering:
- If query mentions time period (2025, Q1, last month, etc.):
  - Use 'Close Date (A)' for deals
  - Use 'Last invoice date' or 'Created Date' for orders
  - Filter data to match the requested period
- If no time period mentioned: Use ALL data but mention "all-time" in output

### Duplicate Prevention:
- Check if same deal appears in both deals_df and orders_df
- If calculating total revenue, avoid double counting:
  - Option 1: Use deals_df for potential revenue, orders_df for actual collected
  - Option 2: Use only one source and clarify which
- Print warning if duplicates detected

## Code Structure Pattern:
```python
# Step 1: Initialize all variables with defaults
result_variable = 0

# Step 2: Check data availability and column existence
if not df.empty and 'column_name' in df.columns:
    # Step 3: Perform calculation with error handling
    try:
        result_variable = df['column_name'].sum()
        print(f"Result: {{result_variable}}")
    except Exception as e:
        print(f"Error: {{e}}")
else:
    print("Data or column not available")
```

## CRITICAL OUTPUT REQUIREMENTS:
1. **OUTPUT ONLY EXECUTABLE PYTHON CODE**
2. **NO explanations, NO text, NO markdown outside code blocks**
3. **Start directly with Python code or ```python**
4. **Do NOT write "To analyze..." or "We will..." - ONLY CODE**
5. **If you include ANY non-code text, the execution will FAIL**

## CODE QUALITY REQUIREMENTS:
1. **NO empty try blocks** - Every try must have code inside it
2. **NO incomplete statements** - Every if/for/while/try must have a body
3. **Proper indentation** - Use 4 spaces consistently
4. **Complete code only** - No partial or commented-out logic
5. **Test your logic** - Ensure code is syntactically valid

## EXAMPLE OF INVALID CODE (DO NOT GENERATE):
```python
try:
    # Comment but no code
except:
    pass
```

## EXAMPLE OF VALID CODE:
```python
try:
    result = df['column'].sum()
    print(f"Result: {{result}}")
except Exception as e:
    print(f"Error: {{e}}")
```

Generate ONLY complete, executable pandas code (no explanations):
"""
        
        response = self.llm.invoke(prompt)
        code = response.content
        
        # Clean up the code (remove markdown formatting and any text before code)
        if "```python" in code:
            # Extract code from python markdown block
            code = code.split("```python")[1].split("```")[0]
        elif "```" in code:
            # Extract code from generic markdown block
            code = code.split("```")[1].split("```")[0]
        else:
            # No markdown blocks - check if there's explanatory text before code
            # Look for common Python keywords to find where code starts
            lines = code.split('\n')
            code_start_idx = 0
            for i, line in enumerate(lines):
                stripped = line.strip()
                # If line starts with Python code indicators, that's where code begins
                if stripped and (
                    stripped.startswith('import ') or
                    stripped.startswith('from ') or
                    stripped.startswith('#') or
                    stripped.startswith('def ') or
                    any(stripped.startswith(kw) for kw in ['if ', 'for ', 'while ', 'try:', 'with '])
                    or '=' in stripped
                ):
                    code_start_idx = i
                    break
            
            # If we found code start, take everything from there
            if code_start_idx > 0:
                code = '\n'.join(lines[code_start_idx:])
        
        return code.strip()
    
    def _execute_pandas_code(
        self,
        code: str,
        deals_df: pd.DataFrame,
        orders_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Execute the generated pandas code safely with validation checks"""
        
        # Safety Check 1: Empty Data Validation
        warnings = []
        if deals_df.empty:
            warnings.append("⚠️ Deals data is empty")
        if orders_df.empty:
            warnings.append("⚠️ Orders data is empty")
        
        # Safety Check 2: Duplicate Detection
        if not deals_df.empty and not orders_df.empty:
            if 'Deal Name' in deals_df.columns and 'Deal name masked' in orders_df.columns:
                deals_names = set(deals_df['Deal Name'].dropna())
                orders_names = set(orders_df['Deal name masked'].dropna())
                duplicates = deals_names & orders_names
                if duplicates:
                    warnings.append(f"⚠️ {len(duplicates)} deals appear in both boards - potential double counting")
        
        # Prepare execution environment
        local_vars = {
            'deals_df': deals_df.copy(),
            'orders_df': orders_df.copy(),
            'pd': pd,
            'np': np,
            'datetime': datetime,
            '_warnings': warnings  # Pass warnings to code
        }
        
        # Safety Check 3: Validate code syntax before execution
        try:
            compile(code, '<string>', 'exec')
        except SyntaxError as e:
            return {
                "output": "",
                "variables": {},
                "error": f"Code Syntax Error: {str(e)}\n\nThe generated code has syntax errors. This usually means the LLM generated malformed code. Please try rephrasing your query.",
                "warnings": warnings
            }
        
        # Capture output
        old_stdout = sys.stdout
        sys.stdout = captured_output = io.StringIO()
        
        try:
            # Execute the code - pass local_vars as both globals and locals
            # This allows the code to access deals_df, orders_df, etc.
            exec(code, local_vars, local_vars)
            
            # Get captured output
            output = captured_output.getvalue()
            
            # Add warnings to output
            if warnings:
                warning_text = "\n".join(warnings)
                output = f"{warning_text}\n\n{output}" if output else warning_text
            
            # Extract computed variables
            result_vars = {}
            for var_name, var_value in local_vars.items():
                if not var_name.startswith('_') and var_name not in ['deals_df', 'orders_df', 'pd', 'np', 'datetime']:
                    try:
                        # Try to serialize the variable
                        result_vars[var_name] = var_value
                    except:
                        pass
            
            return {
                "output": output,
                "variables": result_vars,
                "error": None
            }
            
        except Exception as e:
            error_msg = f"Execution Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            return {
                "output": captured_output.getvalue(),
                "variables": {},
                "error": error_msg
            }
        
        finally:
            # Restore stdout
            sys.stdout = old_stdout
    
    def _get_detailed_column_info(self, df: pd.DataFrame, name: str) -> str:
        """Get detailed column information with types and sample values"""
        if df.empty:
            return f"**{name}**: Empty DataFrame - No columns available"
        
        col_details = []
        col_details.append(f"**Available Columns ({len(df.columns)} total):**")
        
        for col in df.columns:
            dtype = df[col].dtype
            non_null = df[col].notna().sum()
            total = len(df)
            
            # Get sample values (first 3 non-null unique values)
            sample_vals = df[col].dropna().unique()[:3].tolist()
            sample_str = ", ".join([str(v)[:50] for v in sample_vals]) if len(sample_vals) > 0 else "No data"
            
            col_details.append(
                f"  • '{col}' (type: {dtype}, {non_null}/{total} non-null)\n"
                f"    Sample values: {sample_str}"
            )
        
        return "\n".join(col_details)
    
    def _get_dataframe_info(self, df: pd.DataFrame, name: str) -> str:
        """Get summary info about a dataframe"""
        if df.empty:
            return f"{name}: Empty DataFrame"
        
        info = f"""
{name}: {len(df)} rows, {len(df.columns)} columns
Columns: {df.columns.tolist()}
Sample:
{df.head(2).to_string()}
        """
        return info.strip()
    
    def _serialize_variables(self, variables: Dict[str, Any]) -> str:
        """Safely serialize variables to JSON, handling Timestamp and other non-serializable types"""
        import pandas as pd
        from datetime import datetime, date
        
        def convert_value(val):
            """Convert non-serializable types to strings"""
            if isinstance(val, (pd.Timestamp, datetime, date)):
                return str(val)
            elif isinstance(val, (pd.Series, pd.DataFrame)):
                return str(val)
            elif isinstance(val, (np.integer, np.floating)):
                return float(val)
            elif isinstance(val, np.ndarray):
                return val.tolist()
            elif isinstance(val, dict):
                return {k: convert_value(v) for k, v in val.items()}
            elif isinstance(val, (list, tuple)):
                return [convert_value(v) for v in val]
            else:
                return val
        
        try:
            # Convert all values to serializable types
            serializable_vars = {k: convert_value(v) for k, v in variables.items()}
            return json.dumps(serializable_vars, indent=2)
        except Exception as e:
            # If still fails, convert everything to string
            return json.dumps({k: str(v) for k, v in variables.items()}, indent=2)
    
    def _generate_analysis_summary(
        self,
        query: str,
        execution_result: Dict[str, Any],
        classification: Dict[str, Any]
    ) -> str:
        """Generate a human-readable summary of the analysis"""
        
        if execution_result.get("error"):
            return f"❌ Error during analysis: {execution_result['error']}"
        
        output = execution_result.get("output", "")
        variables = execution_result.get("variables", {})
        
        # Build summary prompt
        summary_prompt = f"""
You are a Business Intelligence analyst presenting insights to a founder.

## Original Query
"{query}"

## Analysis Results
{output}

## Computed Variables
{self._serialize_variables(variables)}

## Instructions
Create a founder-level summary that includes:

1. **Executive Summary** (2-3 sentences)
2. **Key Findings** (bullet points with specific numbers)
3. **Business Insights** (what this means)
4. **Recommendations** (actionable next steps)

Format with markdown, use ₹ for currency, be concise but comprehensive.

## Response Format:
```markdown
### 📊 Executive Summary
[2-3 sentence summary]

### 💰 Key Findings
- [Finding 1 with specific number]
- [Finding 2 with percentage]
- [Finding 3 with comparison]

### 💡 Business Insights
- [Insight 1]
- [Insight 2]
- [Insight 3]

### 🎯 Recommendations
- [Action 1]
- [Action 2]
```

Generate the summary now:
"""
        
        response = self.llm.invoke(summary_prompt)
        summary = response.content
        
        # Clean up markdown if present
        if "```markdown" in summary:
            summary = summary.split("```markdown")[1].split("```")[0]
        elif "```" in summary:
            summary = summary.split("```")[1].split("```")[0]
        
        return summary.strip()
    
    def format_for_display(self, result: Dict[str, Any]) -> str:
        """Format the complete result for Streamlit display"""
        
        if not result["success"]:
            return f"""
### ❌ Analysis Failed

**Error:**
```
{result['execution_result']['error']}
```

**Generated Code:**
```python
{result['generated_code']}
```
"""
        
        return f"""
{result['analysis_summary']}

---

### 🔧 Generated Code
```python
{result['generated_code']}
```

### 📊 Execution Output
```
{result['execution_result']['output']}
```
"""
