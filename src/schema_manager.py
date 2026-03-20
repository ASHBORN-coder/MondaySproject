"""
Schema Manager: Fetches and caches board schemas (headers + sample data)
Enables query-driven, targeted data fetching instead of full board loads
"""

import requests
import pandas as pd
from typing import Dict, List, Tuple, Optional
import time

class SchemaManager:
    """Manages board schemas and enables targeted data fetching"""
    
    def __init__(self):
        self.schemas = {}  # Cache: {board_id: schema_info}
    
    def fetch_board_schema(self, api_token: str, board_id: str, board_name: str = "Board", 
                          sample_size: int = 5) -> Dict:
        """
        Fetch board schema: column names, types, and sample data
        
        Args:
            api_token: Monday.com API token
            board_id: Board ID to fetch schema from
            board_name: Human-readable board name
            sample_size: Number of sample rows to fetch (default: 5)
        
        Returns:
            Dict with schema info: {
                'board_id': str,
                'board_name': str,
                'columns': List[str],
                'sample_data': pd.DataFrame,
                'column_types': Dict[str, str],
                'fetched_at': float
            }
        """
        
        query = f"""
        query {{
            boards(ids: {board_id}) {{
                name
                columns {{
                    id
                    title
                    type
                }}
                items_page(limit: {sample_size}) {{
                    items {{
                        id
                        name
                        column_values {{
                            id
                            text
                            value
                        }}
                    }}
                }}
            }}
        }}
        """
        
        try:
            response = requests.post(
                "https://api.monday.com/v2",
                json={'query': query},
                headers={
                    "Authorization": api_token,
                    "Content-Type": "application/json"
                },
                timeout=30
            )
            
            if response.status_code != 200:
                return {
                    'success': False,
                    'error': f"API returned status {response.status_code}"
                }
            
            data = response.json()
            
            if 'errors' in data:
                return {
                    'success': False,
                    'error': str(data['errors'])
                }
            
            board_data = data['data']['boards'][0]
            
            # Extract column information
            columns_info = board_data['columns']
            column_names = [col['title'] for col in columns_info]
            column_types = {col['title']: col['type'] for col in columns_info}
            column_id_to_title = {col['id']: col['title'] for col in columns_info}
            
            # Extract sample data
            items = board_data['items_page']['items']
            rows = []
            
            for item in items:
                row = {'Item Name': item['name']}
                for col_val in item['column_values']:
                    # Find column title from id
                    col_title = next((c['title'] for c in columns_info if c['id'] == col_val['id']), None)
                    if col_title:
                        row[col_title] = col_val['text']
                rows.append(row)
            
            sample_df = pd.DataFrame(rows)
            
            schema_info = {
                'success': True,
                'board_id': board_id,
                'board_name': board_name,
                'columns': column_names,
                'sample_data': sample_df,
                'column_types': column_types,
                'column_id_to_title': column_id_to_title,
                'fetched_at': time.time(),
                'sample_size': len(rows)
            }
            
            # Cache the schema
            self.schemas[board_id] = schema_info
            
            return schema_info
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_cached_schema(self, board_id: str) -> Optional[Dict]:
        """Get cached schema if available"""
        return self.schemas.get(board_id)
    
    def identify_required_columns(self, query: str, schema_info: Dict) -> List[str]:
        """
        Analyze query to determine which columns are needed
        
        Args:
            query: User's natural language query
            schema_info: Schema information from fetch_board_schema
        
        Returns:
            List of column names required for the query
        """
        query_lower = query.lower()
        available_columns = schema_info['columns']
        required_columns = []
        
        # Always include item name
        if 'Item Name' in available_columns:
            required_columns.append('Item Name')
        
        # Revenue/Money related
        if any(word in query_lower for word in ['revenue', 'money', 'amount', 'value', 'collected', 'billed', 'price', 'total']):
            money_cols = [col for col in available_columns if any(
                keyword in col.lower() for keyword in ['amount', 'value', 'price', 'revenue', 'collected', 'billed', 'deal value']
            )]
            required_columns.extend(money_cols)
        
        # Sector/Category related
        if any(word in query_lower for word in ['sector', 'category', 'industry', 'service']):
            sector_cols = [col for col in available_columns if any(
                keyword in col.lower() for keyword in ['sector', 'service', 'category', 'industry']
            )]
            required_columns.extend(sector_cols)
        
        # Status related - CRITICAL for deal queries
        if any(word in query_lower for word in ['status', 'stage', 'open', 'closed', 'won', 'lost', 'deal', 'pipeline', 'how many']):
            status_cols = [col for col in available_columns if any(
                keyword in col.lower() for keyword in ['status', 'stage', 'state', 'deal status']
            )]
            required_columns.extend(status_cols)
        
        # Date related
        if any(word in query_lower for word in ['date', 'when', 'time', 'quarter', 'month', 'year', 'period']):
            date_cols = [col for col in available_columns if any(
                keyword in col.lower() for keyword in ['date', 'time', 'created', 'updated', 'close']
            )]
            required_columns.extend(date_cols)
        
        # Quantity related
        if any(word in query_lower for word in ['quantity', 'count', 'number', 'how many']):
            qty_cols = [col for col in available_columns if any(
                keyword in col.lower() for keyword in ['quantity', 'count', 'number', 'qty']
            )]
            required_columns.extend(qty_cols)
        
        # Owner/Client related
        if any(word in query_lower for word in ['owner', 'client', 'who', 'person']):
            owner_cols = [col for col in available_columns if any(
                keyword in col.lower() for keyword in ['owner', 'client', 'person', 'code']
            )]
            required_columns.extend(owner_cols)
        
        # Remove duplicates while preserving order
        required_columns = list(dict.fromkeys(required_columns))
        
        # IMPORTANT: If very few columns identified, be conservative and fetch more
        # This prevents missing critical columns
        if len(required_columns) <= 2:  # Only Item Name + 1 other column or less
            # Return all columns to be safe
            return available_columns
        
        # If we have 3-5 columns, add common analysis columns as safety net
        if len(required_columns) <= 5:
            # Add status columns if not already included (critical for most queries)
            for col in available_columns:
                if 'status' in col.lower() and col not in required_columns:
                    required_columns.append(col)
            # Add date columns for time-based analysis
            for col in available_columns:
                if 'date' in col.lower() and col not in required_columns:
                    required_columns.append(col)
        
        return required_columns
    
    def fetch_targeted_data(self, api_token: str, board_id: str, 
                           required_columns: List[str], 
                           board_name: str = "Board",
                           limit: int = 500) -> Tuple[pd.DataFrame, float, bool]:
        """
        Fetch only the required columns from the board
        
        Args:
            api_token: Monday.com API token
            board_id: Board ID
            required_columns: List of column names to fetch
            board_name: Human-readable board name
            limit: Max items to fetch
        
        Returns:
            Tuple of (dataframe, response_time, success)
        """
        
        start_time = time.time()
        
        # First, get column IDs mapping
        schema = self.get_cached_schema(board_id)
        if not schema:
            # Need to fetch schema first
            schema = self.fetch_board_schema(api_token, board_id, board_name)
            if not schema['success']:
                return pd.DataFrame(), time.time() - start_time, False
        
        query = f"""
        query {{
            boards(ids: {board_id}) {{
                items_page(limit: {limit}) {{
                    items {{
                        id
                        name
                        column_values {{
                            id
                            text
                            value
                        }}
                    }}
                }}
            }}
        }}
        """
        
        try:
            response = requests.post(
                "https://api.monday.com/v2",
                json={'query': query},
                headers={
                    "Authorization": api_token,
                    "Content-Type": "application/json"
                },
                timeout=30
            )
            
            response_time = time.time() - start_time
            
            if response.status_code != 200:
                return pd.DataFrame(), response_time, False
            
            data = response.json()
            
            if 'errors' in data:
                return pd.DataFrame(), response_time, False
            
            items = data['data']['boards'][0]['items_page']['items']
            
            # Get column ID to title mapping from schema
            column_id_to_title = schema.get('column_id_to_title', {})
            
            rows = []
            for item in items:
                row = {'Item Name': item['name']}
                
                # Map column IDs to titles
                for col_val in item['column_values']:
                    col_id = col_val['id']
                    col_title = column_id_to_title.get(col_id)
                    
                    # Only include if this column is in required_columns
                    if col_title and col_title in required_columns:
                        row[col_title] = col_val['text']
                
                rows.append(row)
            
            df = pd.DataFrame(rows)
            
            return df, response_time, True
            
        except Exception as e:
            return pd.DataFrame(), time.time() - start_time, False
    
    def get_schema_summary(self, board_id: str) -> str:
        """Get human-readable schema summary"""
        schema = self.get_cached_schema(board_id)
        if not schema or not schema.get('success'):
            return "Schema not available"
        
        summary = f"**{schema['board_name']}** ({schema['sample_size']} sample rows)\n"
        summary += f"**Columns ({len(schema['columns'])})**: {', '.join(schema['columns'][:10])}"
        if len(schema['columns']) > 10:
            summary += f"... and {len(schema['columns']) - 10} more"
        
        return summary
