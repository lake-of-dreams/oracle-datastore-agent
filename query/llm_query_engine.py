"""
LLM-Powered Query Engine for Oracle Database 23ai
Uses local LLM models via Ollama to generate and optimize SQL queries
"""
import json
import logging
import requests
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
import re
from common.setup import setup_logging
from database import db_manager


setup_logging()
logger = logging.getLogger(__name__)


@dataclass
class LLMQueryResult:
    """Enhanced query result with LLM insights"""
    original_query: str
    generated_sql: str
    llm_confidence: float
    query_explanation: str
    optimization_suggestions: List[str]
    alternative_queries: List[str]
    execution_time: float


class LLMQueryEngine:
    """LLM-powered query engine using Ollama for intelligent query generation"""
    
    def __init__(self, ollama_host: str = "http://localhost:11434", model_name: str = "llama2"):
        self.ollama_host = ollama_host
        self.model_name = model_name
        self.available_models = []
        self._initialize_ollama()
    
    def _initialize_ollama(self):
        """Initialize connection to Ollama"""
        logger.info("Initialize connection to Ollama")
        try:
            # Check if Ollama is running
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                self.available_models = [model['name'] for model in models]
                logger.info(f"Ollama initialized with models: {self.available_models}")
                
                # Use the first available model if specified model not found
                if self.model_name not in self.available_models and self.available_models:
                    self.model_name = self.available_models[0]
                    logger.info(f"Using model: {self.model_name}")
            else:
                logger.warning("Ollama not available, falling back to base query engine")
                self.model_name = None
                
        except Exception as e:
            logger.warning(f"Failed to initialize Ollama: {e}")
            self.model_name = None
    
    def generate_sql_from_natural_language(self, natural_query: str, 
                                          table_schemas: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate SQL query from natural language using LLM
        
        Args:
            natural_query: Natural language query
            table_schemas: Available table schemas for context
            
        Returns:
            Dictionary with generated SQL and metadata
        """
        logger.info("Generate SQL query from natural language using LLM")
        if not self.model_name:
            return self._fallback_sql_generation(natural_query, table_schemas)
        
        try:
            # Create context for the LLM
            context = self._create_sql_context(natural_query, table_schemas)
            
            # Generate SQL using Ollama
            prompt = self._create_sql_prompt(natural_query, context)
            response = self._call_ollama(prompt)
            logger.info(f"LLM Response: {response}")
            
            # Parse LLM response
            return self._parse_sql_response(response, natural_query)
            
        except Exception as e:
            logger.error(f"Error generating SQL with LLM: {e}")
            return self._fallback_sql_generation(natural_query, table_schemas)
    
    def optimize_query(self, sql_query: str, table_schemas: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Optimize SQL query using LLM analysis
        
        Args:
            sql_query: SQL query to optimize
            table_schemas: Table schemas for context
            
        Returns:
            Dictionary with optimized query and suggestions
        """
        logger.info("Optimize SQL query using LLM analysis")
        if not self.model_name:
            return {"optimized_query": sql_query, "suggestions": []}
        
        try:
            prompt = self._create_optimization_prompt(sql_query, table_schemas)
            response = self._call_ollama(prompt)
            logger.info(f"LLM Response: {response}")
            
            return self._parse_optimization_response(response, sql_query)
            
        except Exception as e:
            logger.error(f"Error optimizing query with LLM: {e}")
            return {"optimized_query": sql_query, "suggestions": []}
    
    def explain_query(self, sql_query: str) -> str:
        """
        Generate human-readable explanation of SQL query
        
        Args:
            sql_query: SQL query to explain
            
        Returns:
            Explanation string
        """
        logger.info(" Generate human-readable explanation of SQL query")
        if not self.model_name:
            return f"SQL Query: {sql_query}"
        
        try:
            prompt = f"""
Explain this SQL query in simple terms:

{sql_query}

Provide a clear, concise explanation of what this query does.
"""
            response = self._call_ollama(prompt)
            logger.info(f"LLM Response: {response}")
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error explaining query with LLM: {e}")
            return f"SQL Query: {sql_query}"
    
    def suggest_alternative_queries(self, natural_query: str, 
                                  table_schemas: List[Dict[str, Any]] = None) -> List[str]:
        """
        Suggest alternative ways to express the same query
        
        Args:
            natural_query: Original natural language query
            table_schemas: Available table schemas
            
        Returns:
            List of alternative query suggestions
        """
        logger.info(" Suggest alternative ways to express the same query")
        if not self.model_name:
            return [natural_query]
        
        try:
            prompt = f"""
Given this query: "{natural_query}"

Suggest 3 alternative ways to express the same request in natural language.
Each alternative should be clear and specific.

Return as a JSON array of strings.
"""
            response = self._call_ollama(prompt)
            logger.info(f"LLM Response: {response}")
            
            # Try to parse as JSON
            try:
                alternatives = json.loads(response)
                if isinstance(alternatives, list):
                    return alternatives[:3]  # Limit to 3 suggestions
            except:
                pass
            
            # Fallback: split by lines
            lines = [line.strip() for line in response.split('\n') if line.strip()]
            return lines[:3]
            
        except Exception as e:
            logger.error(f"Error suggesting alternatives with LLM: {e}")
            return [natural_query]
    
    def create_intelligent_query(self, query: str, table_name: Optional[str] = None) -> LLMQueryResult:
        """
        Create query with LLM-powered intelligence
        
        Args:
            query: Query string (SQL or natural language)
            table_name: Optional table name for context
            
        Returns:
            LLMQueryResult with enhanced metadata
        """
        logger.info("Create query with LLM-powered intelligence")
        start_time = datetime.now()
        
        try:
            # Determine if query is SQL or natural language
            if self._is_sql_query(query):
                generated_sql = query
                llm_confidence = 1.0
                query_explanation = self.explain_query(query)
                optimization_suggestions = []
                alternative_queries = []
            else:
                # Generate SQL from natural language
                table_schemas = self._get_table_schemas(table_name)
                sql_generation = self.generate_sql_from_natural_language(query, table_schemas)
                
                generated_sql = sql_generation.get('sql', query)
                llm_confidence = sql_generation.get('confidence', 0.5)
                query_explanation = sql_generation.get('explanation', '')
                
                # Optimize the generated query
                optimization = self.optimize_query(generated_sql, table_schemas)
                optimized_sql = optimization.get('optimized_query', generated_sql)
                optimization_suggestions = optimization.get('suggestions', [])
                
                # Generate alternative queries
                alternative_queries = self.suggest_alternative_queries(query, table_schemas)
                generated_sql = optimized_sql
            
            execution_time = (datetime.now() - start_time).total_seconds()
            return LLMQueryResult(
                execution_time=execution_time,
                original_query=query,
                generated_sql=generated_sql,
                llm_confidence=llm_confidence,
                query_explanation=query_explanation,
                optimization_suggestions=optimization_suggestions,
                alternative_queries=alternative_queries
            )
            
        except Exception as e:
            logger.error(f"Error executing intelligent query: {e}")
            
            return LLMQueryResult(
                original_query=query,
                generated_sql=query,
                llm_confidence=0.0,
                query_explanation=f"Fallback execution: {str(e)}",
                optimization_suggestions=[],
                alternative_queries=[]
            )
    
    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API with the given prompt"""
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "max_tokens": 1000
                }
            }
            
            response = requests.post(
                f"{self.ollama_host}/api/generate",
                json=payload,
                timeout=300
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            else:
                raise Exception(f"Ollama API error: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error calling Ollama: {e}")
            raise
    
    def _create_sql_context(self, natural_query: str, table_schemas: List[Dict[str, Any]]) -> str:
        """Create context string for SQL generation"""
        context = "Available tables and their schemas:\n\n"
        
        if table_schemas:
            for schema in table_schemas:
                context += f"Table: {schema.get('table_name', 'unknown')}\n"
                context += f"Columns: {', '.join(schema.get('columns', []))}\n"
                context += f"Type: {schema.get('data_type', 'unknown')}\n\n"
        else:
            context += "No specific table schemas provided.\n"
        
        context += "Oracle Database 23ai features available:\n"
        context += "- JSON Relational Duality\n"
        context += "- AI Vector Search\n"
        context += "- Boolean data type\n"
        context += "- Wide Tables (up to 4096 columns)\n"
        context += "- Value LOBs\n"
        
        return context
    
    def _create_sql_prompt(self, natural_query: str, context: str) -> str:
        """Create prompt for SQL generation"""
        return f"""
You are an expert SQL developer specializing in Oracle Database 23ai.

{context}

User Query: "{natural_query}"

Generate an Oracle SQL query that fulfills this request. Consider:
1. Use appropriate Oracle 23ai features when relevant
2. Use proper Oracle SQL syntax
3. Include appropriate WHERE clauses, JOINs, and ORDER BY as needed
4. Use JSON functions for JSON data
5. Use vector functions for similarity search
6. Optimize for performance

Return your response as JSON with this structure:
{{
    "sql": "SELECT ... FROM ...",
    "confidence": 0.95,
    "explanation": "This query does...",
    "oracle_features_used": ["JSON_VALUE", "VECTOR_DISTANCE"]
}}
"""
    
    def _create_optimization_prompt(self, sql_query: str, table_schemas: List[Dict[str, Any]]) -> str:
        """Create prompt for query optimization"""
        return f"""
You are an Oracle Database performance expert.

Analyze this SQL query for optimization opportunities:

{sql_query}

Consider:
1. Index usage
2. Join optimization
3. Oracle 23ai specific optimizations
4. Query structure improvements
5. Performance best practices

Return your response as JSON:
{{
    "optimized_query": "SELECT ... FROM ...",
    "suggestions": ["Use index on column X", "Consider JSON_TABLE for complex JSON queries"],
    "performance_notes": "This optimization improves performance by..."
}}
"""
    
    def _parse_sql_response(self, response: str, original_query: str) -> Dict[str, Any]:
        """Parse LLM response for SQL generation"""
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return {
                    'sql': data.get('sql', original_query),
                    'confidence': float(data.get('confidence', 0.5)),
                    'explanation': data.get('explanation', ''),
                    'oracle_features': data.get('oracle_features_used', [])
                }
        except:
            pass
        
        # Fallback: treat entire response as SQL
        return {
            'sql': response.strip(),
            'confidence': 0.3,
            'explanation': 'Generated SQL query',
            'oracle_features': []
        }
    
    def _parse_optimization_response(self, response: str, original_query: str) -> Dict[str, Any]:
        """Parse LLM response for query optimization"""
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return {
                    'optimized_query': data.get('optimized_query', original_query),
                    'suggestions': data.get('suggestions', []),
                    'performance_notes': data.get('performance_notes', '')
                }
        except:
            pass
        
        return {
            'optimized_query': original_query,
            'suggestions': [],
            'performance_notes': ''
        }
    
    def _is_sql_query(self, query: str) -> bool:
        """Check if query is SQL"""
        sql_keywords = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'ALTER', 'DROP', 'WITH']
        return any(keyword in query.upper() for keyword in sql_keywords)
    
    def _get_table_schemas(self, table_name: Optional[str]) -> List[Dict[str, Any]]:
        """Get table schemas for context"""
        try:
            if table_name:
                # Get specific table schema
                query = f"""
                SELECT column_name, data_type, nullable
                FROM user_tab_columns
                WHERE table_name = UPPER('{table_name}')
                ORDER BY column_id
                """
                result = db_manager.execute_query(query)
                
                return [{
                    'table_name': table_name,
                    'columns': [row['COLUMN_NAME'] for row in result],
                    'data_type': 'unknown'
                }]
            else:
                # Get all table schemas
                query = """
                SELECT table_name, column_name, data_type
                FROM user_tab_columns
                WHERE table_name IN (
                    SELECT table_name FROM user_tables
                )
                ORDER BY table_name, column_id
                """
                result = db_manager.execute_query(query)
                
                # Group by table
                tables = {}
                for row in result:
                    table_name = row['TABLE_NAME']
                    if table_name not in tables:
                        tables[table_name] = {
                            'table_name': table_name,
                            'columns': [],
                            'data_type': 'unknown'
                        }
                    tables[table_name]['columns'].append(row['COLUMN_NAME'])
                
                return list(tables.values())
                
        except Exception as e:
            logger.error(f"Error getting table schemas: {e}")
            return []
    
    def _fallback_sql_generation(self, natural_query: str, table_schemas: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Fallback SQL generation when LLM is not available"""
        # Simple pattern matching for common queries
        query_lower = natural_query.lower()
        
        if 'select' in query_lower or 'show' in query_lower or 'get' in query_lower:
            if table_schemas:
                table_name = table_schemas[0]['table_name']
                return {
                    'sql': f"SELECT * FROM {table_name} FETCH FIRST 10 ROWS ONLY",
                    'confidence': 0.3,
                    'explanation': 'Basic SELECT query',
                    'oracle_features': []
                }
        
        return {
            'sql': natural_query,
            'confidence': 0.1,
            'explanation': 'No SQL generation available',
            'oracle_features': []
        }
    
    def get_available_models(self) -> List[str]:
        """Get list of available Ollama models"""
        return self.available_models
    
    def switch_model(self, model_name: str) -> bool:
        """Switch to a different Ollama model"""
        if model_name in self.available_models:
            self.model_name = model_name
            logger.info(f"Switched to model: {model_name}")
            return True
        else:
            logger.error(f"Model {model_name} not available")
            return False