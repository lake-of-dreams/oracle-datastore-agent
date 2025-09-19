"""
LLM-Powered Data Format Analyzer
Uses Large Language Models to intelligently analyze and classify data formats
"""
import json
import logging
import requests
from typing import Dict, Any, List, Optional, Union
from enum import Enum
from dataclasses import dataclass
import re

from common.setup import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


class DataType(Enum):
    """Supported data types"""
    JSON = "json"
    CSV = "csv"
    TSV = "tsv"
    TEXT = "text"
    BINARY = "binary"
    XML = "xml"
    YAML = "yaml"
    SQL = "sql"
    LOG = "log"
    CODE = "code"
    UNKNOWN = "unknown"


@dataclass
class LLMDataAnalysis:
    """Enhanced data analysis result with LLM insights"""
    data_type: DataType
    confidence: float
    schema: Optional[Dict[str, Any]] = None
    sample_data: Optional[Any] = None
    metadata: Optional[Dict[str, Any]] = None
    recommendations: Optional[List[str]] = None
    llm_insights: Optional[Dict[str, Any]] = None
    detected_patterns: Optional[List[str]] = None
    suggested_queries: Optional[List[str]] = None


class LLMDataAnalyzer:
    """LLM-powered data analyzer with advanced pattern recognition using Ollama"""
    
    def __init__(self, ollama_host: str = "http://localhost:11434", model_name: str = "llama2"):
        self.ollama_host = ollama_host
        self.model_name = model_name
        self.available_models = []
        self._initialize_ollama()
    
    def _initialize_ollama(self):
        """Initialize Ollama connection"""
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
                logger.warning("Ollama not available, falling back to rule-based analysis")
                self.model_name = None
                
        except Exception as e:
            logger.warning(f"Failed to initialize Ollama: {e}")
            self.model_name = None
    
    def analyze_data(self, data: Union[str, bytes, Dict, List]) -> LLMDataAnalysis:
        """
        Analyze data using LLM-powered intelligence
        
        Args:
            data: Input data in various formats
            
        Returns:
            LLMDataAnalysis object with comprehensive insights
        """
        try:
            # Convert data to string for LLM analysis
            if isinstance(data, (dict, list)):
                data_str = json.dumps(data, indent=2)
                data_type_hint = "structured"
            elif isinstance(data, bytes):
                # For binary data, analyze metadata only
                return self._analyze_binary_data_llm(data)
            else:
                data_str = str(data)
                data_type_hint = "text"
            
            # Use LLM for analysis
            if self.model_name and len(data_str) < 50000:  # Limit for API calls
                return self._analyze_with_llm(data_str, data_type_hint)
            else:
                return self._analyze_with_rules(data_str, data_type_hint)
                
        except Exception as e:
            logger.error(f"Error in LLM data analysis: {e}")
            return LLMDataAnalysis(
                data_type=DataType.UNKNOWN,
                confidence=0.0,
                recommendations=[f"Analysis error: {str(e)}"]
            )
    
    def _analyze_with_llm(self, data_str: str, data_type_hint: str) -> LLMDataAnalysis:
        """Analyze data using Ollama LLM"""
        try:
            # Create analysis prompt
            prompt = self._create_analysis_prompt(data_str, data_type_hint)
            
            # Call Ollama API
            llm_response = self._call_ollama(prompt)
            
            # Parse LLM response
            return self._parse_llm_response(llm_response, data_str)
            
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            return self._analyze_with_rules(data_str, data_type_hint)
    
    def _create_analysis_prompt(self, data_str: str, data_type_hint: str) -> str:
        """Create analysis prompt for LLM"""
        # Truncate data if too long
        sample_data = data_str[:2000] if len(data_str) > 2000 else data_str
        
        prompt = f"""
Analyze the following data and provide a comprehensive analysis:

Data Type Hint: {data_type_hint}
Data Sample:
```
{sample_data}
```

Please analyze this data and provide:
1. Data format type (json, csv, tsv, xml, yaml, sql, log, code, text, binary, unknown)
2. Confidence level (0.0 to 1.0)
3. Detected patterns (list of patterns found)
4. Schema structure (if applicable)
5. Recommendations for storage and querying, only Oracle specific
6. Suggested queries that could be useful
7. Any special characteristics or insights

Format your response as JSON with the following structure:
{{
    "data_type": "detected_type",
    "confidence": 0.95,
    "patterns": ["pattern1", "pattern2"],
    "schema": {{"structure": "description"}},
    "recommendations": ["rec1", "rec2"],
    "suggested_queries": ["query1", "query2"],
    "insights": "additional insights about the data"
}}
"""
        return prompt
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for LLM"""
        return """You are an expert data analyst specializing in data format detection and database schema design. 
        You have deep knowledge of Oracle Database 23ai features including JSON Relational Duality, AI Vector Search, 
        Boolean data types, Wide Tables, and Value LOBs.
        
        Your task is to analyze data samples and provide intelligent recommendations for optimal storage and querying 
        using Oracle Database 23ai capabilities.
        
        Be precise, technical, and provide actionable insights."""
    
    def _parse_llm_response(self, llm_response: str, original_data: str) -> LLMDataAnalysis:
        """Parse LLM response into structured analysis"""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
            if json_match:
                llm_data = json.loads(json_match.group())
            else:
                raise ValueError("No JSON found in LLM response")
            
            # Map data type
            data_type = self._map_data_type(llm_data.get('data_type', 'unknown'))
            
            # Create analysis result
            analysis = LLMDataAnalysis(
                data_type=data_type,
                confidence=float(llm_data.get('confidence', 0.5)),
                schema=llm_data.get('schema'),
                sample_data=original_data[:500] if len(original_data) > 500 else original_data,
                metadata={
                    "llm_analyzed": True,
                    "patterns_detected": llm_data.get('patterns', []),
                    "original_length": len(original_data)
                },
                recommendations=llm_data.get('recommendations', []),
                llm_insights=llm_data.get('insights'),
                detected_patterns=llm_data.get('patterns', []),
                suggested_queries=llm_data.get('suggested_queries', [])
            )
            
            # Add Oracle 23ai specific recommendations
            analysis.recommendations.extend(self._get_oracle_23ai_recommendations(data_type, llm_data))
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            return self._analyze_with_rules(original_data, "text")
    
    def _analyze_with_rules(self, data_str: str, data_type_hint: str) -> LLMDataAnalysis:
        """Fallback rule-based analysis"""
        # Basic pattern detection
        patterns = []
        
        # Check for JSON
        if self._is_json(data_str):
            patterns.append("json_structure")
            data_type = DataType.JSON
            confidence = 0.9
        # Check for CSV
        elif self._is_csv(data_str):
            patterns.append("csv_structure")
            data_type = DataType.CSV
            confidence = 0.8
        # Check for XML
        elif self._is_xml(data_str):
            patterns.append("xml_structure")
            data_type = DataType.XML
            confidence = 0.8
        # Check for YAML
        elif self._is_yaml(data_str):
            patterns.append("yaml_structure")
            data_type = DataType.YAML
            confidence = 0.7
        # Check for SQL
        elif self._is_sql(data_str):
            patterns.append("sql_structure")
            data_type = DataType.SQL
            confidence = 0.8
        # Check for log format
        elif self._is_log(data_str):
            patterns.append("log_structure")
            data_type = DataType.LOG
            confidence = 0.7
        # Check for code
        elif self._is_code(data_str):
            patterns.append("code_structure")
            data_type = DataType.CODE
            confidence = 0.6
        else:
            data_type = DataType.TEXT
            confidence = 0.5
            patterns.append("text_content")
        
        return LLMDataAnalysis(
            data_type=data_type,
            confidence=confidence,
            sample_data=data_str[:500],
            metadata={
                "llm_analyzed": False,
                "patterns_detected": patterns,
                "original_length": len(data_str)
            },
            recommendations=self._get_basic_recommendations(data_type),
            detected_patterns=patterns
        )
    
    def _analyze_binary_data_llm(self, data: bytes) -> LLMDataAnalysis:
        """Analyze binary data with LLM insights"""
        import magic
        
        try:
            mime_type = magic.from_buffer(data, mime=True)
            file_type = magic.from_buffer(data)
            
            # Use LLM to analyze binary data characteristics
            if self.model_name:
                prompt = f"""
Analyze this binary data:
- MIME Type: {mime_type}
- File Type: {file_type}
- Size: {len(data)} bytes

Provide recommendations for storage and processing using Oracle Database 23ai features.
"""
                
                llm_insights = self._call_ollama(prompt)
            else:
                llm_insights = "Binary data analysis using rule-based detection"
            
            return LLMDataAnalysis(
                data_type=DataType.BINARY,
                confidence=1.0,
                metadata={
                    "mime_type": mime_type,
                    "file_type": file_type,
                    "size": len(data),
                    "llm_analyzed": self.model_name is not None
                },
                recommendations=self._get_binary_recommendations(mime_type),
                llm_insights=llm_insights
            )
            
        except Exception as e:
            logger.error(f"Error analyzing binary data: {e}")
            return LLMDataAnalysis(
                data_type=DataType.BINARY,
                confidence=0.5,
                recommendations=["Error analyzing binary data"]
            )
    
    def _map_data_type(self, llm_type: str) -> DataType:
        """Map LLM detected type to DataType enum"""
        type_mapping = {
            'json': DataType.JSON,
            'csv': DataType.CSV,
            'tsv': DataType.TSV,
            'xml': DataType.XML,
            'yaml': DataType.YAML,
            'sql': DataType.SQL,
            'log': DataType.LOG,
            'code': DataType.CODE,
            'text': DataType.TEXT,
            'binary': DataType.BINARY
        }
        return type_mapping.get(llm_type.lower(), DataType.UNKNOWN)
    
    def _get_oracle_23ai_recommendations(self, data_type: DataType, llm_data: Dict) -> List[str]:
        """Get Oracle 23ai specific recommendations based on LLM analysis"""
        recommendations = []
        
        if data_type == DataType.JSON:
            recommendations.extend([
                "Use JSON column type for optimal storage",
                "Leverage JSON Relational Duality for complex queries",
                "Create indexes on frequently accessed JSON fields using JSON_VALUE"
            ])
        elif data_type in [DataType.CSV, DataType.TSV]:
            recommendations.extend([
                "Create table with appropriate column types",
                "Use Boolean data type for true/false columns",
                "Leverage Wide Tables if you have many columns"
            ])
        elif data_type == DataType.TEXT:
            recommendations.extend([
                "Use AI Vector Search for semantic similarity",
                "Store as CLOB with vector embeddings",
                "Create full-text indexes for keyword search"
            ])
        elif data_type == DataType.BINARY:
            recommendations.extend([
                "Use Value LOBs for read-and-forget scenarios",
                "Store metadata in separate columns",
                "Consider extracting text content for searchability"
            ])
        
        return recommendations
    
    def _get_basic_recommendations(self, data_type: DataType) -> List[str]:
        """Get basic recommendations for data type"""
        recommendations = {
            DataType.JSON: ["Use JSON column type", "Create JSON indexes"],
            DataType.CSV: ["Create relational table", "Add appropriate indexes"],
            DataType.TSV: ["Create relational table", "Add appropriate indexes"],
            DataType.XML: ["Use XMLType column", "Create XML indexes"],
            DataType.YAML: ["Convert to JSON for better support", "Use CLOB storage"],
            DataType.SQL: ["Store as CLOB", "Consider parsing for metadata"],
            DataType.LOG: ["Use CLOB storage", "Create full-text indexes"],
            DataType.CODE: ["Store as CLOB", "Consider syntax highlighting"],
            DataType.TEXT: ["Use CLOB or VARCHAR2", "Create full-text indexes"],
            DataType.BINARY: ["Use BLOB storage", "Store metadata separately"]
        }
        return recommendations.get(data_type, ["Use appropriate storage type"])
    
    def _get_binary_recommendations(self, mime_type: str) -> List[str]:
        """Get recommendations for binary data based on MIME type"""
        recommendations = ["Use BLOB storage with Value LOBs"]
        
        if mime_type.startswith('image/'):
            recommendations.extend([
                "Store image metadata separately",
                "Consider thumbnail generation",
                "Use appropriate compression"
            ])
        elif mime_type.startswith('application/pdf'):
            recommendations.extend([
                "Extract text content for searchability",
                "Store PDF metadata",
                "Consider OCR for scanned documents"
            ])
        elif mime_type.startswith('text/'):
            recommendations.extend([
                "Consider converting to text for better querying",
                "Store encoding information"
            ])
        
        return recommendations
    
    # Pattern detection methods
    def _is_json(self, data_str: str) -> bool:
        """Check if data is JSON"""
        try:
            json.loads(data_str)
            return True
        except:
            return False
    
    def _is_csv(self, data_str: str) -> bool:
        """Check if data is CSV"""
        lines = data_str.strip().split('\n')
        if len(lines) < 2:
            return False
        
        # Check for consistent comma separation
        first_line_commas = lines[0].count(',')
        return first_line_commas > 0 and all(line.count(',') == first_line_commas for line in lines[1:5])
    
    def _is_xml(self, data_str: str) -> bool:
        """Check if data is XML"""
        return data_str.strip().startswith('<') and '>' in data_str
    
    def _is_yaml(self, data_str: str) -> bool:
        """Check if data is YAML"""
        yaml_indicators = ['---', ':', '  -', '  ']
        return any(indicator in data_str[:100] for indicator in yaml_indicators)
    
    def _is_sql(self, data_str: str) -> bool:
        """Check if data is SQL"""
        sql_keywords = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'ALTER', 'DROP']
        return any(keyword in data_str.upper() for keyword in sql_keywords)
    
    def _is_log(self, data_str: str) -> bool:
        """Check if data is log format"""
        log_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # Date pattern
            r'\d{2}:\d{2}:\d{2}',  # Time pattern
            r'\[.*\]',             # Bracket patterns
            r'(ERROR|WARN|INFO|DEBUG)',  # Log levels
        ]
        return any(re.search(pattern, data_str) for pattern in log_patterns)
    
    def _is_code(self, data_str: str) -> bool:
        """Check if data is code"""
        code_indicators = [
            'def ', 'function ', 'class ', 'import ', 'from ',
            'public ', 'private ', 'protected ', 'static ',
            'int ', 'string ', 'var ', 'let ', 'const ',
            '<?php', '#!/', '//', '/*', '#include'
        ]
        return any(indicator in data_str for indicator in code_indicators)
    
    def generate_smart_queries(self, analysis: LLMDataAnalysis, table_name: str) -> List[str]:
        """Generate smart queries based on LLM analysis"""
        queries = []
        
        if analysis.data_type == DataType.JSON:
            # Generate JSON queries based on detected patterns
            if 'user' in str(analysis.detected_patterns):
                queries.extend([
                    f"SELECT JSON_VALUE(data, '$.user.name') FROM {table_name}",
                    f"SELECT * FROM {table_name} WHERE JSON_VALUE(data, '$.user.active') = 'true'"
                ])
        
        elif analysis.data_type in [DataType.CSV, DataType.TSV]:
            # Generate column-based queries
            queries.extend([
                f"SELECT * FROM {table_name} LIMIT 10",
                f"SELECT COUNT(*) FROM {table_name}"
            ])
        
        elif analysis.data_type == DataType.TEXT:
            # Generate text search queries
            queries.extend([
                f"SELECT * FROM {table_name} WHERE CONTAINS(content, 'search_term')",
                f"SELECT * FROM {table_name} ORDER BY VECTOR_DISTANCE(content_vector, query_vector, COSINE)"
            ])
        
        return queries
    
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
                output = result.get('response', '').strip()
                logger.info(f"LLM Output:{output}")
                return output
            else:
                raise Exception(f"Ollama API error: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error calling Ollama: {e}")
            raise
    
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
