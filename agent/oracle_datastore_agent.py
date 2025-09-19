"""
LLM-Enhanced Oracle Datastore Agent
"""
import logging
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime

from analyzer.llm_data_analyzer import LLMDataAnalysis, LLMDataAnalyzer
from schema import SchemaGenerator
from ingestion import DataIngestionEngine
from query import LLMQueryEngine, LLMQueryResult
from database import db_manager
from common import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


@dataclass
class LLMAgentResponse:
    """Enhanced response from the LLM-enhanced Oracle Datastore Agent"""
    success: bool
    message: str
    data: Optional[Any] = None
    metadata: Optional[Dict[str, Any]] = None
    oracle_23ai_features_used: Optional[List[str]] = None
    
    # LLM-specific fields
    analysis_confidence: Optional[float] = None
    detected_patterns: Optional[List[str]] = None
    suggested_queries: Optional[List[str]] = None
    llm_insights: Optional[str] = None


class LLMEnhancedOracleDatastoreAgent:
    """
    LLM-enhanced Oracle Datastore Agent with intelligent data analysis
    """
    
    def __init__(self, ollama_host: str = "http://localhost:11434", model_name: str = "llama2"):
        self.analyzer = LLMDataAnalyzer(ollama_host, model_name)
        self.schema_generator = SchemaGenerator()
        self.ingestion_engine = DataIngestionEngine()
        self.query_engine = LLMQueryEngine(ollama_host, model_name)
        self.stored_tables = {}
        
        logger.info("LLM-Enhanced Oracle Datastore Agent initialized with Ollama")
    
    def store_data(self, data: Union[str, bytes, Dict, List], 
                  table_name: Optional[str] = None) -> LLMAgentResponse:
        """
        Store data using LLM-enhanced analysis
        
        Args:
            data: Input data in various formats
            table_name: Optional custom table name
            
        Returns:
            LLMAgentResponse with enhanced insights
        """
        try:
            # Step 1: LLM-enhanced analysis
            logger.info("Running LLM-enhanced data analysis...")
            analysis = self.analyzer.analyze_data(data)
            
            if analysis.data_type == "unknown":
                return LLMAgentResponse(
                    success=False,
                    message="Unable to determine data format with LLM analysis",
                    metadata={"analysis": analysis},
                    analysis_confidence=analysis.confidence,
                    llm_insights=analysis.llm_insights
                )
            
            # Step 2: Generate optimal schema
            logger.info(f"Generating schema for {analysis.data_type} data...")
            schema = self.schema_generator.generate_schema(analysis, table_name)
            # Step 3: Store the data
            logger.info(f"Storing data in table {schema.table_name}...")
            ingestion_result = self.ingestion_engine.ingest_data(data, analysis, schema)
            
            if ingestion_result["success"]:
                # Track the stored table
                self.stored_tables[schema.table_name] = {
                    "data_type": analysis.data_type,
                    "schema": schema,
                    "analysis": analysis,
                    "created_at": datetime.now()
                }
                
                return LLMAgentResponse(
                    success=True,
                    message=f"Data stored successfully in table '{schema.table_name}' using LLM-enhanced analysis",
                    data={
                        "table_name": schema.table_name,
                        "rows_inserted": ingestion_result["rows_inserted"],
                        "data_type": analysis.data_type
                    },
                    metadata={
                        "analysis": analysis,
                        "schema": schema,
                        "recommendations": analysis.recommendations
                    },
                    oracle_23ai_features_used=schema.oracle_23ai_features,
                    analysis_confidence=analysis.confidence,
                    detected_patterns=analysis.detected_patterns,
                    suggested_queries=analysis.suggested_queries,
                    llm_insights=analysis.llm_insights if analysis.llm_insights else None,
                )
            else:
                return LLMAgentResponse(
                    success=False,
                    message=f"Failed to store data: {ingestion_result.get('error', 'Unknown error')}",
                    metadata={"analysis": analysis, "schema": schema},
                    analysis_confidence=analysis.confidence,
                )
                
        except Exception as e:
            logger.error(f"Error in LLM-enhanced store_data: {e}")
            return LLMAgentResponse(
                success=False,
                message=f"Error storing data: {str(e)}"
            )
    
    def create_queries(self, query: str, table_name: Optional[str] = None) -> LLMAgentResponse:
        """
        Query stored data with LLM-enhanced query understanding
        
        Args:
            query: Query string (SQL, natural language, or specific query type)
            table_name: Optional table name to query
            
        Returns:
            LLMAgentResponse with query results
        """
        try:
            # Enhanced query processing with LLM insights
            query_analysis = self._analyze_query_intent(query)
            
            # Creating query using LLM query engine
            result = self.query_engine.create_intelligent_query(query, table_name)
            
            return LLMAgentResponse(
                success=True,
                message=f"Query created successfully",
                metadata={
                    "table_name": table_name,
                    "query_analysis": query_analysis,
                    "generated_sql": result.generated_sql,
                    "query_explanation": result.query_explanation,
                    "optimization_suggestions": result.optimization_suggestions,
                    "alternative_queries": result.alternative_queries,
                    "original_query": result.original_query,
                    "llm_confidence": result.llm_confidence,
                    "execution_time": result.execution_time
                },
                llm_insights=f"Query analyzed with confidence {result.llm_confidence:.2f}. {result.query_explanation}"
            )
            
        except Exception as e:
            logger.error(f"Error in LLM-enhanced query_data: {e}")
            return LLMAgentResponse(
                success=False,
                message=f"Error creating query: {str(e)}"
            )
    
    def analyze_data_intelligence(self, data: Union[str, bytes, Dict, List]) -> LLMAgentResponse:
        """
        Perform comprehensive LLM-powered data analysis without storing
        
        Args:
            data: Input data to analyze
            
        Returns:
            LLMAgentResponse with detailed analysis
        """
        try:
            analysis = self.analyzer.analyze_data(data)
            
            # Generate additional insights
            smart_queries = []
            if analysis.suggested_queries:
                smart_queries = analysis.suggested_queries
            elif analysis.data_type in ["json", "csv", "tsv"]:
                # Generate queries based on detected patterns
                smart_queries = self._generate_smart_queries(analysis)
            
            return LLMAgentResponse(
                success=True,
                message=f"Comprehensive analysis completed for {analysis.data_type} data",
                data={
                    "data_type": analysis.data_type,
                    "confidence": analysis.confidence,
                    "patterns": analysis.detected_patterns,
                    "recommendations": analysis.recommendations,
                    "smart_queries": smart_queries
                },
                metadata={
                    "analysis": analysis,
                    "confidence_breakdown": analysis.confidence_breakdown
                },
                analysis_confidence=analysis.confidence,
                detected_patterns=analysis.detected_patterns,
                suggested_queries=smart_queries,
                llm_insights=analysis.llm_insights if analysis.llm_insights else None,
            )
            
        except Exception as e:
            logger.error(f"Error in data intelligence analysis: {e}")
            return LLMAgentResponse(
                success=False,
                message=f"Error analyzing data: {str(e)}"
            )
    
    def get_smart_recommendations(self, table_name: str) -> LLMAgentResponse:
        """
        Get smart recommendations for a stored table
        
        Args:
            table_name: Name of the table
            
        Returns:
            LLMAgentResponse with recommendations
        """
        try:
            if table_name not in self.stored_tables:
                return LLMAgentResponse(
                    success=False,
                    message=f"Table '{table_name}' not found"
                )
            
            table_info = self.stored_tables[table_name]
            analysis = table_info["analysis"]
            
            # Generate smart recommendations
            recommendations = self.analyzer.generate_smart_recommendations(analysis)
            
            # Add table-specific recommendations
            table_recommendations = self._generate_table_specific_recommendations(table_name, analysis)
            recommendations.extend(table_recommendations)
            
            return LLMAgentResponse(
                success=True,
                message=f"Smart recommendations generated for table '{table_name}'",
                data={
                    "table_name": table_name,
                    "recommendations": recommendations,
                    "data_type": analysis.data_type,
                    "confidence": analysis.confidence
                },
                analysis_confidence=analysis.confidence,
                detected_patterns=analysis.detected_patterns
            )
            
        except Exception as e:
            logger.error(f"Error generating smart recommendations: {e}")
            return LLMAgentResponse(
                success=False,
                message=f"Error generating recommendations: {str(e)}"
            )
    
    def _analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """Analyze query intent using pattern recognition"""
        query_lower = query.lower()
        
        # SQL patterns
        if any(keyword in query_lower for keyword in ['select', 'insert', 'update', 'delete', 'create', 'alter']):
            return {"type": "sql", "confidence": 0.9}
        
        # JSON patterns
        if any(keyword in query_lower for keyword in ['json', 'field', 'path', '$.']):
            return {"type": "json", "confidence": 0.8}
        
        # Vector similarity patterns
        if any(keyword in query_lower for keyword in ['similar', 'vector', 'embedding', 'find similar']):
            return {"type": "vector", "confidence": 0.8}
        
        # Natural language patterns
        if any(keyword in query_lower for keyword in ['find', 'search', 'show', 'get', 'list']):
            return {"type": "natural_language", "confidence": 0.7}
        
        return {"type": "unknown", "confidence": 0.5}

    
    def _generate_smart_queries(self, analysis: LLMDataAnalysis) -> List[str]:
        """Generate smart queries based on analysis"""
        queries = []
        
        if analysis.data_type == "json":
            queries.extend([
                "SELECT JSON_VALUE(data, '$.field_name') FROM table_name",
                "SELECT * FROM table_name WHERE JSON_VALUE(data, '$.status') = 'active'"
            ])
        elif analysis.data_type in ["csv", "tsv"]:
            queries.extend([
                "SELECT * FROM table_name LIMIT 10",
                "SELECT COUNT(*) FROM table_name",
                "SELECT DISTINCT column_name FROM table_name"
            ])
        elif analysis.data_type == "text":
            queries.extend([
                "SELECT * FROM table_name WHERE CONTAINS(content, 'search_term')",
                "SELECT * FROM table_name ORDER BY VECTOR_DISTANCE(content_vector, query_vector, COSINE)"
            ])
        
        return queries
    
    def _generate_table_specific_recommendations(self, table_name: str, analysis: LLMDataAnalysis) -> List[str]:
        """Generate table-specific recommendations"""
        recommendations = []
        
        # Add recommendations based on table characteristics
        if analysis.data_type == "json":
            recommendations.append(f"Create indexes on frequently queried JSON fields in {table_name}")
            recommendations.append(f"Consider using JSON_TABLE for complex JSON queries on {table_name}")
        
        elif analysis.data_type in ["csv", "tsv"]:
            recommendations.append(f"Add appropriate indexes on key columns in {table_name}")
            recommendations.append(f"Consider partitioning {table_name} if it grows large")
        
        elif analysis.data_type == "text":
            recommendations.append(f"Create vector indexes for similarity search on {table_name}")
            recommendations.append(f"Consider full-text indexes for keyword search on {table_name}")
        
        return recommendations
    
    def _find_table_by_type(self, data_type: str) -> Optional[str]:
        """Find a table by data type"""
        for table_name, info in self.stored_tables.items():
            if info["data_type"] == data_type:
                return table_name
        return None
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get summary of all stored tables and their analyses"""
        summary = {
            "total_tables": len(self.stored_tables),
            "tables": []
        }
        
        for table_name, info in self.stored_tables.items():
            analysis = info["analysis"]
            table_summary = {
                "table_name": table_name,
                "data_type": analysis.data_type,
                "confidence": analysis.confidence,
                "patterns_detected": len(analysis.detected_patterns or []),
                "oracle_features": len(info["schema"].oracle_23ai_features),
                "created_at": info["created_at"]
            }
            summary["tables"].append(table_summary)
        
        return summary
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            db_manager.close()
            logger.info("LLM-Enhanced Oracle Datastore Agent cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def get_oracle_23ai_features(self) -> LLMAgentResponse:
        """Get information about Oracle Database 23ai features"""
        try:
            features = db_manager.check_oracle_23ai_features()
            
            return LLMAgentResponse(
                success=True,
                message="Oracle Database 23ai features check completed",
                data=features
            )
            
        except Exception as e:
            logger.error(f"Error checking Oracle 23ai features: {e}")
            return LLMAgentResponse(
                success=False,
                message=f"Error checking Oracle 23ai features: {str(e)}"
            )
    
    def get_table_info(self, table_name: Optional[str] = None) -> LLMAgentResponse:
        """
        Get information about stored tables
        
        Args:
            table_name: Optional specific table name
            
        Returns:
            AgentResponse with table information
        """
        try:
            if table_name:
                if table_name in self.stored_tables:
                    table_info = self.query_engine.get_table_metadata(table_name)
                    stored_info = self.stored_tables[table_name]
                    
                    return LLMAgentResponse(
                        success=True,
                        message=f"Table information for '{table_name}'",
                        data={
                            "table_name": table_name,
                            "data_type": stored_info["data_type"].value,
                            "created_at": stored_info["created_at"],
                            "metadata": table_info
                        },
                        oracle_23ai_features_used=stored_info["schema"].oracle_23ai_features
                    )
                else:
                    return LLMAgentResponse(
                        success=False,
                        message=f"Table '{table_name}' not found"
                    )
            else:
                # Return all tables
                tables_info = []
                for name, info in self.stored_tables.items():
                    tables_info.append({
                        "table_name": name,
                        "data_type": info["data_type"].value,
                        "created_at": info["created_at"],
                        "oracle_23ai_features": info["schema"].oracle_23ai_features
                    })
                
                return LLMAgentResponse(
                    success=True,
                    message=f"Found {len(tables_info)} tables",
                    data=tables_info
                )
                
        except Exception as e:
            logger.error(f"Error in get_table_info: {e}")
            return LLMAgentResponse(
                success=False,
                message=f"Error getting table information: {str(e)}"
            )


llm_oracle_agent = LLMEnhancedOracleDatastoreAgent()