"""
Oracle Datastore Agent Package
A powerful agent for intelligent data storage and retrieval using Oracle Database 23ai
"""

__version__ = "1.0.0"
__author__ = "Oracle Datastore Team"
__description__ = "Intelligent data storage and retrieval with Oracle Database 23ai"

from .agent import llm_oracle_agent, OracleDatastoreAgent, LLMAgentRespone
from .analyzer import LLMDataAnalyzer, DataType, LLMDataAnalysis
from .schema import SchemaGenerator, TableSchema
from .ingestion import DataIngestionEngine
from .query import LLMQueryEngine, LLMQueryResult
from .database import db_manager, OracleConnectionManager

__all__ = [
    'llm_oracle_agent',
    'OracleDatastoreAgent', 
    'LLMAgentRespone',
    'LLMDataAnalyzer',
    'DataType', 
    'LLMDataAnalysis',
    'SchemaGenerator',
    'TableSchema',
    'DataIngestionEngine',
    'LLMQueryEngine',
    'LLMQueryResult',
    'db_manager',
    'OracleConnectionManager'
]

