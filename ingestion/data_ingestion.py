"""
Data Ingestion Engine for Oracle Database 23ai
Handles data insertion with Oracle 23ai specific features
"""
import json
import logging
import hashlib
import io
import csv
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from sentence_transformers import SentenceTransformer
from common.setup import setup_logging
from database import db_manager
from analyzer import LLMDataAnalysis, DataType
from schema import TableSchema


setup_logging()
logger = logging.getLogger(__name__)


class DataIngestionEngine:
    """Handles data ingestion with Oracle 23ai features"""
    
    def __init__(self):
        self.vector_model = None
        self._initialize_vector_model()
    
    def _initialize_vector_model(self):
        """Initialize sentence transformer model for vector embeddings"""
        try:
            # Use a lightweight model for embeddings
            self.vector_model = SentenceTransformer(model_name_or_path='all-MiniLM-L6-v2', device='cpu')
            logger.info("Vector model initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize vector model: {e}")
            self.vector_model = None
    
    def ingest_data(self, data: Union[str, bytes, Dict, List], 
                   analysis: LLMDataAnalysis, schema: TableSchema) -> Dict[str, Any]:
        """
        Ingest data into Oracle Database 23ai
        
        Args:
            data: Input data
            analysis: Data analysis result
            schema: Generated table schema
            
        Returns:
            Dictionary with ingestion results
        """
        try:
            # Create table if it doesn't exist
            self._create_table(schema)
            
            # Prepare data for insertion
            prepared_data = self._prepare_data_for_insertion(data, analysis, schema)
            
            # Insert data
            result = self._insert_data(prepared_data, schema)
            
            logger.info(f"Data ingested successfully into table {schema.table_name}")
            return {
                "success": True,
                "table_name": schema.table_name,
                "rows_inserted": result.get("rows_inserted", 0),
                "oracle_23ai_features_used": schema.oracle_23ai_features
            }
            
        except Exception as e:
            logger.error(f"Data ingestion failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "table_name": schema.table_name
            }
    
    def _create_table(self, schema: TableSchema):
        """Create table using the generated schema"""
        try:
            # Check if table exists
            check_query = f"""
            SELECT COUNT(*) as table_count 
            FROM user_tables 
            WHERE table_name = UPPER('{schema.table_name}')
            """
            
            result = db_manager.execute_query(check_query)
            if result[0]["TABLE_COUNT"] > 0:
                logger.info(f"Table {schema.table_name} already exists")
                return
            
            # Generate and execute DDL
            from schema.schema_generator import SchemaGenerator
            generator = SchemaGenerator()
            ddl = generator.generate_ddl(schema)
            # Execute DDL statements
            ddl_statements = ddl.split(";\n\n")
            for statement in ddl_statements:
                if statement.strip():
                    db_manager.execute_ddl(statement.strip())
            
            logger.info(f"Table {schema.table_name} created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create table {schema.table_name}: {e}")
            raise
    
    def _prepare_data_for_insertion(self, data: Union[str, bytes, Dict, List], 
                                  analysis: LLMDataAnalysis, schema: TableSchema) -> List[Dict[str, Any]]:
        """Prepare data for database insertion"""
        prepared_data = []
        
        if analysis.data_type == DataType.JSON:
            prepared_data = self._prepare_json_data(data, schema)
        elif analysis.data_type in [DataType.CSV, DataType.TSV]:
            prepared_data = self._prepare_tabular_data(data, analysis, schema)
        elif analysis.data_type == DataType.TEXT:
            prepared_data = self._prepare_text_data(data, schema)
        elif analysis.data_type == DataType.BINARY:
            prepared_data = self._prepare_binary_data(data, analysis, schema)
        else:
            prepared_data = self._prepare_generic_data(data, analysis, schema)
        
        return prepared_data
    
    def _prepare_json_data(self, data: Union[Dict, List], schema: TableSchema) -> List[Dict[str, Any]]:
        """Prepare JSON data for insertion"""
        prepared_data = []
        
        if isinstance(data, list):
            for item in data:
                row_data = {
                    "json_data": json.dumps(item),
                    "created_at": datetime.now(),
                    "updated_at": datetime.now()
                }
                
                if "metadata" in [col["name"] for col in schema.columns]:
                    row_data["metadata"] = json.dumps({"source": "json_array", "count": len(data)})
                
                prepared_data.append(row_data)
        else:
            row_data = {
                "json_data": json.dumps(data),
                "created_at": datetime.now(),
                "updated_at": datetime.now()
            }
            
            if "metadata" in [col["name"] for col in schema.columns]:
                row_data["metadata"] = json.dumps({"source": "json_object"})
            
            prepared_data.append(row_data)
        
        return prepared_data
    
    def _prepare_tabular_data(self, data: str, analysis: LLMDataAnalysis, schema: TableSchema) -> List[Dict[str, Any]]:
        """Prepare CSV/TSV data for insertion"""
        prepared_data = []
        
        # Parse CSV/TSV data
        delimiter = '\t' if analysis.data_type == DataType.TSV else ','
        csv_reader = csv.reader(io.StringIO(data), delimiter=delimiter)
        rows = list(csv_reader)
        
        if not rows:
            return prepared_data
        
        header = rows[0]
        
        for row in rows[1:]:  # Skip header
            if len(row) != len(header):
                continue  # Skip malformed rows
            
            row_data = {
                "created_at": datetime.now(),
                "updated_at": datetime.now()
            }
            
            # Map data to columns
            for i, value in enumerate(row):
                if i < len(header):
                    col_name = self._sanitize_column_name(header[i])
                    row_data[col_name] = self._convert_value(value, schema, col_name)
            
            prepared_data.append(row_data)
        
        return prepared_data
    
    def _prepare_text_data(self, data: str, schema: TableSchema) -> List[Dict[str, Any]]:
        """Prepare text data for insertion with vector embedding"""
        prepared_data = []
        
        # Generate vector embedding if model is available
        content_vector = None
        if self.vector_model:
            try:
                embedding = self.vector_model.encode(data)
                content_vector = embedding.tolist()
            except Exception as e:
                logger.warning(f"Failed to generate vector embedding: {e}")
        
        # Calculate content hash
        content_hash = hashlib.sha256(data.encode('utf-8')).hexdigest()
        
        # Count words
        word_count = len(data.split())
        
        row_data = {
            "content": data,
            "content_vector": content_vector,
            "content_hash": content_hash,
            "word_count": word_count,
            "created_at": datetime.now()
        }
        
        # Add content length if column exists
        if "content_length" in [col["name"] for col in schema.columns]:
            row_data["content_length"] = len(data)
        
        prepared_data.append(row_data)
        
        return prepared_data
    
    def _prepare_binary_data(self, data: bytes, analysis: LLMDataAnalysis, schema: TableSchema) -> List[Dict[str, Any]]:
        """Prepare binary data for insertion"""
        prepared_data = []
        
        # Calculate file hash
        file_hash = hashlib.sha256(data).hexdigest()
        
        row_data = {
            "binary_data": data,
            "file_size": len(data),
            "file_hash": file_hash,
            "created_at": datetime.now()
        }
        
        # Add metadata if available
        if analysis.metadata:
            if "mime_type" in analysis.metadata:
                row_data["mime_type"] = analysis.metadata["mime_type"]
            if "file_type" in analysis.metadata:
                row_data["file_type_description"] = analysis.metadata["file_type"]
        
        prepared_data.append(row_data)
        
        return prepared_data
    
    def _prepare_generic_data(self, data: Union[str, bytes], analysis: LLMDataAnalysis, schema: TableSchema) -> List[Dict[str, Any]]:
        """Prepare generic data for insertion"""
        prepared_data = []
        
        if isinstance(data, bytes):
            data_str = data.decode('utf-8', errors='ignore')
        else:
            data_str = str(data)
        
        row_data = {
            "raw_data": data_str,
            "data_type": analysis.data_type.value,
            "created_at": datetime.now()
        }
        
        prepared_data.append(row_data)
        
        return prepared_data
    
    def _insert_data(self, prepared_data: List[Dict[str, Any]], schema: TableSchema) -> Dict[str, Any]:
        """Insert prepared data into the database"""
        if not prepared_data:
            return {"rows_inserted": 0}
        
        # Get column names (excluding ID which is auto-generated)
        columns = [col["name"] for col in schema.columns if col["name"].upper() != "ID"]
        
        # Build INSERT statement
        placeholders = ", ".join([f":{col.lower()}" for col in columns])
        insert_sql = f"INSERT INTO {schema.table_name} ({', '.join(columns)}) VALUES ({placeholders})"
        
        rows_inserted = 0
        
        try:
            with db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                for row_data in prepared_data:
                    # Convert data types for Oracle
                    oracle_row = self._convert_for_oracle(row_data, schema)
                    
                    # Execute insert
                    cursor.execute(insert_sql, oracle_row)
                    rows_inserted += 1
                
                conn.commit()
                cursor.close()
                
        except Exception as e:
            logger.error(f"Failed to insert data: {e}")
            raise
        
        return {"rows_inserted": rows_inserted}
    
    def _convert_value(self, value: str, schema: TableSchema, column_name: str) -> Any:
        """Convert string value to appropriate type based on schema"""
        # Find column definition
        column_def = None
        for col in schema.columns:
            if col["name"] == column_name:
                column_def = col
                break
        
        if not column_def:
            return value
        
        column_type = column_def["type"].upper()
        
        try:
            if "NUMBER" in column_type:
                if '.' in value:
                    return float(value)
                else:
                    return int(value)
            elif "DATE" in column_type or "TIMESTAMP" in column_type:
                # Try to parse common date formats
                from datetime import datetime
                for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%d-%m-%Y', '%Y-%m-%d %H:%M:%S']:
                    try:
                        return datetime.strptime(value, fmt)
                    except ValueError:
                        continue
                return value  # Return as string if can't parse
            elif "BOOLEAN" in column_type:
                return value.lower() in ['true', '1', 'yes', 'y']
            else:
                return value
        except (ValueError, TypeError):
            return value  # Return original value if conversion fails
    
    def _convert_for_oracle(self, row_data: Dict[str, Any], schema: TableSchema) -> Dict[str, Any]:
        """Convert Python data types to Oracle-compatible types"""
        oracle_row = {}
        
        for key, value in row_data.items():
            if value is None:
                oracle_row[key.lower()] = None
            elif isinstance(value, (list, dict)):
                # Convert to JSON string for Oracle
                oracle_row[key.lower()] = json.dumps(value)
            elif isinstance(value, bytes):
                # Handle binary data
                oracle_row[key.lower()] = value
            else:
                oracle_row[key.lower()] = value
        
        return oracle_row
    
    def _sanitize_column_name(self, name: str) -> str:
        """Sanitize column name for Oracle database"""
        import re
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        
        if sanitized and not sanitized[0].isalpha():
            sanitized = f"col_{sanitized}"
        
        if not sanitized:
            sanitized = "unnamed_column"
        
        if len(sanitized) > 30:
            sanitized = sanitized[:30]
        
        return sanitized.upper()
    
    def batch_ingest(self, data_list: List[Union[str, bytes, Dict, List]], 
                    analyses: List[LLMDataAnalysis], schemas: List[TableSchema]) -> List[Dict[str, Any]]:
        """Ingest multiple datasets in batch"""
        results = []
        
        for data, analysis, schema in zip(data_list, analyses, schemas):
            result = self.ingest_data(data, analysis, schema)
            results.append(result)
        
        return results

