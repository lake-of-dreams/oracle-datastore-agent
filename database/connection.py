"""
Oracle Database 23ai Connection Manager
Handles database connections and Oracle 23ai specific features
"""
import logging
import oracledb
from typing import Optional, Dict, Any, List
from contextlib import contextmanager
from common.setup import setup_logging
from config import settings


setup_logging()
logger = logging.getLogger(__name__)


class OracleConnectionManager:
    """Manages Oracle Database 23ai connections and features"""
    
    def __init__(self):
        self.connection_pool = None
        self._initialize_connection()
    
    def _initialize_connection(self):
        """Initialize Oracle connection pool"""
        try:
            # Build connection string
            if settings.oracle_service_name:
                dsn = f"{settings.oracle_host}:{settings.oracle_port}/{settings.oracle_service_name}"
            elif settings.oracle_sid:
                dsn = f"{settings.oracle_host}:{settings.oracle_port}:{settings.oracle_sid}"
            else:
                dsn = f"{settings.oracle_host}:{settings.oracle_port}/XE"
            
            # Create connection pool
            self.connection_pool = oracledb.create_pool(
                user=settings.oracle_user,
                password=settings.oracle_password,
                dsn=dsn,
                min=1,
                max=10,
                increment=1
            )
            
            logger.info("Oracle Database connection pool initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Oracle connection: {e}")
            raise
    
    @contextmanager
    def get_connection(self):
        """Get a database connection from the pool"""
        connection = None
        try:
            connection = self.connection_pool.acquire()
            yield connection
        except Exception as e:
            if connection:
                connection.rollback()
            logger.error(f"Database connection error: {e}")
            raise
        finally:
            if connection:
                self.connection_pool.release(connection)
    
    def execute_query(self, query: str, params: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Execute a SELECT query and return results"""
        logger.info(f"Executing Query: {query}, params:{params}")
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                
                # Get column names
                columns = [desc[0] for desc in cursor.description]
                
                # Fetch all results
                results = []
                for row in cursor.fetchall():
                    results.append(dict(zip(columns, row)))
                
                return results
                
            finally:
                cursor.close()
    
    def execute_ddl(self, ddl_statement: str) -> bool:
        """Execute DDL statements (CREATE, ALTER, DROP)"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                logger.info(f"Executing DDL: {ddl_statement}")
                cursor.execute(ddl_statement)
                conn.commit()
                logger.info(f"DDL executed successfully: {ddl_statement[:100]}...")
                return True
            except Exception as e:
                conn.rollback()
                logger.error(f"DDL execution failed: {e}")
                raise
            finally:
                cursor.close()
    
    def execute_dml(self, dml_statement: str, params: Optional[Dict] = None) -> int:
        """Execute DML statements (INSERT, UPDATE, DELETE)"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                if params:
                    cursor.execute(dml_statement, params)
                else:
                    cursor.execute(dml_statement)
                
                conn.commit()
                rowcount = cursor.rowcount
                logger.info(f"DML executed successfully, {rowcount} rows affected")
                return rowcount
                
            except Exception as e:
                conn.rollback()
                logger.error(f"DML execution failed: {e}")
                raise
            finally:
                cursor.close()
    
    def check_oracle_23ai_features(self) -> Dict[str, bool]:
        """Check if Oracle Database 23ai features are available"""
        features = {}
        
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Check Oracle version
                cursor.execute("SELECT BANNER FROM V$VERSION WHERE ROWNUM = 1")
                version = cursor.fetchone()[0]
                features['version'] = version
                
                # Check for JSON support
                try:
                    cursor.execute("SELECT JSON_OBJECT('test' VALUE 'value') FROM DUAL")
                    features['json_support'] = True
                except:
                    features['json_support'] = False
                
                # Check for Boolean data type
                try:
                    cursor.execute("CREATE GLOBAL TEMPORARY TABLE test_boolean (id NUMBER, flag BOOLEAN)")
                    cursor.execute("DROP TABLE test_boolean")
                    features['boolean_support'] = True
                except:
                    features['boolean_support'] = False
                
                # Check for Vector Search (AI Vector Search)
                try:
                    cursor.execute("SELECT * FROM V$OPTION WHERE PARAMETER = 'Oracle AI Vector Search'")
                    result = cursor.fetchone()
                    features['vector_search'] = result is not None and result[1] == 'TRUE'
                except:
                    features['vector_search'] = False
                
                # Check for Wide Tables (4096 columns)
                try:
                    cursor.execute("SELECT * FROM V$PARAMETER WHERE NAME = 'max_columns_per_table'")
                    result = cursor.fetchone()
                    features['wide_tables'] = result is not None
                except:
                    features['wide_tables'] = False
                
                cursor.close()
                
        except Exception as e:
            logger.error(f"Error checking Oracle 23ai features: {e}")
            features['error'] = str(e)
        
        return features
    
    def create_vector_index(self, table_name: str, column_name: str, index_name: str) -> bool:
        """Create a vector index for AI Vector Search"""
        try:
            ddl = f"""
            CREATE VECTOR INDEX {index_name}
            ON {table_name} ({column_name})
            INDEXTYPE IS VECTOR_INDEX
            PARAMETERS ('DIMENSION {settings.vector_dimension} DISTANCE COSINE')
            """
            
            self.execute_ddl(ddl)
            logger.info(f"Vector index {index_name} created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create vector index: {e}")
            return False
    
    def close(self):
        """Close the connection pool"""
        if self.connection_pool:
            self.connection_pool.close()
            logger.info("Oracle connection pool closed")


# Global connection manager instance
db_manager = OracleConnectionManager()
