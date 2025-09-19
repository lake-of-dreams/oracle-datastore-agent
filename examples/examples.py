"""
Examples for Oracle Datastore Agent
"""
import json
import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import  llm_oracle_agent
from analyzer import LLMDataAnalyzer
from query import LLMQueryEngine


def example_basic_functionality():
    """Example: Basic Oracle Datastore Agent functionality"""
    print("=" * 60)
    print("📊 BASIC ORACLE DATASTORE AGENT EXAMPLES")
    print("=" * 60)
    
    # Sample data for testing
    sample_data = {
        "employees": [
            {"id": 1, "name": "Alice Johnson", "department": "Engineering", "salary": 75000, "active": True},
            {"id": 2, "name": "Bob Smith", "department": "Marketing", "salary": 65000, "active": True},
            {"id": 3, "name": "Charlie Brown", "department": "Engineering", "salary": 80000, "active": False}
        ]
    }
    
    print("\n1. Storing JSON Data")
    print("-" * 30)
    response = llm_oracle_agent.store_data(sample_data, "employees_data")
    
    if response.success:
        print(f"✅ Data stored successfully!")
        print(f"   Table: {response.data['table_name']}")
        print(f"   Rows: {response.data['rows_inserted']}")
        print(f"   Oracle 23ai Features: {', '.join(response.oracle_23ai_features_used)}")
        
        # Query the data
        print("\n2. Querying Stored Data")
        print("-" * 30)
        
        queries = [
            "find all active employees",
            "find all employees grouped by department",
            "find all employees with sssalary > 70000"
        ]
        
        for query in queries:
            print(f"\nQuery: {query}")
            result = llm_oracle_agent.create_queries(query)
            if result.success:
                print(f"**OriginaL Query:** {result.metadata.get('original_query', 'Unknown')}\n")
                print(f"**Generated SQL:** {result.metadata.get('generated_sql', 'Unknown')}\n")
                print(f"**Execution Time:** {result.metadata.get('execution_time', 0):.3f} seconds\n\n")
                print(f"**LLM Confidence:** {result.metadata.get('llm_confidence', 0):.3f} seconds\n\n")
                print(f"**Query Explaination:** {result.metadata.get('query_explanation', 'Unknown')}\n")
                print(f"**Optimization Suggestions:** {result.metadata.get('optimization_suggestions', 'Unknown')}\n")
                print(f"**Alternative Queries:** {result.metadata.get('alternative_queries', 'Unknown')}\n")
            else:
                print(f"   ❌ Failed: {result.message}")
    else:
        print(f"❌ Storage failed: {response.message}")


def example_llm_enhanced_analysis():
    """Example: LLM-enhanced data analysis"""
    print("\n" + "=" * 60)
    print("🧠 LLM-ENHANCED ANALYSIS EXAMPLES")
    print("=" * 60)
    
    # Complex data for LLM analysis
    complex_data = """
# Configuration for Oracle Datastore Agent
database:
  host: localhost
  port: 1521
  service_name: FREEPDB1
  
features:
  enable_llm_analysis: true
  enable_vector_search: true
  max_connections: 10

# User roles and permissions
users:
  - name: admin
    role: administrator
    permissions: [read, write, delete, manage]
  - name: analyst
    role: data_analyst
    permissions: [read, query, analyze]
  - name: viewer
    role: read_only
    permissions: [read]

# API endpoints
endpoints:
  - path: /api/v1/data
    method: POST
    description: Store data
  - path: /api/v1/query
    method: GET
    description: Query data
"""
    
    print("\n1. LLM Data Analysis")
    print("-" * 30)
    
    analyzer = LLMDataAnalyzer()
    print(f"\nLLM Data Analysis:")
    try:
        analysis = analyzer.analyze_data(complex_data)
        print(f"   Type: {analysis.data_type}")
        print(f"   Confidence: {analysis.confidence}")
        
        if hasattr(analysis, 'detected_patterns') and analysis.detected_patterns:
            print(f"   Patterns: {', '.join(analysis.detected_patterns[:3])}")
        
        if hasattr(analysis, 'llm_insights') and analysis.llm_insights:
            print(f"   LLM Insights: {analysis.llm_insights[:100]}...")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n2. LLM-Enhanced Data Storage")
    print("-" * 30)
    
    # E-commerce data
    ecommerce_data = {
        "orders": [
            {
                "order_id": "ORD-001",
                "customer": {
                    "id": 12345,
                    "name": "Alice Johnson",
                    "tier": "premium"
                },
                "items": [
                    {"product": "Wireless Headphones", "price": 199.99, "quantity": 1},
                    {"product": "Phone Case", "price": 29.99, "quantity": 2}
                ],
                "total": 259.97,
                "status": "completed"
            }
        ]
    }
    
    response = llm_oracle_agent.store_data(ecommerce_data, "ecommerce_orders")
    
    if response.success:
        print(f"✅ LLM-enhanced storage successful!")
        print(f"   Table: {response.data['table_name']}")
        print(f"   Analysis Confidence: {response.analysis_confidence}")
        
        if response.detected_patterns:
            print(f"   Patterns: {', '.join(response.detected_patterns)}")
        
        if response.llm_insights:
            print(f"   LLM Insights: {response.llm_insights[:150]}...")


def example_llm_query_engine():
    """Example: LLM query engine capabilities"""
    print("\n" + "=" * 60)
    print("🔄 LLM QUERY ENGINE EXAMPLES")
    print("=" * 60)
    
    # Initialize query engines
    engine = LLMQueryEngine()
    
    # Test queries
    test_queries = [
        "show me all users",
        "find users with admin role",
        "SELECT * FROM user_tables FETCH FIRST 5 ROWS ONLY",
        "get the count of all tables",
        "find similar content to machine learning"
    ]
    
    print("\n1. Query Engine Comparison")
    print("-" * 30)
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        try:
            result = engine.create_intelligent_query(query)
            confidence = result.llm_confidence
            print(f"   confidence: {confidence:.2f}")
        except Exception as e:
            print(f"  Error - {str(e)[:30]}...")


def example_natural_language_queries():
    """Example: Natural language query processing"""
    print("\n" + "=" * 60)
    print("🗣️  NATURAL LANGUAGE QUERY EXAMPLES")
    print("=" * 60)
    
    # First store some sample data
    print("\n1. Storing Sample Data")
    print("-" * 30)
    
    sample_data = {
        "products": [
            {"id": 1, "name": "Laptop", "category": "Electronics", "price": 999.99, "in_stock": True},
            {"id": 2, "name": "Mouse", "category": "Electronics", "price": 29.99, "in_stock": True},
            {"id": 3, "name": "Keyboard", "category": "Electronics", "price": 79.99, "in_stock": False},
            {"id": 4, "name": "Monitor", "category": "Electronics", "price": 299.99, "in_stock": True}
        ]
    }
    
    response = llm_oracle_agent.store_data(sample_data, "products")
    
    if not response.success:
        print(f"❌ Failed to store data: {response.message}")
        return
    
    table_name = response.data['table_name']
    print(f"✅ Data stored in table: {table_name}")
    
    # Test natural language queries
    print("\n2. Natural Language Queries")
    print("-" * 30)
    
    nl_queries = [
        "show me all products",
        "find products that are in stock",
        "get products with price greater than 50",
        "show me electronics products",
        "find products with name containing 'Laptop'",
        "count all products",
        "get the most expensive product"
    ]
    
    for query in nl_queries:
        print(f"\nQuery: '{query}'")
        try:
            result = llm_oracle_agent.create_queries(query, table_name)
            
            if result.success:
                print(f"   ⏱️  Time: {result.metadata['execution_time']:.3f}s")
                
                if 'generated_sql' in result.metadata:
                    print(f"   📝 SQL: {result.metadata['generated_sql']}")
                
                if 'query_explanation' in result.metadata:
                    print(f"   💡 Explanation: {result.metadata['query_explanation']}")
                
                if result.llm_insights:
                    print(f"   🧠 LLM Insights: {result.llm_insights}")
            else:
                print(f"   ❌ Failed: {result.message}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")


def example_system_status():
    """Example: System status and capabilities"""
    print("\n" + "=" * 60)
    print("📊 SYSTEM STATUS & CAPABILITIES")
    print("=" * 60)
    
    print("\n1. Oracle Database 23ai Features")
    print("-" * 30)
    
    try:
        features = llm_oracle_agent.get_oracle_23ai_features()
        if features.success:
            feature_data = features.data
            print(f"✅ Database Version: {feature_data.get('version', 'Unknown')}")
            print(f"✅ JSON Support: {'Yes' if feature_data.get('json_support') else 'No'}")
            print(f"✅ Boolean Support: {'Yes' if feature_data.get('boolean_support') else 'No'}")
            print(f"✅ Vector Search: {'Yes' if feature_data.get('vector_search') else 'No'}")
            print(f"✅ Wide Tables: {'Yes' if feature_data.get('wide_tables') else 'No'}")
        else:
            print(f"❌ Failed to check features: {features.message}")
    except Exception as e:
        print(f"❌ Error checking features: {e}")
    
    print("\n2. LLM Capabilities")
    print("-" * 30)
    
    try:
        # Check Ollama availability
        llm_engine = LLMQueryEngine()
        models = llm_engine.get_available_models()
        
        if models:
            print(f"✅ Ollama Available with models: {models}")
            print(f"✅ Current model: {llm_engine.model_name}")
        else:
            print("⚠️  Ollama not available or no models installed")
            print("   Run: python setup_ollama.py")
    except Exception as e:
        print(f"⚠️  LLM capabilities not available: {e}")
    
    print("\n3. Stored Tables")
    print("-" * 30)
    
    try:
        response = llm_oracle_agent.get_table_info()
        if response.success:
            tables = response.data
            print(f"✅ Found {len(tables)} tables:")
            for table in tables:
                print(f"   • {table['table_name']} ({table['data_type']})")
                print(f"     Created: {table['created_at']}")
                if table.get('oracle_23ai_features'):
                    print(f"     Features: {', '.join(table['oracle_23ai_features'])}")
        else:
            print(f"❌ Failed to get table info: {response.message}")
    except Exception as e:
        print(f"❌ Error getting table info: {e}")


def main():
    """Run all combined examples"""
    print("🗄️  ORACLE DATASTORE AGENT - COMBINED EXAMPLES")
    print("   Comprehensive demonstration of all features")
    print("   " + "=" * 60)
    
    try:
        # Run all examples
        example_basic_functionality()
        example_llm_enhanced_analysis()
        example_llm_query_engine()
        example_natural_language_queries()
        example_system_status()
        
        print("\n" + "=" * 60)
        print("✅ ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\n🎉 Oracle Datastore Agent is working perfectly!")
        print("\n📚 Next Steps:")
        print("   • Start the chatbot: streamlit run app.py")
        print("   • Or use the single entry point: python run.py")
        print("   • Access the UI at: http://localhost:8501")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        print("\n🔧 Troubleshooting:")
        print("   • Make sure Oracle Database 23ai is running")
        print("   • Check database connection settings")
        print("   • For LLM features, run: python setup_ollama.py")
    
    finally:
        # Cleanup
        try:
            llm_oracle_agent.cleanup()
        except:
            pass


if __name__ == "__main__":
    main()
