#!/usr/bin/env python3
"""
Oracle Datastore Agent - Single Entry Point
Starts the complete system with default settings
"""
import os

from common.setup import setup_logging
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys
import time
import logging
import subprocess
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


from config import settings
from database import db_manager
from agent import llm_oracle_agent


setup_logging()
logger = logging.getLogger(__name__)


def check_dependencies():
    """Check if all dependencies are available"""
    logger = logging.getLogger(__name__)
    
    # Check Oracle Database connection
    try:
        features = db_manager.check_oracle_23ai_features()
        logger.info("✅ Oracle Database connection successful")
        logger.info(f"Database version: {features.get('version', 'Unknown')}")
        return True
    except Exception as e:
        logger.error(f"❌ Oracle Database connection failed: {e}")
        return False


def wait_for_services():
    """Wait for external services to be ready"""
    logger = logging.getLogger(__name__)
    
    # Wait for Oracle Database
    logger.info("⏳ Waiting for Oracle Database to be ready...")
    max_retries = 30
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            features = db_manager.check_oracle_23ai_features()
            logger.info("✅ Oracle Database is ready!")
            break
        except Exception as e:
            retry_count += 1
            logger.info(f"⏳ Waiting for Oracle Database... ({retry_count}/{max_retries})")
            time.sleep(10)
    else:
        logger.error("❌ Oracle Database failed to start within timeout")
        return False
    
    # Wait for Ollama (optional)
    logger.info("⏳ Checking Ollama availability...")
    try:
        models = llm_oracle_agent.query_engine.get_available_models()
        if models:
            logger.info(f"✅ Ollama is ready with models: {models}")
        else:
            logger.warning("⚠️  Ollama is running but no models available")
    except Exception as e:
        logger.warning(f"⚠️  Ollama not available: {e}")
    
    return True


def start_streamlit_app():
    """Start the Streamlit chatbot application"""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("🚀 Starting Oracle Datastore Agent Chatbot...")
        logger.info("🌐 Application will be available at: http://localhost:8501")
        
        # Set Streamlit configuration
        os.environ['STREAMLIT_SERVER_PORT'] = '8501'
        os.environ['STREAMLIT_SERVER_ADDRESS'] = '0.0.0.0'
        os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
        os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
        
        # Start Streamlit
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 'app.py',
            '--server.port=8501',
            '--server.address=0.0.0.0',
            '--server.headless=true',
            '--browser.gatherUsageStats=false'
        ], check=True)
        
    except KeyboardInterrupt:
        logger.info("👋 Shutting down Oracle Datastore Agent...")
    except Exception as e:
        logger.error(f"❌ Error starting Streamlit app: {e}")
        raise


def print_startup_info():
    """Print startup information"""
    print("=" * 60)
    print("🗄️  Oracle Datastore Agent")
    print("   Intelligent Data Storage with Oracle Database 23ai")
    print("=" * 60)
    print()
    print("🔧 Configuration:")
    print(f"   Oracle Database: {settings.oracle_host}:{settings.oracle_port}")
    print(f"   Service Name: {settings.oracle_service_name or 'FREEPDB1'}")
    print(f"   User: {settings.oracle_user}")
    print(f"   Ollama Host: {getattr(settings, 'ollama_host', 'http://localhost:11434')}")
    print(f"   Ollama Model: {getattr(settings, 'ollama_model', 'llama2')}")
    print()
    print("🌐 Access Points:")
    print("   Chatbot UI: http://localhost:8501")
    print("   Oracle DB: localhost:1521")
    print("   Ollama API: http://localhost:11434")
    print()
    print("📚 Features:")
    print("   ✅ JSON Relational Duality")
    print("   ✅ AI Vector Search")
    print("   ✅ Boolean Data Type")
    print("   ✅ Wide Tables (4096 columns)")
    print("   ✅ Value LOBs")
    print("   ✅ Local LLM with Ollama")
    print("   ✅ Natural Language Queries")
    print("   ✅ Intelligent Data Analysis")
    print()
    print("=" * 60)


def main():
    """Main entry point"""
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Print startup information
        print_startup_info()
        
        # Wait for services to be ready
        if not wait_for_services():
            logger.error("❌ Services not ready, exiting...")
            sys.exit(1)

        
        # Check dependencies
        if not check_dependencies():
            logger.error("❌ Dependencies check failed, exiting...")
            sys.exit(1)
        
        # Start the application
        start_streamlit_app()
        
    except KeyboardInterrupt:
        logger.info("👋 Shutdown requested by user")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        sys.exit(1)
    finally:
        # Cleanup
        try:
            llm_oracle_agent.cleanup()
            logger.info("✅ Cleanup completed")
        except Exception as e:
            logger.warning(f"⚠️  Cleanup warning: {e}")


if __name__ == "__main__":
    main()
