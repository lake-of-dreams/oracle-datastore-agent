"""
Oracle Datastore Chatbot
Interactive chatbot interface for the Oracle Datastore Agent
"""
import logging
import streamlit as st
from typing import Dict, Any
import json
from datetime import datetime

from agent import llm_oracle_agent
from common.setup import setup_logging


setup_logging()
logger = logging.getLogger(__name__)


class OracleDatastoreChatbot:
    """Interactive chatbot for Oracle Datastore Agent"""
    
    def __init__(self):
        self.session_history = []
        self.initialize_session()
    
    def initialize_session(self):
        """Initialize Streamlit session state"""
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        if 'tables_created' not in st.session_state:
            st.session_state.tables_created = []
        if 'oracle_features' not in st.session_state:
            st.session_state.oracle_features = None
    
    def run_chatbot(self):
        """Run the main chatbot interface"""
        st.set_page_config(
            page_title="Oracle Datastore Agent",
            page_icon="🗄️",
            layout="wide"
        )
        
        st.title("🗄️ Oracle Datastore Agent")
        st.markdown("**Intelligent Data Storage and Retrieval with Oracle Database 23ai**")
        
        # Sidebar for Oracle features and table info
        self.render_sidebar()
        
        # Main chat interface
        self.render_chat_interface()
        
        # Data upload section
        self.render_data_upload()
    
    def render_sidebar(self):
        """Render sidebar with Oracle features and table information"""
        with st.sidebar:
            st.header("🔧 Oracle Database 23ai Features")
            
            if st.button("Check Oracle 23ai Features"):
                with st.spinner("Checking Oracle Database 23ai features..."):
                    response = llm_oracle_agent.get_oracle_23ai_features()
                    if response.success:
                        st.session_state.oracle_features = response.data
                        st.success("Features checked successfully!")
                    else:
                        st.error(f"Error: {response.message}")
            
            if st.session_state.oracle_features:
                st.subheader("Available Features:")
                features = st.session_state.oracle_features
                
                if features.get('json_support'):
                    st.success("✅ JSON Support")
                if features.get('boolean_support'):
                    st.success("✅ Boolean Data Type")
                if features.get('vector_search'):
                    st.success("✅ AI Vector Search")
                if features.get('wide_tables'):
                    st.success("✅ Wide Tables (4096 columns)")
                
                if 'version' in features:
                    st.info(f"Database Version: {features['version']}")

            if st.button("Show Tables"):
                st.header("📊 Created Tables")
                if st.session_state.tables_created:
                    for table in st.session_state.tables_created:
                        with st.expander(f"📋 {table['name']}"):
                            st.write(f"**Type:** {table['type']}")
                            st.write(f"**Created:** {table['created']}")
                            if table.get('features'):
                                st.write("**Oracle 23ai Features:**")
                                for feature in table['features']:
                                    st.write(f"• {feature}")
                else:
                    st.info("No tables created yet")
    
    def render_chat_interface(self):
        """Render the main chat interface"""
        st.header("💬 Chat with Oracle Datastore Agent")
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Display additional data if available
                if message.get("data"):
                    self.display_message_data(message["data"])
        
        # Chat input
        if prompt := st.chat_input("Ask me about your data or request operations..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Process the message
            with st.chat_message("assistant"):
                with st.spinner("Processing your request..."):
                    response = self.process_user_message(prompt)
                    
                    # Display response
                    st.markdown(response["content"])
                    
                    # Display additional data
                    if response.get("data"):
                        self.display_message_data(response["data"])
                    
                    # Add to session
                    st.session_state.messages.append(response)
    
    def render_data_upload(self):
        """Render data upload section"""
        st.header("📤 Upload Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Upload File")
            uploaded_file = st.file_uploader(
                "Choose a file",
                type=['json', 'csv', 'txt', 'pdf', 'png', 'jpg', 'jpeg', 'docx'],
                help="Supported formats: JSON, CSV, TXT, PDF, Images, Word documents"
            )
            
            if uploaded_file:
                if st.button("Store File Data"):
                    self.handle_file_upload(uploaded_file)
        
        with col2:
            st.subheader("Paste Data")
            data_type = st.selectbox(
                "Data Type",
                ["JSON", "CSV", "Text", "Auto-detect"]
            )
            
            data_input = st.text_area(
                "Paste your data here",
                height=200,
                help="Paste JSON, CSV, or text data"
            )
            
            if data_input and st.button("Store Pasted Data"):
                self.handle_text_upload(data_input, data_type)
    
    def process_user_message(self, message: str) -> Dict[str, Any]:
        """Process user message and return response"""
        message_lower = message.lower()
        
        try:
            # Handle different types of queries
            if any(keyword in message_lower for keyword in ['store', 'save', 'upload', 'insert']):
                return self.handle_storage_request(message)
            
            elif any(keyword in message_lower for keyword in ['query', 'search', 'find', 'get']):
                return self.handle_query_request(message)
            
            elif any(keyword in message_lower for keyword in ['table', 'tables', 'list', 'show']):
                return self.handle_table_info_request(message)
            
            elif any(keyword in message_lower for keyword in ['analyze', 'what is', 'detect']):
                return self.handle_analysis_request(message)
            
            elif any(keyword in message_lower for keyword in ['help', 'how', 'what can']):
                return self.handle_help_request()
            
            else:
                return self.handle_general_query(message)
                
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return {
                "role": "assistant",
                "content": f"Sorry, I encountered an error: {str(e)}"
            }
    
    def handle_storage_request(self, message: str) -> Dict[str, Any]:
        """Handle data storage requests"""
        return {
            "role": "assistant",
            "content": "To store data, please use the upload section below. You can upload files or paste data directly. I'll automatically analyze the format and create the optimal storage schema using Oracle Database 23ai features."
        }
    
    def handle_query_request(self, message: str) -> Dict[str, Any]:
        """Handle query requests"""
        try:
            response = llm_oracle_agent.create_queries(message)
            
            if response.success:
                content = f"✅ Query generated successfully!\n\n"
                content += f"**OriginaL Query:** {response.metadata.get('original_query', 'Unknown')}\n\n"
                content += f"**Generated SQL:** {response.metadata.get('generated_sql', 'Unknown')}\n\n"
                content += f"**Execution Time:** {response.metadata.get('execution_time', 0):.3f} seconds\n\n"
                content += f"**LLM Confidence:** {response.metadata.get('llm_confidence', 0):.3f}\n\n"
                content += f"**Query Explaination:** {response.metadata.get('query_explanation', 'Unknown')}\n\n"
                content += f"**Optimization Suggestions:** {response.metadata.get('optimization_suggestions', 'Unknown')}\n\n"
                content += f"**Alternative Queries:** {response.metadata.get('alternative_queries', 'Unknown')}\n\n"
                
                if response.oracle_23ai_features_used:
                    content += "**Oracle 23ai Features Used:**\n"
                    for feature in response.oracle_23ai_features_used:
                        content += f"• {feature}\n"
                    content += "\n"
                
                return {
                    "role": "assistant",
                    "content": content,
                    "data": response.data
                }
            else:
                return {
                    "role": "assistant",
                    "content": f"❌ Query failed: {response.message}"
                }
                
        except Exception as e:
            return {
                "role": "assistant",
                "content": f"❌ Error executing query: {str(e)}"
            }
    
    def handle_table_info_request(self, message: str) -> Dict[str, Any]:
        """Handle table information requests"""
        try:
            response = llm_oracle_agent.get_table_info()
            
            if response.success:
                content = f"📊 **Tables in Database:**\n\n"
                
                if response.data:
                    for table in response.data:
                        content += f"**{table['table_name']}**\n"
                        content += f"• Type: {table['data_type']}\n"
                        content += f"• Created: {table['created_at']}\n"
                        if table.get('oracle_23ai_features'):
                            content += f"• Features: {', '.join(table['oracle_23ai_features'])}\n"
                        content += "\n"
                else:
                    content += "No tables found. Upload some data to get started!"
                
                return {
                    "role": "assistant",
                    "content": content
                }
            else:
                return {
                    "role": "assistant",
                    "content": f"❌ Error getting table info: {response.message}"
                }
                
        except Exception as e:
            return {
                "role": "assistant",
                "content": f"❌ Error: {str(e)}"
            }
    
    def handle_analysis_request(self, message: str) -> Dict[str, Any]:
        """Handle data analysis requests"""
        return {
            "role": "assistant",
            "content": "To analyze data format, please paste your data in the upload section below. I'll analyze it and show you the optimal storage format and Oracle Database 23ai features that would be used."
        }
    
    def handle_help_request(self) -> Dict[str, Any]:
        """Handle help requests"""
        content = """
## 🗄️ Oracle Datastore Agent Help

I'm an intelligent agent that helps you store and query data using Oracle Database 23ai features.

### What I can do:

**📤 Store Data:**
- Upload files (JSON, CSV, TXT, PDF, Images, etc.)
- Paste data directly
- Automatically detect format and create optimal schema

**🔍 Query Data:**
- SQL queries: `SELECT * FROM my_table`
- JSON queries: `user.name = "John"`
- Vector similarity: `find similar content to "machine learning"`
- Full-text search: `search for "artificial intelligence"`

**📊 Get Information:**
- List all tables: `show tables`
- Table details: `table info for my_table`
- Oracle 23ai features: Check sidebar

**🧠 Oracle Database 23ai Features I Use:**
- **JSON Relational Duality** for JSON documents
- **AI Vector Search** for text similarity
- **Boolean Data Type** for true/false values
- **Wide Tables** (up to 4096 columns)
- **Value LOBs** for binary data

### Example Commands:
- `store this JSON data`
- `find all users with status = "active"`
- `search for documents about AI`
- `show me all tables`
- `what Oracle 23ai features are available?`

Just ask me anything about your data!
        """
        
        return {
            "role": "assistant",
            "content": content
        }
    
    def handle_general_query(self, message: str) -> Dict[str, Any]:
        """Handle general queries"""
        return {
            "role": "assistant",
            "content": "I'm here to help you with data storage and retrieval using Oracle Database 23ai. You can upload data, query existing tables, or ask about Oracle 23ai features. What would you like to do?"
        }
    
    def handle_file_upload(self, uploaded_file):
        """Handle file upload"""
        try:
            # Read file content
            if uploaded_file.type.startswith('text/'):
                content = uploaded_file.read().decode('utf-8')
            else:
                content = uploaded_file.read()
            
            # Store the data
            with st.spinner(f"Storing {uploaded_file.name}..."):
                response = llm_oracle_agent.store_data(content, uploaded_file.name.replace('.', '_'))
                
                if response.success:
                    st.success(f"✅ File stored successfully!")
                    st.write(f"**Table:** {response.data['table_name']}")
                    st.write(f"**Type:** {response.data['data_type']}")
                    st.write(f"**Rows:** {response.data['rows_inserted']}")
                    st.write(f"**LLM Insights:** {response.llm_insights}")
                    
                    # Add to session tables
                    st.session_state.tables_created.append({
                        "name": response.data['table_name'],
                        "type": response.data['data_type'],
                        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "features": response.oracle_23ai_features_used
                    })
                    
                    # Add message to chat
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"✅ Successfully stored {uploaded_file.name} in table '{response.data['table_name']}' using Oracle Database 23ai features!"
                    })
                    
                else:
                    st.error(f"❌ Failed to store file: {response.message}")
                    
        except Exception as e:
            st.error(f"❌ Error uploading file: {str(e)}")
    
    def handle_text_upload(self, data_input: str, data_type: str):
        """Handle text data upload"""
        try:
            # Parse data based on type
            if data_type == "JSON":
                try:
                    data = json.loads(data_input)
                except json.JSONDecodeError:
                    st.error("Invalid JSON format")
                    return
            elif data_type == "CSV":
                data = data_input
            else:
                data = data_input
            
            # Store the data
            with st.spinner("Storing data..."):
                response = llm_oracle_agent.store_data(data)
                
                if response.success:
                    st.success("✅ Data stored successfully!")
                    st.write(f"**Table:** {response.data['table_name']}")
                    st.write(f"**Type:** {response.data['data_type']}")
                    st.write(f"**Rows:** {response.data['rows_inserted']}")
                    st.write(f"**LLM Insights:** {response.llm_insights}")
                    
                    # Add to session tables
                    st.session_state.tables_created.append({
                        "name": response.data['table_name'],
                        "type": response.data['data_type'],
                        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "features": response.oracle_23ai_features_used
                    })
                    
                    # Add message to chat
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"✅ Successfully stored {data_type} data in table '{response.data['table_name']}' using Oracle Database 23ai features!"
                    })
                    
                else:
                    st.error(f"❌ Failed to store data: {response.message}")
                    
        except Exception as e:
            st.error(f"❌ Error storing data: {str(e)}")
    
    def display_message_data(self, data: Any):
        """Display additional data in chat messages"""
        if isinstance(data, list) and len(data) > 0:
            # Show first few rows
            display_data = data[:5]  # Show first 5 rows
            st.json(display_data)
            
            if len(data) > 5:
                st.info(f"Showing first 5 of {len(data)} results")
        elif isinstance(data, dict):
            st.json(data)
        else:
            st.write(str(data))


def main():
    """Main function to run the chatbot"""
    chatbot = OracleDatastoreChatbot()
    chatbot.run_chatbot()


if __name__ == "__main__":
    main()

