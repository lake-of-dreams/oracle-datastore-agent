# 🗄️ Oracle Datastore Agent

A powerful, intelligent agent that leverages Oracle Database 23ai features for optimal data storage and generaing **possible** queries for retrieval. The system combines traditional rule-based analysis with cutting-edge LLM capabilities using local Ollama models, providing privacy-first, cost-effective AI-powered data management. See https://bit.ly/3K8Pkil for more details.

## ✨ Key Features

### 🧠 Intelligent Data Analysis
- **Multi-Format Support**: JSON, CSV, TSV, XML, YAML, SQL, Log files, Code, Text, Binary
- **Smart Schema Generation**: Creates optimal database schemas automatically
- **Confidence Scoring**: Provides reliability metrics for analysis results

### 🚀 Oracle Database 23ai Integration
- **JSON Relational Duality**: Seamless JSON and relational data integration
- **AI Vector Search**: Semantic similarity search for text data
- **Boolean Data Type**: Native boolean support for true/false values
- **Wide Tables**: Support for up to 4,096 columns
- **Value LOBs**: Optimized binary data storage for read-and-forget scenarios

### 🦙 Local LLM with Ollama
- **Privacy-First**: All data processing happens locally
- **No API Costs**: No external API charges or rate limits
- **Offline Capable**: Works without internet connection
- **Natural Language Queries**: Convert natural language to SQL
- **Query Optimization**: AI-powered performance improvements

### 💬 Interactive Chatbot Interface
- **Streamlit UI**: Famous conversational web interface
- **Real-time Chat**: Natural language interaction
- **File Upload**: Drag-and-drop support for various formats
- **Data Visualization**: Rich display of results and metadata

## 🚀 Quick Start

```bash
# Clone the repository
git clone <repository-url>
cd oracle-datastore-agent
```

### Run Oracle Database 23ai container
```bash
mkdir -p /opt/oracle/oradata
mkdir -p /opt/oracle/scripts/startup
cp ./database/init/01_init.sql /opt/oracle/scripts/startup/.

podman run --name oracle-datastore-db -p 1521:1521 -p 5500:5500 -e ORACLE_PWD=Oracle123 -e ORACLE_CHARACTERSET=AL32UTF8 -v /opt/oracle/oradata:/opt/oracle/oradata -v /opt/oracle/scripts/startup:/opt/oracle/scripts/startup container-registry.oracle.com/database/free:latest
```

### Run Ollama
```bash
mkdir -p ~/ollamadata

podman run --name oracle-datastore-ollama -v ~/ollamadata:/root/.ollama -p 11434:11434 -e OLLAMA_HOST=0.0.0.0 ollama/ollama:latest
```

### Run the agent and UI

```bash
pip install -r requirements.txt

# Run with default settings (no configuration needed)
python run.py
```

## 🏗️ Architecture

```
oracle-datastore/
├── agent/                    # Main agent orchestrators
│   ├── oracle_datastore_agent.py      # LLM-enhanced agent
├── analyzer/                # Data format analysis
│   ├── llm_data_analyzer.py          # LLM-powered analyzer
├── query/                    # Query engines
│   ├── llm_query_engine.py           # LLM query engine
├── chatbot/                  # Streamlit interface
│   └── chatbot.py
├── database/                 # Oracle connection
│   ├── connection.py
│   └── init/                 # Database initialization
├── examples/                 # Usage examples
│   └── examples.py
├── ingestion/                # Data ingestion
│   └── data_ingestion.py
├── schema/                   # Schema generation
│   └── schema_generator.py
├── run.py                    # Single entry point
├── app.py                    # Streamlit app
└── setup_ollama.py          # Ollama setup
```

## 🔧 Configuration

### Default Settings (No Configuration Required)

The system works out-of-the-box with these defaults:

```python
# Oracle Database
ORACLE_USER=datastore_user
ORACLE_PASSWORD=datastore_pass
ORACLE_HOST=localhost (or oracle-db in Docker)
ORACLE_PORT=1521
ORACLE_SERVICE_NAME=FREEPDB1

# Ollama
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama2
```

### Custom Configuration

Create `.env` file for custom settings:

```env
# Oracle Database
ORACLE_USER=your_username
ORACLE_PASSWORD=your_password
ORACLE_HOST=your_host
ORACLE_PORT=1521
ORACLE_SERVICE_NAME=your_service

# Ollama
OLLAMA_HOST=http://your-ollama-host:11434
OLLAMA_MODEL=your_preferred_model
```

### Access Points

- **Chatbot UI**: http://localhost:8501
- **Oracle Database**: localhost:1521 (pdbadmin/Oracle123)
- **Ollama API**: http://localhost:11434

## 🧪 Testing & Examples

### Run All Examples

```bash
# Run all examples
python run_examples.py
```

### Test Individual Components

```bash
# Test Oracle connection
python -c "from database import db_manager; print(db_manager.check_oracle_23ai_features())"

# Test Ollama connection
python setup_ollama.py

```

## 📈 Performance & Optimization

### Hardware Recommendations

- **CPU**: 8+ cores recommended
- **RAM**: 16GB+ for 7B models, 32GB+ for 13B models
- **Storage**: 50G

### Model Performance

| Model | RAM Usage | GPU VRAM | Speed | Quality | Use Case |
|-------|-----------|----------|-------|---------|----------|
| llama2:7b | 8GB | 4GB | Fast | Good | General purpose |
| llama2:13b | 16GB | 8GB | Medium | Better | High accuracy |
| codellama:7b | 8GB | 4GB | Fast | Excellent | Code analysis |
| mistral:7b | 8GB | 4GB | Fast | Good | Alternative option |


### Health Checks

```bash
# Check Oracle Database
docker exec oracle-datastore-db sqlplus -L pdbadmin/Oracle123@localhost:1521/FREEPDB1 @/dev/null

# Check Ollama
curl -f http://localhost:11434/api/tags

# Check Application
curl -f http://localhost:8501/_stcore/health
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Update documentation
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

- **Documentation**: Check the examples and README files
- **Issues**: Report bugs and feature requests on GitHub
- **Oracle Database 23ai**: Refer to Oracle's official documentation
- **Ollama**: Check Ollama documentation for model management
---

**Built with ❤️ using Oracle Database 23ai, Ollama, and Python**

*The Oracle Datastore Agent represents the future of intelligent data management, combining the power of Oracle Database 23ai with cutting-edge AI capabilities in a privacy-first, cost-effective solution.*
