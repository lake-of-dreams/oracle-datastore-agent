#!/usr/bin/env python3
"""
Setup script for Ollama integration with Oracle Datastore Agent
"""
import requests
import sys


def check_ollama_connection(host="http://localhost:11434", timeout=5):
    """Check if Ollama is running and accessible"""
    try:
        response = requests.get(f"{host}/api/tags", timeout=timeout)
        if response.status_code == 200:
            models = response.json().get('models', [])
            print(f"✅ Ollama is running at {host}")
            print(f"📋 Available models: {[model['name'] for model in models]}")
            return True
        else:
            print(f"❌ Ollama responded with status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to Ollama at {host}")
        print("   Make sure Ollama is running or start it with: docker-compose up -d")
        return False
    except Exception as e:
        print(f"❌ Error checking Ollama: {e}")
        return False


def pull_recommended_models(host="http://localhost:11434"):
    """Pull recommended models for the Oracle Datastore Agent"""
    recommended_models = [
        "llama2:7b",      # General purpose model
        "codellama:7b",   # Code-specific model
        "mistral:7b",     # Alternative general model
    ]
    
    print("🚀 Pulling recommended models...")
    
    for model in recommended_models:
        print(f"📥 Pulling {model}...")
        try:
            response = requests.post(
                f"{host}/api/pull",
                json={"name": model},
                stream=True,
                timeout=300
            )
            
            if response.status_code == 200:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        print(str(chunk, encoding="utf-8"), end="")
                response.close()
                print(f"✅ Successfully pulled {model}")
            else:
                print(f"❌ Failed to pull {model}: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error pulling {model}: {e}")


def test_model_generation(host="http://localhost:11434", model="llama2"):
    """Test model generation capabilities"""
    print(f"🧪 Testing model {model}...")
    
    try:
        payload = {
            "model": model,
            "prompt": "Hello, I'm testing the Oracle Datastore Agent. Please respond with a brief greeting.",
            "stream": False
        }
        
        response = requests.post(
            f"{host}/api/generate",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            generated_text = result.get('response', '').strip()
            print(f"✅ Model test successful!")
            print(f"📝 Response: {generated_text[:100]}...")
            return True
        else:
            print(f"❌ Model test failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing model: {e}")
        return False


def create_ollama_config():
    """Create configuration file for Ollama integration"""
    config_content = '''# Ollama Configuration for Oracle Datastore Agent
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama2
OLLAMA_TIMEOUT=30
OLLAMA_MAX_TOKENS=1000
OLLAMA_TEMPERATURE=0.1
'''
    
    config_file = "ollama_config.env"
    try:
        with open(config_file, 'w') as f:
            f.write(config_content)
        print(f"✅ Created configuration file: {config_file}")
        return True
    except Exception as e:
        print(f"❌ Error creating config file: {e}")
        return False


def main():
    """Main setup function"""
    print("🔧 Oracle Datastore Agent - Ollama Setup")
    print("=" * 50)
    
    # Check Ollama connection
    if not check_ollama_connection():
        print("\n💡 To start Ollama with Docker:")
        print("   docker-compose up -d")
        print("\n💡 To install Ollama locally:")
        print("   curl -fsSL https://ollama.ai/install.sh | sh")
        print("   ollama serve")
        return False
    
    # Pull recommended models
    print("\n" + "=" * 50)
    pull_recommended_models()
    
    # Test model generation
    print("\n" + "=" * 50)
    if test_model_generation():
        print("✅ Ollama setup completed successfully!")
    else:
        print("⚠️  Ollama is running but model generation failed")
        print("   Try pulling a model manually: ollama pull llama2")
    
    # Create configuration
    print("\n" + "=" * 50)
    create_ollama_config()
    
    print("\n🎉 Setup complete! You can now use the LLM-enhanced features.")
    print("\n📚 Next steps:")
    print("1. Run examples: python examples/examples.py")
    print("2. Start the chatbot: python run.py")
    print("3. Check available models: ollama list")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
