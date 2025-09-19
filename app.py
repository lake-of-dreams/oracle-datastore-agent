"""
Main application entry point for Oracle Datastore Agent
"""
from chatbot import OracleDatastoreChatbot


def main():
    """Main application function"""
    chatbot = OracleDatastoreChatbot()
    chatbot.run_chatbot()


if __name__ == "__main__":
    main()

