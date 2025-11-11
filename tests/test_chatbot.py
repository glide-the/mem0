import os
from typing import List, Dict
from mem0 import Memory
from datetime import datetime

from openai import OpenAI

# Set up environment variables
os.environ["OPENAI_API_KEY"] = "sk-5fbc1f3bae5e48b495c422aaae0f7d13"  # needed for embedding model
os.environ["OPENAI_BASE_URL"] = "https://api.deepseek.com/v1"  # needed for embedding model
 
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:8687")
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "mem0graph")

 
class SupportChatbot:
    def __init__(self):
        # Initialize Mem0 with Anthropic's Claude
        self.config = {
            "llm": {
                "provider": "openai",
                "config": {
                    "model": "deepseek-chat",
                    "temperature": 0.1,
                    "max_tokens": 2000,
                },
            },
            
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": "qwen3-embedding",
                    "api_key": "gpustack_dce0bd7c9fe85b3c_ce9df245a8bd278bc5ec07656ef0bd75",
                    "openai_base_url": "http://10.0.0.49:9090/v1",
                    "embedding_dims": 4096,
                }
            },
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "host": "localhost",
                    "port": 6333,
                    "collection_name": "memories",
                    "embedding_model_dims": 4096
                }
            },
                
            "graph_store": {
                "provider": "neo4j",
                "config": {"url": NEO4J_URI, "username": NEO4J_USERNAME, "password": NEO4J_PASSWORD, "database": "neo4j" },
            },
        }

        self.client = OpenAI(
            api_key=os.environ.get('OPENAI_API_KEY'),
            base_url=os.environ.get('OPENAI_BASE_URL')
        ) 
        self.memory = Memory.from_config(self.config)

        # Define support context
        self.system_context = """
        You are a helpful customer support agent. Use the following guidelines:
        - Be polite and professional
        - Show empathy for customer issues
        - Reference past interactions when relevant
        - Maintain consistent information across conversations
        - If you're unsure about something, ask for clarification
        - Keep track of open issues and follow-ups
        """

    def store_customer_interaction(self, user_id: str, message: str, response: str, metadata: Dict = None):
        """Store customer interaction in memory."""
        if metadata is None:
            metadata = {}

        # Add timestamp to metadata
        metadata["timestamp"] = datetime.now().isoformat()

        # Format conversation for storage
        conversation = [{"role": "user", "content": message}, {"role": "assistant", "content": response}]

        # Store in Mem0
        self.memory.add(conversation, user_id=user_id, metadata=metadata)

    def get_relevant_history(self, user_id: str, query: str) -> List[Dict]:
        """Retrieve relevant past interactions."""
        return self.memory.search(
            query=query,
            user_id=user_id,
            limit=5,  # Adjust based on needs
        )

    def handle_customer_query(self, user_id: str, query: str) -> str:
        """Process customer query with context from past interactions."""

        # Get relevant past interactions
        relevant_history = self.get_relevant_history(user_id, query)

        # Build context from relevant history
        context = "Previous relevant interactions:\n"
        for memory in relevant_history["results"]:
            try:
                context += f"Customer: {memory['memory']}\n"
                context += f"Support: {memory['memory']}\n"
                context += "---\n"
            except Exception as e:
                pass
    
        # Prepare prompt with context and current query
        prompt = f"""
        {self.system_context}

        {context}

        Current customer query: {query}

        Provide a helpful response that takes into account any relevant past interactions.
        """
        print(prompt)
        # Generate response using Claude
        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
            temperature=0.1,
        )

        # Store interaction
        self.store_customer_interaction(
            user_id=user_id, message=query, response=response.choices[0].message.content, metadata={"type": "support_query"}
        ) 
        return response.choices[0].message.content
    
    
chatbot = SupportChatbot()
user_id = "customer_bot"
print("Welcome to Customer Support! Type 'exit' to end the conversation.")

while True:
    # Get user input
    query = input()
    print("Customer:", query)

    # Check if user wants to exit
    if query.lower() == "exit":
        print("Thank you for using our support service. Goodbye!")
        break

    # Handle the query and print the response
    response = chatbot.handle_customer_query(user_id, query)
    print("Support:", response, "\n\n")