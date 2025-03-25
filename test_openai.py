"""
Simple test script to verify OpenAI client functionality.
"""
import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_openai_client():
    """Test if the OpenAI client can be initialized properly."""
    try:
        # Initialize with the API key from environment variables
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        print("✅ OpenAI client initialized successfully!")
        
        # Optional: Test a simple API call
        # Uncomment the following lines to test an actual API call
        # response = client.chat.completions.create(
        #     model="o3-mini",
        #     messages=[{"role": "user", "content": "Hello!"}],
        #     max_tokens=10
        # )
        # print("✅ API call successful!")
        # print(f"Response: {response.choices[0].message.content}")
        
        return True
    except Exception as e:
        print(f"❌ Error initializing OpenAI client: {e}")
        return False

if __name__ == "__main__":
    test_openai_client() 