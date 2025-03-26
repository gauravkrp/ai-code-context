"""
Script to create an OpenAI client with a custom HTTP client.
"""
import os
import httpx
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

print("\nTrying with direct httpx client...")
try:
    # Create a direct httpx client with no proxies
    direct_client = httpx.Client()
    
    # Use it with OpenAI
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        http_client=direct_client
    )
    print("✅ Successfully created OpenAI client!")
    
except Exception as e:
    print(f"❌ Error creating OpenAI client: {e}")
    
    # Get more details about the exception
    import traceback
    print("\nDetailed error information:")
    traceback.print_exc() 