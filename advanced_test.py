"""
Advanced test script for OpenAI client with explicit configurations.
"""
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Clear any potential HTTP_PROXY or HTTPS_PROXY environment variables
for key in list(os.environ.keys()):
    if 'PROXY' in key.upper() or 'proxy' in key:
        print(f"Removing environment variable: {key}")
        del os.environ[key]

# Add debugging for OpenAI module
os.environ['OPENAI_LOG'] = 'debug'

# Now import the OpenAI client
print("Importing OpenAI...")
from openai import OpenAI

# Try creating the client directly
print("\nAttempting to create OpenAI client with explicit parameters (no proxies)...")
try:
    # Create client with explicit parameters, avoiding any global configs
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        max_retries=0,
        timeout=30.0,
    )
    print("✅ Successfully created OpenAI client!")
    
    # Optionally test a simple request
    # print("\nTesting simple completion request...")
    # response = client.chat.completions.create(
    #     model="o3-mini",
    #     messages=[{"role": "user", "content": "Hello!"}],
    #     max_tokens=10
    # )
    # print(f"Response: {response.choices[0].message.content}")
    
except Exception as e:
    print(f"❌ Error creating OpenAI client: {e}")
    
    # Get more details about the exception
    import traceback
    print("\nDetailed error information:")
    traceback.print_exc()
    
    # Check if monkey patching might be involved
    print("\nChecking for potential monkey patching...")
    if "asyncio" in str(e) or "async" in str(e):
        print("This might be related to asyncio. Check if there's event loop interference.")
    if "httpx" in str(e):
        print("This might be related to httpx configuration. Check for global httpx settings.")
    if "proxy" in str(e).lower() or "proxies" in str(e).lower():
        print("This is definitely a proxy-related issue. Check for global proxy configurations in your Python environment.") 