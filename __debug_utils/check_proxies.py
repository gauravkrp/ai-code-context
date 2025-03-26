"""
Script to check for proxies configuration in OpenAI.
"""
import openai
import inspect

# Check if there's a global _proxy module and inspect it
print("Checking for _proxy module:")
proxy_module = getattr(openai._utils, "_proxy", None)
if proxy_module:
    print("Found _proxy module")
    print("Module attributes:")
    for attr in dir(proxy_module):
        if not attr.startswith('_'):  # Skip private attributes
            print(f" - {attr}")
    
    # Check if there's a get_proxies function
    get_proxies = getattr(proxy_module, "get_proxies", None)
    if get_proxies:
        print("\nget_proxies function found.")
        print(f"Signature: {inspect.signature(get_proxies)}")
        try:
            print(f"Source code: {inspect.getsource(get_proxies)}")
        except Exception as e:
            print(f"Could not get source: {e}")
        
        # Check what get_proxies returns
        try:
            proxies = get_proxies()
            print(f"\nget_proxies() returns: {proxies}")
        except Exception as e:
            print(f"Error calling get_proxies(): {e}")
else:
    print("No _proxy module found in openai._utils") 