"""
Script to inspect the OpenAI client class.
"""
import inspect
from openai import OpenAI

# Inspect the OpenAI class
print("OpenAI client __init__ method signature:")
print(inspect.signature(OpenAI.__init__))

print("\nOpenAI client __init__ method source:")
try:
    print(inspect.getsource(OpenAI.__init__))
except (TypeError, OSError) as e:
    print(f"Could not get source: {e}")

# Check for any monkey patching
print("\nOpenAI module attributes:")
import openai
for attr in dir(openai):
    if not attr.startswith('_'):  # Skip private attributes
        print(f" - {attr}")

# Check for environment hooks
print("\nChecking for environment hooks:")
import sys
for module_name, module in sys.modules.items():
    if 'openai' in module_name:
        print(f" - {module_name}") 