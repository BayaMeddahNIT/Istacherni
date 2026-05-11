from google import genai
import os
from dotenv import load_dotenv
from pathlib import Path

# Load env to get API key
load_dotenv(Path(__file__).resolve().parent / ".env")
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("❌ Error: GEMINI_API_KEY not found in .env")
    exit(1)

print(f"Using API Key: {api_key[:10]}...")

client = genai.Client(api_key=api_key)

print("\n--- Available Models ---")
try:
    for model in client.models.list():
        print(f"  - {model.name} (Supports: {model.supported_actions})")
except Exception as e:
    print(f"❌ Error listing models: {e}")

print("\n--- Testing Gemini 2.0 Flash ---")
try:
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents="Say hello!"
    )
    print(f"✅ Success: {response.text}")
except Exception as e:
    print(f"❌ Error with gemini-2.0-flash: {e}")

print("\n--- Testing Gemini 1.5 Flash ---")
try:
    response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents="Say hello!"
    )
    print(f"✅ Success: {response.text}")
except Exception as e:
    print(f"❌ Error with gemini-1.5-flash: {e}")
