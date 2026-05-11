import os
from dotenv import load_dotenv
from google import genai

load_dotenv('.env')
_API_KEY = os.getenv("GEMINI_API_KEY")

if not _API_KEY:
    print("No API key found in .env")
    exit(1)

client = genai.Client(api_key=_API_KEY)

print("Listing available models:")
try:
    for model in client.models.list():
        print(f"- {model.name}")
except Exception as e:
    print(f"Error listing models: {e}")
