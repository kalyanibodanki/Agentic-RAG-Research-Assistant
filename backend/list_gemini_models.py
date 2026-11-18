import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("Gemini API key not found in .env")

client = genai.Client(api_key=api_key)

print("Checking available Gemini models...\n")

try:
    for model in client.models.list():
        name = model.name or ""
        actions = model.supported_actions or []

        if "generateContent" in actions:
            print("Model:", name)
            print("Supported actions:", actions)
            print("-------------------------")

except Exception as error:
    print("Could not list models:")
    print(error)

finally:
    client.close()