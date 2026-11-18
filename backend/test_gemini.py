import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("Gemini API key not found in .env")

client = genai.Client(api_key=api_key)

try:
    print("Testing Gemini 3.5 Flash Lite...")

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents="Explain Agentic RAG in three simple sentences."
    )

    print("\nSUCCESS! Gemini is working.")
    print("\nGenerated Answer:")
    print(response.text)

except Exception as error:
    print("\nGemini Error:")
    print(error)

finally:
    client.close()