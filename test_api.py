from dotenv import load_dotenv
import os

from langchain_openai import ChatOpenAI


# Load variables from .env
load_dotenv()

# Check if API key exists
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    print("❌ OPENAI_API_KEY not found in .env")
    exit()

print("✅ API key found")

# Create LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# Send a simple test request
response = llm.invoke("Say only: API connection successful")

print("🤖 Response:")
print(response.content)