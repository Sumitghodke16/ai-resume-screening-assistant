from langchain_ollama import ChatOllama

# Load local Llama 3.2 3B model
llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)

# Test the model
response = llm.invoke(
    "Explain what a resume screening system does in two simple sentences."
)

print("\nLLM RESPONSE:\n")
print(response.content)