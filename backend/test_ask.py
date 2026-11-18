
import requests

url = "http://127.0.0.1:5000/api/ask"

data = {
    "question": "What are the limitations of traditional RAG systems?"
}

response = requests.post(url, json=data)

print("Status code:", response.status_code)
print(response.json())
