import requests
url = "http://localhost:8000/chat"
payload = {"question": "ما هي شروط عقد البيع؟"}
response = requests.post(url, json=payload)
print(response.json())