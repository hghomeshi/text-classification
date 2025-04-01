import requests

response = requests.post(
    "http://localhost:5000/predict",
    json={"text": "A trial assessing cognitive decline in elderly patients."}
)
print(response.json())