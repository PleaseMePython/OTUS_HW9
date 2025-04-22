import requests

url = "http://127.0.0.1:8000/api/predict/"

response = requests.get(
    url, params={"pregnancies": 2, "glucose": 40.0, "bmi": 15.5, "age": 32.0}
)
print(
    "Предсказание: диабет",
    "не выявлен" if response.json()["diabetes"] is False else "выявлен",
)
