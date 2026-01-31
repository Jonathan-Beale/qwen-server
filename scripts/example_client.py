import requests
import sys
import time

def stream_request(prompt, max_tokens=256):
    url = "http://localhost:8000/v1/completions"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "/models/Qwen-7B-4bit",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "stream": True
    }
    with requests.post(url, json=data, headers=headers, stream=True) as r:
        for line in r.iter_lines():
            if line:
                print(line.decode(), flush=True)

def simple_request(prompt, max_tokens=256):
    url = "http://localhost:8000/v1/completions"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "/models/Qwen-7B-4bit",
        "prompt": prompt,
        "max_tokens": max_tokens
    }
    r = requests.post(url, json=data, headers=headers)
    print(r.json())

if __name__ == "__main__":
    prompt = "def hello_world():\n    "
    print("--- Streaming Example ---")
    stream_request(prompt)
    print("--- Non-Streaming Example ---")
    simple_request(prompt)
