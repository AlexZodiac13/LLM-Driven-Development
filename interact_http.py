import httpx
import json

def interact_with_vllm_http():
    url = "http://localhost:8000/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    
    data = {
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "messages": [
            {"role": "user", "content": "What is the capital of Germany?"}
        ],
        "max_tokens": 50
    }
    
    print(f"Sending request to {url}...")
    try:
        response = httpx.post(url, headers=headers, json=data, timeout=30.0)
        response.raise_for_status()
        result = response.json()
        print("Response received:")
        print(json.dumps(result, indent=2))
        
        content = result['choices'][0]['message']['content']
        print(f"\nAnswer: {content}")
        
    except httpx.HTTPError as e:
        print(f"HTTP Error: {e}")
        if e.response is not None:
            print(f"Error details: {e.response.text}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    interact_with_vllm_http()
