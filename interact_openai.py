from openai import OpenAI
import os

def interact_with_vllm_openai():
    # vLLM exposes an OpenAI-compatible API
    # base_url should point to the vLLM server
    # api_key can be anything for vLLM usually, but required by the client
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="EMPTY"
    )

    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    
    print(f"Sending request to vLLM model '{model_name}' via OpenAI SDK...")
    
    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": "What is the capital of France?"}
            ],
            max_tokens=50
        )
        
        print("Response received:")
        print(completion)
        
        answer = completion.choices[0].message.content
        print(f"\nAnswer: {answer}")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    interact_with_vllm_openai()
