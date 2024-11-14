from openai import OpenAI


def call_chagpt(model_name: str):
    api_key = ""
    client = OpenAI(api_key)
    completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "{{YOUR PROMPT}}"},
        ]
    )