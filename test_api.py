import requests

key = input("粘贴你的API Key: ")

response = requests.post(
    'https://api.siliconflow.cn/v1/chat/completions',
    headers={
        'Authorization': 'Bearer ' + key.strip(),
        'Content-Type': 'application/json'
    },
    json={
        'model': 'Qwen/Qwen2.5-7B-Instruct',
        'messages': [{'role': 'user', 'content': '你好'}]
    }
)
print(response.status_code)
print(response.text)
