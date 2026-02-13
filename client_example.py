import requests

# Upload a document
with open('instrumentation.docx', 'rb') as f:
    response = requests.post(
        'https://ideal-waddle-4jvj5r4xvwq5h74x-8000.app.github.dev/upload',
        files={'file': f}
    )
    print(response.json())

# Ask a question
response = requests.post(
    'https://ideal-waddle-4jvj5r4xvwq5h74x-8000.app.github.dev/query',
    json={
        'question': 'What is the methodology used in this research?',
        'num_sources': 3
    }
)
print(response.json()['answer'])