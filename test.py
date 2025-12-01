from mistralai import Mistral
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("MISTRAL_API_KEY")
client = Mistral(api_key=API_KEY)

response = client.chat.complete(
    model="mistral-small-latest",
    messages=[{"role": "user", "content": "Hello from Mistral!"}]
)

print("Mistral Response:", response.choices[0].message.content)
