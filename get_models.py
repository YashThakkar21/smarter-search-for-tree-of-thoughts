import os
import requests
from dotenv import load_dotenv

load_dotenv()

# Build the exact URL
base_url = os.environ.get("OPENAI_API_BASE")
url = f"{base_url}/models"
api_key = os.environ.get("OPENAI_API_KEY")

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

print(f"Sending raw request to: {url}...")

try:
    # We add a 10-second timeout so it can't hang forever
    response = requests.get(url, headers=headers, timeout=10)

    print("\n--- Server Response ---")
    print(f"Status Code: {response.status_code}")

    try:
        # Try to print pretty JSON if possible
        print(response.json())
    except:
        # Otherwise print the raw text/HTML error
        print(response.text)

except Exception as e:
    print(f"\nNetwork Error: {e}")