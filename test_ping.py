import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
gkey = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=gkey)

print("Starting ping test...")
for m in ['gemini-2.5-flash', 'gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-pro']:
    try:
        model = genai.GenerativeModel(m)
        resp = model.generate_content("ping")
        print(f"Success with {m}: {resp.text}")
    except Exception as e:
        print(f"Error with {m}: {repr(e)}")

