import os
import io
import sys
import json
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
gkey = os.getenv('GEMINI_API_KEY')
print("API KEY length:", len(gkey) if gkey else 0)
genai.configure(api_key=gkey)

try:
    for m in genai.list_models():
        print(m.name, m.supported_generation_methods)
except Exception as e:
    print("Error listing models:", e)
