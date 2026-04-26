import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
gkey = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=gkey)

out = []
try:
    for m in genai.list_models():
        out.append({'name': m.name, 'methods': m.supported_generation_methods})
except Exception as e:
    out.append({'error': str(e)})

with open('model_list.json', 'w') as f:
    json.dump(out, f, indent=2)
