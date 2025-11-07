import os
import google.generativeai as genai
import streamlit as st

# Configure the Gemini API
# Read API key from multiple sources for compatibility
API_KEY = os.getenv("GEMINI_API_KEY")
try:
    if API_KEY:
        genai.configure(api_key=API_KEY)
except Exception as e:
    # We print the error for the developer log, but show a user-friendly error in the app
    print(f"CRITICAL: Failed to configure Gemini API: {e}")

# System prompt to define the AI's role
SYSTEM_PROMPT = """
You are an expert Python, Streamlit, and Data Science debugging assistant. 
You are embedded in a Streamlit dashboard.
A user has encountered an error and is providing you with the error message and the code that caused it.
Your task is to:
1.  Briefly and clearly explain what the error means in the context of their code.
2.  Provide the corrected, runnable Python code snippet.
3.  Explain *why* the fix works.
Format your response clearly using Markdown (e.g., headings, bullet points, and code blocks).
"""

def get_ai_explanation_stream(error_message, code_snippet):
    """
    Calls the Gemini API to get an explanation for an error and streams the response.
    """
    try:
        # Check if API was configured (in case it failed on startup)
        if not API_KEY:
            yield "**Error: Gemini API key not configured.**\n\n"
            yield "Please set `GEMINI_API_KEY` in `.env` (or Streamlit secrets) and restart the app."
            return

        requested_model = (
            os.getenv('GEMINI_MODEL')
            or 'gemini-1.5-flash'
        )
        fallback_models = [
            requested_model,
            'gemini-1.5-flash',
            'gemini-1.5-pro',
            'gemini-pro'
        ]
        fallback_models = [m for m in fallback_models if m]

        last_err = None
        model = None
        for m in fallback_models:
            try:
                model = genai.GenerativeModel(m, system_instruction=SYSTEM_PROMPT)
                # validate availability early
                _ = model.generate_content("ping")
                break
            except Exception as _e:
                last_err = _e
                model = None
                continue

        if model is None:
            # Final fallback: discover available models from API and pick a supported one
            try:
                candidates = []
                for m in genai.list_models():
                    methods = getattr(m, 'supported_generation_methods', []) or []
                    if 'generateContent' in methods:
                        candidates.append(getattr(m, 'name', ''))
                preferred_order = [
                    'gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-1.5-flash-8b',
                    'gemini-pro'
                ]
                pick = None
                for pref in preferred_order:
                    pick = next((c for c in candidates if pref in c), None)
                    if pick:
                        break
                if not pick and candidates:
                    pick = candidates[0]
                if pick:
                    model = genai.GenerativeModel(pick, system_instruction=SYSTEM_PROMPT)
                    _ = model.generate_content('ping')
                else:
                    raise last_err or RuntimeError('No compatible Gemini model available. Try setting GEMINI_MODEL to a supported id (e.g., gemini-1.5-flash).')
            except Exception as _e2:
                raise last_err or _e2
        
        prompt = f"""
        Here is the error my application produced:
        ---
        {error_message}
        ---
        
        Here is the code snippet that caused it:
        ---
        {code_snippet}
        ---
        
        Please explain this error and provide the corrected code.
        """
        
        # Use stream=True for a better chat-like experience in Streamlit
        response_stream = model.generate_content(prompt, stream=True)
        
        # Yield each chunk of text as it is generated
        for chunk in response_stream:
            if chunk.parts:
                yield chunk.parts[0].text

    except Exception as e:
        # Handle API errors gracefully
        msg = str(e)
        if 'API key not valid' in msg or 'API_KEY_INVALID' in msg:
            yield "**Gemini API key is invalid.** Please verify the key from Google AI Studio and update `GEMINI_API_KEY`.\n\n"
        else:
            yield f"**An error occurred while contacting the AI assistant:**\n\n{msg}\n\n"
        yield "Ensure the API key is correct and your project has quota/access to the selected model."

