import pytest
import os
import sys
from nltk.corpus import stopwords
import nltk

# --- Setup ---
# Add the 'src' directory to the Python path so we can import our modules
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if module_path not in sys.path:
    sys.path.append(module_path)

# Import the modules we want to test
from src import preprocessor
from src import analyzer

# Download NLTK data (needed for analyzer)
nltk.download('vader_lexicon')

# --- Tests for preprocessor.py ---

def test_clean_text_removes_urls():
    print("Testing URL removal...")
    # FIX 1: Added closing triple-quote for the multiline string
    text = """Check http://test.com
 now!"""
    # FIX 2: The expected result is "check now", not "check".
    # Our function correctly keeps "now" and removes the "!"
    expected = "check now" 
    result = preprocessor.clean_text(text)
    assert result == expected

def test_clean_text_removes_punctuation():
    print("Testing punctuation removal...")
    text = "Hello, world!!!"
    expected = "hello world"
    result = preprocessor.clean_text(text)
    assert result == expected

# --- Tests for analyzer.py ---

def test_vader_sentiment_analysis():
    print("Testing VADER positive/negative...")
    text_pos = "I love this product!"
    text_neg = "I hate this product!"
    assert analyzer.get_vader_sentiment(text_pos) == "positive"
    assert analyzer.get_vader_sentiment(text_neg) == "negative"

def test_vader_sentiment_empty_text():
    print("Testing VADER empty string...")
    empty_text = ""
    assert analyzer.get_vader_sentiment(empty_text) == "neutral"

