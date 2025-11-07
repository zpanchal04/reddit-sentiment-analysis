import re
import string
import pandas as pd
from nltk.tokenize import word_tokenize

def clean_text(text):
    """
    Applies basic text cleaning to a single string or pandas Series.
    - Converts to lowercase
    - Removes URLs
    - Removes punctuation
    - Removes numbers
    - Removes extra whitespace
    """
    if isinstance(text, pd.Series):
        # Vectorized operations for pandas Series
        # Replace NaN with empty string
        text = text.fillna("")
        # Convert to lowercase
        text = text.str.lower()
        # Remove URLs
        text = text.str.replace(r'http\S+|www\S+', '', regex=True)
        # Remove punctuation
        text = text.str.replace(f'[{re.escape(string.punctuation)}]', '', regex=True)
        # Remove numbers
        text = text.str.replace(r'\d+', '', regex=True)
        # Remove extra spaces
        text = text.str.replace(r'\s+', ' ', regex=True).str.strip()
        return text
    
    # Single string processing
    if not isinstance(text, str):
        return ""
        
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def tokenize_and_remove_stopwords(text, stop_words_set):
    """
    Tokenizes the text and removes stopwords FROM THE PROVIDED SET.
    
    Args:
        text (str): A cleaned text string.
        stop_words_set (set): A set of stopwords provided by the caller.
        
    Returns:
        list: A list of filtered tokens.
    """
    
    # Tokenize the cleaned text
    tokens = word_tokenize(text)
    
    # Filter out stopwords (from the argument) and non-alphabetic/short tokens
    filtered_tokens = [
        word for word in tokens 
        if word.isalpha() and word not in stop_words_set and len(word) > 1
    ]
    
    return filtered_tokens

