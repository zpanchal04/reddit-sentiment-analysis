import os
import io
import sys
import json
import pandas as pd
import re
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from wordcloud import WordCloud
from datetime import datetime
import pickle
import joblib
import praw # Added for comment analyzer
import praw.exceptions # Added for comment analyzer
from dotenv import load_dotenv # Added for comment analyzer
try:
    # Use Google's Generative AI (Gemini) if available
    import google.generativeai as genai
except Exception:
    genai = None

# Load environment variables
load_dotenv()

if genai is not None:
    # Prefer explicit API key from environment/Streamlit secrets
    gkey = (
        os.getenv('GEMINI_API_KEY')
        or os.getenv('GEMINI_API_KEY_JSON')
    )
    try:
        if gkey:
            # configure accepts api_key in many versions
            genai.configure(api_key=gkey)
    except Exception:
        # ignore configure issues; we'll attempt calls and surface errors later
        pass

def get_ai_model_comparison(selected_models):
    """Get AI-generated theoretical comparison of selected ML models using Google Gemini.

    This function relies solely on the Gemini API. If the SDK is not installed,
    or the API call fails, the function will return None and the app will show an error.
    """
    if genai is None:
        st.error('Google Generative AI SDK (google.generativeai) is not installed. Install it to enable AI comparisons.')
        return None

    prompt = (
        f"Generate a concise comparison for these models in sentiment analysis: {', '.join(selected_models)}.\n"
        "Return a single JSON object: keys are model names, values are objects with EXACTLY these fields: Algorithm Type, Key Strengths in sentiment analysis, Limitations for text analysis, Best Use Cases, AI/ML Implementation Details.\n"
        "Write each field as a short phrase of at most 7 words, no lists or long sentences, avoid brackets/quotes/markdown."
    )

    try:
        # Validate API key presence before attempting a call
        key_present = (
            os.getenv('GEMINI_API_KEY')
            or os.getenv('GEMINI_API_KEY_JSON')
            or (getattr(st, 'secrets', None) and st.secrets.get('GEMINI_API_KEY'))
        )
        if not key_present:
            st.error('Gemini API key not configured. Set GEMINI_API_KEY in .env or Streamlit secrets, then restart the app.')
            return None

        # Use the high-level GenerativeModel wrapper provided by the SDK.
        # Instantiate a model with safe defaults and fallbacks.
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
        gen_model = None
        for m in fallback_models:
            try:
                gen_model = genai.GenerativeModel(m)
                # quick no-op dry-run via content generation with minimal prompt to validate access
                _ = gen_model.generate_content("ping")
                break
            except Exception as _e:
                last_err = _e
                gen_model = None
                continue

        if gen_model is None:
            # Final fallback: discover available models from API and pick a supported one
            try:
                candidates = []
                for m in genai.list_models():
                    methods = getattr(m, 'supported_generation_methods', []) or []
                    if 'generateContent' in methods:
                        candidates.append(getattr(m, 'name', ''))
                # Prefer 1.5 flash/pro variants
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
                    gen_model = genai.GenerativeModel(pick)
                    _ = gen_model.generate_content('ping')
                else:
                    raise last_err or RuntimeError('No compatible Gemini model available. Try setting GEMINI_MODEL to a supported id (e.g., gemini-1.5-flash).')
            except Exception as _e2:
                raise last_err or _e2

        resp = gen_model.generate_content(prompt)
        # The response object exposes .text for convenience
        response_text = getattr(resp, 'text', None)
        if not response_text:
            # Try candidates structure
            try:
                response_text = resp.candidates[0].content[0].text
            except Exception:
                response_text = None

        if not response_text:
            st.error('No response text returned from Gemini model. Check GEMINI_API_KEY, model access and quotas.')
            return None

        # Extract JSON block from response
        start = response_text.find('{')
        end = response_text.rfind('}')
        if start != -1 and end != -1 and end > start:
            json_text = response_text[start:end+1]
        else:
            json_text = response_text

        comparison_data = json.loads(json_text)
        return comparison_data
    except Exception as e:
        err_text = str(e)
        if 'API key not valid' in err_text or 'API_KEY_INVALID' in err_text:
            st.error('Gemini API key is invalid. Verify the key from Google AI Studio and set GEMINI_API_KEY, then restart.')
        elif 'not found' in err_text or '404' in err_text:
            # Be quiet during dataset refreshes; continue without AI analysis
            st.info('AI analysis is temporarily unavailable for the current Gemini model. Continuing without it.')
        else:
            st.info('AI analysis could not be generated. Continuing without it.')
        return None

# Data format support
try:
    import pyarrow  # for parquet and feather support
    import tables   # for HDF5 support
    import lxml     # for XML support
    import openpyxl # for Excel support
except ImportError:
    pass  # Will handle missing dependencies when needed

from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# --- Setup ---
# Load environment variables (needed for PRAW in comment analyzer)
load_dotenv()

# Ensure src is importable
# Assuming dashboard.py is in the project root (1.Sentiment Analysis)
PROJECT_ROOT = os.path.dirname(__file__) # Get the directory where this script is located
SRC_PATH = os.path.join(PROJECT_ROOT, 'src')
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

try:
    from src.analyzer import get_vader_sentiment
    from src.preprocessor import clean_text
    print("Successfully imported analyzer and preprocessor.")
except ImportError as e:
    st.error(f"Error importing modules from 'src': {e}. Ensure src/__init__.py, src/analyzer.py, src/preprocessor.py, and src/ai_helper.py exist.")
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred during module import: {e}")
    st.stop()


# Create models directory if it doesn't exist
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

# -------------------------------
# Enhanced Styling (From Original Dashboard)
# -------------------------------
st.set_page_config(
    page_title=' Next-Gen Reddit Sentiment AI Dashboard',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Custom CSS (combined light/dark mode adjustments)
# (Fixed CSS selectors for section headers)
st.markdown("""
<style>
    /* General Styles */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem; border-radius: 10px; margin-bottom: 2rem; color: white; text-align: center;
    }
    .metric-card { /* Used for KPI boxes */
        border: 1px solid #e0e6ed; border-radius: 15px; padding: 1.5rem; margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05); transition: transform 0.2s;
    }
    .metric-card:hover { transform: translateY(-2px); box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1); }
    .kpi-box { /* More detailed KPI box style */
        border-radius: 14px; padding: 16px; box-shadow: 0 4px 6px rgba(0,0,0,.06);
    }
    .kpi-title { font-size: 0.9rem; opacity: 0.9; }
    .kpi-value { font-size: 2rem; font-weight: 800; }
    .best-model { /* Highlight best model */
        background: linear-gradient(135deg, #ffd700 0%, #ffed4e 100%); border: 2px solid #ff6b35; animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(255, 107, 53, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(255, 107, 53, 0); }
        100% { box-shadow: 0 0 0 0 rgba(255, 107, 53, 0); }
    }
    .section-header { /* Headers for sections */
        color: #ffffff; border-bottom: 3px solid #60a5fa; padding-bottom: 0.5rem; margin: 2rem 0 1rem 0;
    }
    .stProgress > div > div > div > div { /* Progress bar gradient */
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    /* Centering for Model Training */
     .center-container {
        display: flex; flex-direction: column; align-items: center; justify-content: center;
     }
     /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] { padding: 8px 14px; border-radius: 10px; }

    /* --- Light Theme Specific --- */
    body.light .kpi-box { background: linear-gradient(135deg,#f5f7fa 0%, #dde5f0 100%); border:1px solid #e0e6ed; }
    body.light .kpi-title { color:#374151; }
    body.light .kpi-value { color:#111827; }
    body.light .metric-card { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); border-color: #e0e6ed; }
    body.light .stTabs [data-baseweb="tab"] { background: #f3f4f6; }
    body.light .stTabs [aria-selected="true"] { background: #e5e7eb; color: #111827; }
    body.light [data-testid="stSidebar"] { background:#f8fafc; }
    body.light [data-testid="stSidebar"] * { color:#0f172a !important; }
    body.light .stButton>button { background:#2563eb; color:#ffffff; border:1px solid #1d4ed8; border-radius:12px; }
    body.light div[data-baseweb="select"]>div { background:#ffffff; border:1px solid #e5e7eb; border-radius:10px; color:#0f172a; }
    body.light .stDateInput>div>div>input, body.light .stTextInput input { background:#ffffff; color:#0f172a; border:1px solid #e5e7eb; }
    body.light section[data-testid="stFileUploader"] div[tabindex] { background:#ffffff; border:1px dashed #cbd5e1; border-radius:10px; }
    body.light .stDownloadButton>button { background:#059669; color:#ffffff; border-radius:10px; }
    body.light .section-header { color: #ffffff; border-bottom-color: #60a5fa; } /* Fixed: Force white heading */

    /* --- Dark Theme Specific --- */
    body.dark, body.dark .main, body.dark .block-container { background-color:#0e1117; color:#e5e7eb; }
    body.dark .kpi-box { background: linear-gradient(135deg,#1f2937 0%,#0b1220 100%); border:1px solid #374151; color:#e5e7eb; }
    body.dark .kpi-title { color:#9ca3af; } /* Slightly lighter text for title */
    body.dark .kpi-value { color:#f9fafb; } /* Brightest text for value */
    body.dark .metric-card { background: linear-gradient(135deg,#1f2937 0%,#111827 100%); border-color:#374151; }
    body.dark .stTabs [data-baseweb="tab"] { background:#111827; color:#e5e7eb; border:1px solid #374151; }
    body.dark .stTabs [aria-selected="true"] { background:#1f2937; color:#ffffff; }
    body.dark [data-testid="stSidebar"] { background-color:#0b1220; }
    body.dark [data-testid="stSidebar"] * { color:#e5e7eb !important; }
    body.dark .stButton>button { background:#334155; color:#e5e7eb; border:1px solid #475569; border-radius:12px; }
    body.dark div[data-baseweb="select"]>div { background:#111827; border:1px solid #374151; border-radius:10px; color:#e5e7eb; }
    body.dark .stDateInput>div>div>input, body.dark .stTextInput input { background:#111827; color:#e5e7eb; border:1px solid #374151; }
    body.dark section[data-testid="stFileUploader"] div[tabindex] { background:#111827; border:1px dashed #374151; border-radius:10px; }
    body.dark .stDownloadButton>button { background:#0ea5e9; color:#0b1220; border-radius:10px; }
    body.dark .section-header { color: #ffffff; border-bottom-color: #60a5fa; } /* Force white headings in dark mode */

</style>
""", unsafe_allow_html=True)

# -------------------------------
# Helper Functions (Combined)
# -------------------------------
@st.cache_data(show_spinner=False)
def load_processed_data():
    """Loads all necessary processed data files."""
    processed_dir = os.path.join(PROJECT_ROOT, 'data', 'processed')
    labeled_path = os.path.join(processed_dir, 'labeled-posts.csv')
    cleaned_path = os.path.join(processed_dir, 'cleaned-posts.csv')
    raw_path = os.path.join(PROJECT_ROOT, 'data', 'raw', 'reddit-posts.json')

    df_labeled = pd.read_csv(labeled_path) if os.path.exists(labeled_path) else None
    df_cleaned = pd.read_csv(cleaned_path) if os.path.exists(cleaned_path) else None
    df_raw = pd.read_json(raw_path) if os.path.exists(raw_path) else None

    # Normalize columns (simple version for consistency)
    def normalize_columns(df):
        if df is None: return None
        df.columns = df.columns.str.lower().str.strip()
        # Specific renames if needed
        if 'vader_sentiment' in df.columns and 'sentiment' not in df.columns:
             df = df.rename(columns={'vader_sentiment': 'sentiment'})
        return df

    return normalize_columns(df_labeled), normalize_columns(df_cleaned), normalize_columns(df_raw)

def ensure_datetime(series):
    """Safely converts a series to datetime, handling potential errors."""
    if series is None: return None
    try:
        return pd.to_datetime(series, errors='coerce')
    except Exception:
        return pd.Series([pd.NaT] * len(series))

def compute_vader_if_missing(df, text_col='text'):
    """Computes VADER sentiment if 'sentiment' column is missing."""
    if df is None: return None
    if 'sentiment' not in df.columns:
        df = df.copy()
        # Find the best available text column
        text_col_to_use = next((col for col in ['tokens_joined', 'cleaned_text', 'text'] if col in df.columns), None)
        if not text_col_to_use:
            st.warning("Could not find a suitable text column for VADER analysis.")
            return df  # Return df as is if no text column found
            
        df['sentiment'] = df[text_col_to_use].astype(str).fillna('').apply(get_vader_sentiment)
    return df

def optimize_dataframe(df):
    """Downcast numerics and convert high-cardinality strings to more efficient dtypes."""
    try:
        if df is None:
            return df
        df = df.copy()
        # Downcast numerics
        for col in df.select_dtypes(include=['int64', 'int32', 'float64']).columns:
            try:
                if str(df[col].dtype).startswith('float'):
                    df[col] = pd.to_numeric(df[col], downcast='float')
                else:
                    df[col] = pd.to_numeric(df[col], downcast='integer')
            except Exception:
                pass
        # Convert common string cols to category
        for col in ['sentiment', 'subreddit']:
            if col in df.columns and df[col].dtype == object:
                try:
                    df[col] = df[col].astype('category')
                except Exception:
                    pass
        return df
    except Exception:
        return df

def get_available_models():
    """Returns a dictionary of available sklearn models (broad selection)."""
    from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier, RidgeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier
    from sklearn.svm import SVC
    return {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=2000),
        'Support Vector Machine (LinearSVC)': LinearSVC(random_state=42, max_iter=2000),
        'SVC (RBF Kernel)': SVC(random_state=42, kernel='rbf', probability=False),
        'Naive Bayes (Multinomial)': MultinomialNB(alpha=0.1),
        'SGD Classifier': SGDClassifier(random_state=42, max_iter=2000),
        'Passive Aggressive': PassiveAggressiveClassifier(random_state=42, max_iter=2000),
        'Ridge Classifier': RidgeClassifier(random_state=42),
        'KNN (k=5)': KNeighborsClassifier(n_neighbors=5),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=None),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=300),
        'Extra Trees': ExtraTreesClassifier(random_state=42, n_estimators=300),
        'AdaBoost': AdaBoostClassifier(random_state=42, n_estimators=200),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=200)
    }

# --- REDDIT AUTHENTICATION (for Comment Analyzer) ---
@st.cache_resource
def get_reddit_instance():
    """Authenticates and returns a PRAW Reddit instance."""
    try:
        reddit = praw.Reddit(
            client_id=os.getenv("CLIENT_ID"),
            client_secret=os.getenv("CLIENT_SECRET"),
            username=os.getenv("USERNAME"),
            password=os.getenv("PASSWORD"),
            user_agent=os.getenv("USER_AGENT")
        )
        # Test connection silently in the background
        _ = reddit.user.me()
        print("Successfully connected and authenticated with Reddit.")
        return reddit
    except Exception as e:
        print(f"Error connecting to Reddit API: {e}")
        return None

# --- COMMENT ANALYZER CORE FUNCTION ---
@st.cache_data(ttl=600, show_spinner=False) # Cache results for 10 minutes
def fetch_and_analyze_comments(_reddit_instance, url):
    """ Fetches, cleans, and analyzes comments from a Reddit post URL."""
    status_placeholder = st.empty()
    try:
        if not _reddit_instance:
             return None, "Reddit connection failed. Cannot fetch comments."
        # 1. Fetch Submission
        status_placeholder.info("🔗 Fetching submission from Reddit...")
        submission = _reddit_instance.submission(url=url)

        # 2. Fetch Comments
        status_placeholder.info("💬 Fetching and expanding all comments (this might take a moment)...")
        submission.comments.replace_more(limit=None) # Fetch all comments

        comment_list = []
        for comment in submission.comments.list():
            if isinstance(comment, praw.models.Comment):
                comment_list.append({
                    "author": str(comment.author),
                    "text": comment.body,
                    "score": comment.score
                })

        if not comment_list:
            status_placeholder.empty()
            return None, "🤷 No comments found on this post."

        df = pd.DataFrame(comment_list)

        # 3. Clean and Analyze
        status_placeholder.info("🧹 Cleaning and analyzing comments with VADER...")
        # Use vectorized clean_text operation
        df['cleaned_text'] = clean_text(df['text'])
        df['sentiment'] = df['cleaned_text'].apply(get_vader_sentiment) # Use imported get_vader_sentiment

        status_placeholder.empty() # Clear status message
        return df, None

    except praw.exceptions.InvalidURL:
        status_placeholder.empty()
        return None, "❌ Invalid Reddit URL. Please check the URL format."
    except Exception as e:
        status_placeholder.empty()
        return None, f"❌ An unexpected error occurred: {e}"


# --- MODEL TRAINING FUNCTION (From Original Dashboard) ---
def train_model(df, model_name, text_col='tokens_joined', label_col='sentiment'):
    """Train a single model and return results including pipeline."""
    if df is None or len(df) < 10:
        st.warning("Not enough data to train.")
        return None

    # Filter out neutral for binary classification (Pos vs Neg)
    df_filtered = df[df[label_col] != 'neutral'].copy()
    if len(df_filtered) < 10:
        st.warning("Not enough positive/negative data to train after filtering neutrals.")
        return None

    # Use the correct text_col
    X = df_filtered[text_col].fillna('')
    y = df_filtered[label_col]

    if y.nunique() < 2:
        st.warning(f"Only one class ('{y.unique()[0]}') found after filtering. Cannot train classifier.")
        return None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    models = get_available_models()
    if model_name not in models:
        st.error(f"Model '{model_name}' not found.")
        return None
    model = models[model_name]

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=2000, stop_words='english', ngram_range=(1, 2))),
        ('classifier', model)
    ])

    pipeline.fit(X_train, Y_train := y_train)
    y_pred = pipeline.predict(X_test)

    accuracy = pipeline.score(X_test, y_test)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)

    cm = confusion_matrix(y_test, y_pred)
    labels = sorted(y_test.unique())

    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'labels': labels,
        'pipeline': pipeline,
        'classification_report': classification_report(y_test, y_pred, zero_division=0),
        'y_test': pd.Series(y_test),
        'y_pred': pd.Series(y_pred).reset_index(drop=True),
        'test_indices': pd.Series(y_test).index.tolist()
    }

# --- MODEL SAVING/LOADING ---
def save_model(model_result, model_name):
    """Save trained model pipeline to disk using joblib."""
    if model_result and 'pipeline' in model_result:
        # Sanitize model name for filename
        safe_model_name = model_name.replace(' ', '_').replace('(', '').replace(')', '').lower()
        model_path = os.path.join(MODELS_DIR, f"{safe_model_name}_model.joblib") # Use joblib
        try:
            joblib.dump(model_result['pipeline'], model_path)
            print(f"Model {model_name} saved to {model_path}")
            return model_path
        except Exception as e:
            st.error(f"Error saving model {model_name}: {e}")
    return None

def load_model(model_name):
    """Load trained model pipeline from disk using joblib."""
    safe_model_name = model_name.replace(' ', '_').replace('(', '').replace(')', '').lower()
    model_path = os.path.join(MODELS_DIR, f"{safe_model_name}_model.joblib")
    if os.path.exists(model_path):
        try:
            return joblib.load(model_path)
        except Exception as e:
            st.error(f"Error loading model {model_name}: {e}")
            return None
    return None

# -------------------------------
# Theme Handling & Plot Styling
# -------------------------------
# Auto-detect Streamlit's base theme
current_base_theme = (st.get_option('theme.base') or 'light').lower()
st.session_state['ui_theme'] = 'Dark' if current_base_theme == 'dark' else 'Light'

# Apply body class based on detected theme for CSS targeting
st.markdown(f'<body class="{st.session_state.ui_theme.lower()}">', unsafe_allow_html=True)

def _theme_colors():
    """Returns color dict based on theme state."""
    # This function is now redundant because of the new style_fig, but harmless.
    if st.session_state.get('ui_theme') == 'Dark':
        return {'bg': '#0e1117', 'panel': '#111827', 'grid': '#1f2937', 'text': '#e5e7eb'}
    return {'bg': '#ffffff', 'panel': '#f5f7fa', 'grid': '#eef2f7', 'text': '#111827'}

def style_fig(fig):
    """Applies theme colors to a Plotly figure."""
    # Force black background with white text for all figures
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#000000',
        plot_bgcolor='#000000',
        font_color='#ffffff',
        legend_bgcolor='#000000',
        legend_bordercolor='#333333',
        legend_borderwidth=0,
        title=dict(font=dict(color='#ffffff'))
    )
    # 2D axes styling
    fig.update_xaxes(
        gridcolor='#333333',
        title_font_color='#ffffff',
        tickfont_color='#ffffff'
    )
    fig.update_yaxes(
        gridcolor='#333333',
        title_font_color='#ffffff',
        tickfont_color='#ffffff'
    )
    # 3D scene styling (no-op for non-3D figs)
    fig.update_scenes(
        bgcolor='#000000',
        xaxis=dict(
            backgroundcolor='#000000', gridcolor='#333333', zerolinecolor='#333333',
            color='#ffffff', title=dict(font=dict(color='#ffffff'))
        ),
        yaxis=dict(
            backgroundcolor='#000000', gridcolor='#333333', zerolinecolor='#333333',
            color='#ffffff', title=dict(font=dict(color='#ffffff'))
        ),
        zaxis=dict(
            backgroundcolor='#000000', gridcolor='#333333', zerolinecolor='#333333',
            color='#ffffff', title=dict(font=dict(color='#ffffff'))
        )
    )
    return fig

# -------------------------------
# Main App Header
# -------------------------------
st.markdown('<div class="main-header"><h1> Next-Gen Reddit Sentiment AI Dashboard</h1><p>Advanced Machine Learning & Real-time Analytics for AI Influencer Sentiment Analysis</p></div>', unsafe_allow_html=True)

# -------------------------------
# Load Data (Once at the start)
# -------------------------------
df_labeled, df_cleaned, df_raw = load_processed_data()

# Session-state flags for refresh / uploaded override
if 'app_cleared' not in st.session_state:
    st.session_state['app_cleared'] = False
if 'uploaded_df' not in st.session_state:
    st.session_state['uploaded_df'] = None

# Check if essential data exists (only error when not cleared and no uploaded override)
if not st.session_state['app_cleared'] and st.session_state['uploaded_df'] is None and df_labeled is None:
    st.error('❌ **Critical Error:** `labeled-posts.csv` not found in `data/processed/`.')
    st.error("Please ensure you have run the data processing notebooks (02_preprocessor.ipynb, 03_modeling.ipynb) first to generate this file.")
    st.stop()

if df_cleaned is None and not st.session_state['app_cleared']:
    st.warning('⚠️ `cleaned-posts.csv` not found. Some features like word clouds might be limited.')
if df_raw is None and not st.session_state['app_cleared']:
    st.warning('⚠️ `reddit-posts.json` not found. Timestamps and engagement data might be missing.')

# Decide source DataFrame: uploaded override -> cleared state -> disk file
if st.session_state.get('uploaded_df') is not None:
    # Use uploaded dataset stored in session state
    base_df = st.session_state['uploaded_df'].copy()
elif st.session_state.get('app_cleared'):
    # User requested a full reset: start with an empty dataframe
    base_df = pd.DataFrame()
else:
    # Default: use labeled data from disk
    base_df = df_labeled.copy()

# Add cleaned text if available (merge only when base_df is not empty)
if base_df is not None and not base_df.empty and df_cleaned is not None and 'id' in df_cleaned.columns:
     if 'tokens_joined' in df_cleaned.columns and 'id' in df_cleaned.columns:
         base_df = pd.merge(base_df, df_cleaned[['id', 'tokens_joined']], on='id', how='left')
     elif 'cleaned_text' in df_cleaned.columns and 'id' in df_cleaned.columns:
         base_df = pd.merge(base_df, df_cleaned[['id', 'cleaned_text']], on='id', how='left')


# Add raw data info if available (merge only when base_df is not empty)
if base_df is not None and not base_df.empty and df_raw is not None and 'id' in df_raw.columns:
    raw_cols_to_merge = ['id']
    for col in ['created_date', 'score', 'num_comments', 'subreddit']:
        if col in df_raw.columns:
            raw_cols_to_merge.append(col)
    if len(raw_cols_to_merge) > 1:
        base_df = pd.merge(base_df, df_raw[raw_cols_to_merge], on='id', how='left')


# Ensure required columns exist and handle missing sentiment
text_col_options = ['tokens_joined', 'cleaned_text', 'text']
text_col = None
if base_df is not None and not base_df.empty:
    text_col = next((col for col in text_col_options if col in base_df.columns), None)
    if not text_col:
         st.error("❌ Could not find a suitable text column ('tokens_joined', 'cleaned_text', or 'text') in the loaded data.")
         st.stop()
    base_df = compute_vader_if_missing(base_df, text_col) # Ensures 'sentiment' column exists

# Date handling
if base_df is not None and not base_df.empty and 'created_date' in base_df.columns:
    base_df['created_date'] = ensure_datetime(base_df['created_date'])
elif base_df is not None and base_df.empty:
    # If app was cleared and we have an empty base_df, do not warn about dates
    pass
else:
    st.warning("⚠️ 'created_date' column not found. Time series analysis will be unavailable.")

# Optimize base dataframe for faster downstream operations
base_df = optimize_dataframe(base_df)

# -------------------------------
# Sidebar Controls (Applied to base_df)
# -------------------------------
st.sidebar.markdown("## 🎛️ Control Panel")

# Data upload
uploaded_file = st.sidebar.file_uploader(
    '📁 Upload Additional Data',
    type=['csv','tsv','txt','json','xlsx','xls','parquet','feather','pkl','pickle','h5','hdf','hdf5','orc','html']
)
if uploaded_file is not None:
    # Generate a stable token for this file to avoid reprocessing on every rerun
    try:
        file_token = f"{uploaded_file.name}:{getattr(uploaded_file, 'size', None) or uploaded_file.getbuffer().nbytes}"
    except Exception:
        # Fallback to name-only token if size isn't available
        file_token = f"{uploaded_file.name}"

    if 'uploaded_file_token' not in st.session_state:
        st.session_state['uploaded_file_token'] = None

    # Only process when the uploaded file is NEW (token changed)
    if st.session_state['uploaded_file_token'] != file_token:
        try:
            # Auto-detect file type and read with pandas
            ext = os.path.splitext(uploaded_file.name)[1].lower()
            up_df = None
            # Read the uploaded file into bytes once; reuse for multiple parsers
            try:
                data_bytes = uploaded_file.getbuffer().tobytes()
            except Exception:
                try:
                    data_bytes = uploaded_file.read()
                except Exception:
                    data_bytes = None
            try:
                if ext in ['.csv']:
                    up_df = pd.read_csv(io.BytesIO(data_bytes) if data_bytes is not None else uploaded_file)
                elif ext in ['.tsv']:
                    up_df = pd.read_csv(io.BytesIO(data_bytes) if data_bytes is not None else uploaded_file, sep='\t')
                elif ext in ['.txt']:
                    up_df = pd.read_csv(io.BytesIO(data_bytes) if data_bytes is not None else uploaded_file, sep=None, engine='python')
                elif ext in ['.json']:
                    try:
                        up_df = pd.read_json(io.BytesIO(data_bytes) if data_bytes is not None else uploaded_file, lines=False)
                    except ValueError:
                        up_df = pd.read_json(io.BytesIO(data_bytes) if data_bytes is not None else uploaded_file, lines=True)
                elif ext in ['.xlsx', '.xls']:
                    try:
                        engine = 'openpyxl' if ext == '.xlsx' else None
                        up_df = pd.read_excel(io.BytesIO(data_bytes) if data_bytes is not None else uploaded_file, engine=engine)
                    except ImportError as e:
                        st.sidebar.error('Excel support requires openpyxl (for .xlsx) or xlrd (for .xls).')
                        up_df = None
                elif ext in ['.parquet']:
                    try:
                        up_df = pd.read_parquet(io.BytesIO(data_bytes) if data_bytes is not None else uploaded_file)
                    except Exception:
                        st.sidebar.error('Parquet support requires pyarrow or fastparquet installed.')
                        up_df = None
                elif ext in ['.feather']:
                    try:
                        up_df = pd.read_feather(io.BytesIO(data_bytes) if data_bytes is not None else uploaded_file)
                    except Exception:
                        st.sidebar.error('Feather support requires pyarrow installed.')
                        up_df = None
                elif ext in ['.pkl', '.pickle']:
                    up_df = pd.read_pickle(io.BytesIO(data_bytes) if data_bytes is not None else uploaded_file)
                elif ext in ['.h5', '.hdf', '.hdf5']:
                    try:
                        up_df = pd.read_hdf(io.BytesIO(data_bytes) if data_bytes is not None else uploaded_file)
                    except (ImportError, ValueError):
                        st.sidebar.error('HDF5 support requires PyTables (tables) installed.')
                        up_df = None
                elif ext in ['.orc']:
                    try:
                        # pandas can read ORC via pyarrow if available
                        up_df = pd.read_orc(io.BytesIO(data_bytes) if data_bytes is not None else uploaded_file)
                    except Exception:
                        st.sidebar.error('ORC support requires pyarrow installed.')
                        up_df = None
                elif ext in ['.html']:
                    try:
                        tables = pd.read_html(io.BytesIO(data_bytes) if data_bytes is not None else uploaded_file)
                    except ImportError:
                        st.sidebar.error('HTML table parsing requires lxml installed.')
                        tables = []
                    up_df = tables[0] if tables else None
                else:
                    # Fallback: try CSV
                    up_df = pd.read_csv(io.BytesIO(data_bytes) if data_bytes is not None else uploaded_file)
            except Exception:
                # Final fallback: try to parse as CSV with python engine
                try:
                    up_df = pd.read_csv(io.BytesIO(data_bytes) if data_bytes is not None else uploaded_file, sep=None, engine='python')
                except Exception as _:
                    up_df = None

            if up_df is None:
                # Could not read uploaded file; keep UI silent per request
                print('Could not read uploaded file or unsupported format.')
            else:
                # Basic column normalization for uploaded file
                up_df.columns = up_df.columns.str.lower().str.strip()
                if 'vader_sentiment' in up_df.columns and 'sentiment' not in up_df.columns:
                    up_df = up_df.rename(columns={'vader_sentiment': 'sentiment'})
                # Optimize types to speed up downstream operations
                up_df = optimize_dataframe(up_df)

                # Ensure text and sentiment columns
                up_text_col = next((col for col in text_col_options if col in up_df.columns), None)
                if up_text_col:
                    # Compute VADER labels for the uploaded dataframe
                    up_df = compute_vader_if_missing(up_df, up_text_col)

                    # Store uploaded dataframe in session state so the app will use it as the primary dataset
                    st.session_state['uploaded_df'] = up_df.reset_index(drop=True)
                    st.session_state['uploaded_file_token'] = file_token
                    st.session_state['app_cleared'] = False
                    st.session_state['upload_complete'] = True

                    # Clear cached computations and rerun to refresh UI using the uploaded data
                    try:
                        st.cache_data.clear()
                    except Exception:
                        pass
                    # No explicit rerun: Streamlit will rerun automatically due to session_state changes
                else:
                    # Do not display sidebar messages per request; log missing column
                    print("Uploaded file missing required text column ('text', 'cleaned_text', or 'tokens_joined').")

        except Exception as e:
            # Do not show sidebar error messages; log to console instead
            print('An error occurred while processing the uploaded file:', e)
    else:
        # Skip reprocessing the same file to avoid infinite rerun loops
        pass

# Show upload completion prompt (if any)
if st.session_state.get('upload_complete'):
    st.sidebar.info('✅ Uploading complete. Click Refresh to apply the new data.')

# Sidebar manual refresh button (below upload)
refresh_clicked = st.sidebar.button('🔄 Refresh')
if refresh_clicked:
    # If no upload was provided, show the upload prompt by clearing to an empty app state
    if st.session_state.get('uploaded_df') is None:
        st.session_state['app_cleared'] = True
        st.session_state['uploaded_file_token'] = None
    # Apply current data without clearing uploaded data
    st.session_state['upload_complete'] = False
    try:
        st.cache_data.clear()
    except Exception:
        pass
    # Rerun so the UI reflects the current dataset
    st.rerun()


# Sample Data Section
st.sidebar.markdown("### 📥 Sample Data")

# List of supported file formats by pandas
file_formats = {
    'CSV (.csv)': 'csv',
    'JSON (.json)': 'json',
    'Excel (.xlsx)': 'xlsx',
    'Parquet (.parquet)': 'parquet',
    'Feather (.feather)': 'feather',
    'HTML (.html)': 'html',
    'XML (.xml)': 'xml',
    'HDF5 (.h5)': 'h5',
    'Pickle (.pkl)': 'pkl',
    'SQL (.sql)': 'sql'
}

selected_format = st.sidebar.selectbox('Convert sample data to:', list(file_formats.keys()))

if st.sidebar.button('Download Sample Data'):
    # Read the sample data
    sample_data_path = os.path.join('data', 'processed', 'sample_data.csv')
    if os.path.exists(sample_data_path):
        df = pd.read_csv(sample_data_path)
        
        # Convert to selected format
        format_extension = file_formats[selected_format]
        output = io.BytesIO()
        
        try:
            if format_extension == 'csv':
                output_data = df.to_csv(index=False)
                file_name = 'sample_data.csv'
                mime = 'text/csv'
            elif format_extension == 'json':
                output_data = df.to_json(orient='records')
                file_name = 'sample_data.json'
                mime = 'application/json'
            elif format_extension == 'xlsx':
                with io.BytesIO() as excel_buffer:
                    df.to_excel(excel_buffer, index=False)
                    output_data = excel_buffer.getvalue()
                file_name = 'sample_data.xlsx'
                mime = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            elif format_extension == 'parquet':
                with io.BytesIO() as parquet_buffer:
                    df.to_parquet(parquet_buffer)
                    output_data = parquet_buffer.getvalue()
                file_name = 'sample_data.parquet'
                mime = 'application/octet-stream'
            elif format_extension == 'feather':
                with io.BytesIO() as feather_buffer:
                    df.to_feather(feather_buffer)
                    output_data = feather_buffer.getvalue()
                file_name = 'sample_data.feather'
                mime = 'application/octet-stream'
            elif format_extension == 'html':
                output_data = df.to_html(index=False)
                file_name = 'sample_data.html'
                mime = 'text/html'
            elif format_extension == 'xml':
                output_data = df.to_xml()
                file_name = 'sample_data.xml'
                mime = 'application/xml'
            elif format_extension == 'h5':
                with io.BytesIO() as h5_buffer:
                    df.to_hdf(h5_buffer, key='sample_data', mode='w')
                    output_data = h5_buffer.getvalue()
                file_name = 'sample_data.h5'
                mime = 'application/x-hdf5'
            elif format_extension == 'pkl':
                with io.BytesIO() as pkl_buffer:
                    df.to_pickle(pkl_buffer)
                    output_data = pkl_buffer.getvalue()
                file_name = 'sample_data.pkl'
                mime = 'application/octet-stream'
            elif format_extension == 'sql':
                # Create SQL table creation and insert statements
                table_name = 'sample_data'
                # Get column types
                dtype_map = {
                    'object': 'TEXT',
                    'int64': 'INTEGER',
                    'float64': 'FLOAT',
                    'datetime64[ns]': 'TIMESTAMP',
                    'bool': 'BOOLEAN'
                }
                columns = [f"{col} {dtype_map.get(str(df[col].dtype), 'TEXT')}" for col in df.columns]
                create_table = f"CREATE TABLE {table_name} (\n    " + ",\n    ".join(columns) + "\n);\n\n"
                
                # Generate INSERT statements
                insert_statements = []
                for _, row in df.iterrows():
                    values = []
                    for val in row:
                        if pd.isna(val):
                            values.append('NULL')
                        elif isinstance(val, (int, float)):
                            values.append(str(val))
                        else:
                            # Fix the escaping issue by using string concatenation instead of f-string
                            escaped_val = str(val).replace("'", "''")
                            values.append(f"'{escaped_val}'")
                    insert_statements.append(f"INSERT INTO {table_name} ({', '.join(df.columns)}) VALUES ({', '.join(values)});")
                
                # Combine all SQL statements
                output_data = create_table + "\n".join(insert_statements)
                file_name = 'sample_data.sql'
                mime = 'text/plain'
                
            st.sidebar.download_button(
                label=f'📥 Click to download {file_name}',
                data=output_data,
                file_name=file_name,
                mime=mime
            )
        except Exception as e:
            st.sidebar.error(f'Error converting file: {str(e)}')
            if format_extension == 'xlsx':
                st.sidebar.info('For Excel files, please ensure openpyxl is installed')
            elif format_extension in ['parquet', 'feather']:
                st.sidebar.info('For Parquet/Feather files, please ensure pyarrow is installed')
            elif format_extension == 'h5':
                st.sidebar.info('For HDF5 files, please ensure tables (PyTables) is installed')

# Filters (Applied to a copy for display)
st.sidebar.markdown("### 🔍 Filters")
filtered_df = base_df.copy() # Apply filters to a copy

sentiment_filter = st.sidebar.selectbox('Sentiment', ['All', 'Positive', 'Negative', 'Neutral'])
if sentiment_filter != 'All':
    filtered_df = filtered_df[filtered_df['sentiment'].str.lower() == sentiment_filter.lower()]

# Subreddit filter is removed as requested in the code

if 'created_date' in filtered_df.columns and not filtered_df['created_date'].isnull().all():
    min_date = filtered_df['created_date'].min()
    max_date = filtered_df['created_date'].max()
    if pd.notna(min_date) and pd.notna(max_date):
        date_range = st.sidebar.date_input(
            '📅 Date Range',
            value=(min_date.date(), max_date.date()),
            min_value=min_date.date(),
            max_value=max_date.date()
        )
        if len(date_range) == 2:
            start_dt = pd.to_datetime(date_range[0])
            # Add one day and subtract a tiny amount to include the end date fully
            end_dt = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            filtered_df = filtered_df[
                (filtered_df['created_date'] >= start_dt) &
                (filtered_df['created_date'] <= end_dt)
            ]

# -------------------------------
# Main Navigation Tabs
# -------------------------------

def render_comment_analyzer():
    # Placeholder - real implementation is provided in the TAB 4 section below.
    return

# If the app was cleared (or base_df is empty), show a centered prompt to upload datasets
if base_df is None or (hasattr(base_df, 'empty') and base_df.empty):
    st.markdown('<div style="display:flex; align-items:center; justify-content:center; height:200px;"><h2>Upload the datasets</h2></div>', unsafe_allow_html=True)
    # Always show a button for Real Time Sentiment Analysis even after refresh
    if st.button('Real Time Sentiment Analysis'):
        # Render the comment analyzer inline below the message
        render_comment_analyzer()
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs([
    '📊 Overview Dashboard',
    '🧠 Model Training & Evaluation',
    '💡 Insights & Exports',
    '💬 Real-time Comment Analyzer'
])


def render_comment_analyzer():
    """Reusable comment analyzer UI (extracted from TAB 4)."""
    st.markdown('<h2 class="section-header">💬 Real-time Comment Analyzer</h2>', unsafe_allow_html=True)
    st.markdown("Paste the URL of any Reddit post to fetch all its comments and analyze their sentiment in real-time.")

    # 1. Get Reddit Instance (reuse the cached one)
    reddit_instance_for_comments = get_reddit_instance()
    if not reddit_instance_for_comments:
        st.error("Could not connect to Reddit API for comment analysis. Check credentials.")
        return

    # 2. User Input
    url = st.text_input("Enter a Reddit Post URL:", placeholder="https://www.reddit.com/r/...", key="comment_url_input")

    if st.button("Analyze Comments", key="analyze_comment_button"):
        if url:
            with st.spinner("Fetching and analyzing comments..."):
                analyzed_df, error = fetch_and_analyze_comments(reddit_instance_for_comments, url)

            if error:
                st.error(error)
            elif analyzed_df is not None:
                st.success(f"Analysis complete! Found and analyzed {len(analyzed_df)} comments.")

                # --- Overall Sentiment ---
                st.subheader("Overall Comment Sentiment")
                sentiment_counts = analyzed_df['sentiment'].value_counts()
                sentiment_df = pd.DataFrame({'sentiment': sentiment_counts.index, 'count': sentiment_counts.values})
                total_comments = int(sentiment_df['count'].sum())
                fig_comment_pie = px.pie(
                    sentiment_df, values='count', names='sentiment',
                    title='Comment Sentiment Distribution',
                    hole=0.3,
                    color='sentiment',
                    color_discrete_map={'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#f39c12'}
                )
                fig_comment_pie.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(style_fig(fig_comment_pie), use_container_width=True)

                # Numeric summary: counts and percentages
                pos = int(sentiment_df.loc[sentiment_df['sentiment']=='positive', 'count'].sum()) if 'positive' in sentiment_df['sentiment'].values else 0
                neg = int(sentiment_df.loc[sentiment_df['sentiment']=='negative', 'count'].sum()) if 'negative' in sentiment_df['sentiment'].values else 0
                neu = int(sentiment_df.loc[sentiment_df['sentiment']=='neutral', 'count'].sum()) if 'neutral' in sentiment_df['sentiment'].values else 0
                pos_p = (pos/total_comments*100) if total_comments else 0
                neg_p = (neg/total_comments*100) if total_comments else 0
                neu_p = (neu/total_comments*100) if total_comments else 0

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown(f'<div class="kpi-box"><div class="kpi-title">Positive</div><div class="kpi-value">{pos} ({pos_p:.1f}%)</div></div>', unsafe_allow_html=True)
                with c2:
                    st.markdown(f'<div class="kpi-box"><div class="kpi-title">Negative</div><div class="kpi-value">{neg} ({neg_p:.1f}%)</div></div>', unsafe_allow_html=True)
                with c3:
                    st.markdown(f'<div class="kpi-box"><div class="kpi-title">Neutral</div><div class="kpi-value">{neu} ({neu_p:.1f}%)</div></div>', unsafe_allow_html=True)

                # --- Top Comments ---
                st.subheader("Top Positive & Negative Comments")
                col_top1, col_top2 = st.columns(2)
                with col_top1:
                    st.markdown("#### Top Positive Comments (by Score)")
                    pos_comments = analyzed_df[analyzed_df['sentiment'] == 'positive'].sort_values(by='score', ascending=False)
                    if not pos_comments.empty:
                        for i, row in pos_comments.head(3).iterrows():
                            st.success(f"**Score: {row['score']}** | `{row['text'][:150].strip()}...`")
                    else:
                        st.info("No positive comments found.")

                with col_top2:
                    st.markdown("#### Top Negative Comments (by Score)")
                    neg_comments = analyzed_df[analyzed_df['sentiment'] == 'negative'].sort_values(by='score', ascending=False)
                    if not neg_comments.empty:
                        for i, row in neg_comments.head(3).iterrows():
                            st.error(f"**Score: {row['score']}** | `{row['text'][:150].strip()}...`")
                    else:
                        st.info("No negative comments found.")
            else:
                st.warning("No data returned. The post may have no comments or analysis failed.")
        else:
            st.warning("⚠️ Please enter a Reddit Post URL.")

# --- TAB 1: Overview Dashboard ---
with tab1:
    st.markdown('<h2 class="section-header">📊 Overview Dashboard</h2>', unsafe_allow_html=True)
    st.write(f"Displaying results for **{len(filtered_df)}** posts based on current filters.")

    if not len(filtered_df):
        st.warning("No data matches the current filters.")
    else:
        # KPIs
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f'<div class="kpi-box"><div class="kpi-title">📄 Total Posts</div><div class="kpi-value">{len(filtered_df)}</div></div>', unsafe_allow_html=True)
        with col2:
            pos_rate = (filtered_df['sentiment'].str.lower() == 'positive').mean()
            st.markdown(f'<div class="kpi-box"><div class="kpi-title">😊 Positive %</div><div class="kpi-value">{pos_rate*100:.1f}%</div></div>', unsafe_allow_html=True)
        with col3:
            neg_rate = (filtered_df['sentiment'].str.lower() == 'negative').mean()
            st.markdown(f'<div class="kpi-box"><div class="kpi-title">☹️ Negative %</div><div class="kpi-value">{neg_rate*100:.1f}%</div></div>', unsafe_allow_html=True)
        with col4:
            neu_rate = (filtered_df['sentiment'].str.lower() == 'neutral').mean()
            st.markdown(f'<div class="kpi-box"><div class="kpi-title">😐 Neutral %</div><div class="kpi-value">{neu_rate*100:.1f}%</div></div>', unsafe_allow_html=True)

        # Charts
        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            if 'sentiment' in filtered_df.columns:
                sent_counts = filtered_df['sentiment'].value_counts()
                fig_pie = px.pie(
                    values=sent_counts.values, names=sent_counts.index,
                    title='📊 Sentiment Distribution', hole=0.3,
                    color=sent_counts.index,
                    color_discrete_map={'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#f39c12'}
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(style_fig(fig_pie), use_container_width=True)

        with col_chart2:
             if 'subreddit' in filtered_df.columns and not filtered_df['subreddit'].isnull().all():
                 sub_counts = filtered_df['subreddit'].value_counts().nlargest(10) # Get top 10
                 if not sub_counts.empty:
                     fig_sub = px.bar(
                         x=sub_counts.values, y=sub_counts.index, orientation='h',
                         title='🧭 Top 10 Subreddits',
                         color=sub_counts.values, color_continuous_scale=px.colors.sequential.Viridis
                     )
                     fig_sub.update_layout(yaxis={'categoryorder':'total ascending'}, coloraxis_showscale=False)
                     st.plotly_chart(style_fig(fig_sub), use_container_width=True)
                 else:
                     st.info("No subreddit data available for the current filter.")


        # Time Series
        if 'created_date' in filtered_df.columns and not filtered_df['created_date'].isnull().all():
            ts_df = filtered_df.dropna(subset=['created_date']).copy()
            if len(ts_df) > 1: # Need at least 2 points for a line
                ts_df['date'] = ts_df['created_date'].dt.to_period('D').astype(str) # Group by day
                daily_sentiment = ts_df.groupby(['date', 'sentiment']).size().reset_index(name='count')

                if not daily_sentiment.empty:
                    fig_ts = px.area(
                        daily_sentiment, x='date', y='count', color='sentiment',
                        title='📈 Sentiment Trends Over Time',
                        color_discrete_map={'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#f39c12'}
                    )
                    st.plotly_chart(style_fig(fig_ts), use_container_width=True)
                else:
                    st.info("Not enough data points for time series analysis with current filters.")
            else:
                 st.info("Not enough data points for time series analysis with current filters.")


# --- TAB 2: Model Training & Evaluation ---
with tab2:
    st.markdown('<h2 class="section-header">🧠 Model Training & Evaluation</h2>', unsafe_allow_html=True)
    

    with st.container():
        st.markdown('<div class="center-container">', unsafe_allow_html=True)
        available_model_names = list(get_available_models().keys())
        # Use session state to remember selected models
        if 'selected_models' not in st.session_state:
             st.session_state['selected_models'] = [
                 'Logistic Regression'
             ]

        selected_models = st.multiselect(
            '🤖 Select Models to Train',
            available_model_names,
            default=st.session_state['selected_models']
        )
        st.session_state['selected_models'] = selected_models # Update session state

        train_btn = st.button('🚀 Train Selected Models', type='primary')
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Training Logic ---
    if train_btn:
        if not selected_models:
            st.warning('⚠️ Please select at least one model to train.')
        else:
            progress_bar = st.progress(0)
            status_placeholder = st.empty()
            results = []

            for i, model_name in enumerate(selected_models):
                status_placeholder.info(f'🔄 Training {model_name}...')
                result = train_model(base_df, model_name, text_col, 'sentiment')

                if result:
                    results.append(result)
                    save_model(result, model_name) # Save the trained model
                    status_placeholder.success(f'✅ {model_name} trained successfully!')
                else:
                    status_placeholder.error(f'❌ Failed to train {model_name}. Check data and logs.')

                progress_bar.progress((i + 1) / len(selected_models))

            st.session_state['model_results'] = results # Store results

            if results:
                st.success(f'🎉 Successfully trained {len(results)} models!')
            else:
                st.error("Model training failed for all selected models.")

    # --- Display Results ---
    if 'model_results' in st.session_state and st.session_state['model_results']:
        results = st.session_state['model_results']

        st.markdown('### 📊 Model Performance Comparison')
        comparison_data = []
        for result in results:
            comparison_data.append({
                'Model': result['model_name'],
                'Accuracy': result['accuracy'], # Keep as float for sorting
                'Precision': result['precision'],
                'Recall': result['recall'],
                'F1-Score': result['f1_score']
            })

        comparison_df = pd.DataFrame(comparison_data).sort_values('Accuracy', ascending=False).reset_index(drop=True)

        # Highlight best model
        best_model_name = comparison_df.iloc[0]['Model']

        # Format numbers for display after sorting
        display_df = comparison_df.copy()
        for col in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
            display_df[col] = display_df[col].map('{:.3f}'.format)

        st.dataframe(display_df, use_container_width=True)

        st.markdown(
             f'<div class="center-container"><h3>🏆 Best Model: <span style="color:#ff6b35;">{best_model_name}</span></h3><p>Highest accuracy: {comparison_df.iloc[0]["Accuracy"]:.3f}</p></div>',
             unsafe_allow_html=True
         )


        col_graph1, col_graph2 = st.columns(2)
        with col_graph1:
            # Bar chart for Accuracy
            fig_metrics = px.bar(
                comparison_df, x='Model', y='Accuracy',
                title='📈 Model Accuracy Comparison',
                color='Accuracy', color_continuous_scale=px.colors.sequential.Viridis
            )
            fig_metrics.update_layout(yaxis_range=[0,1])
            fig_metrics.update_xaxes(tickangle=30)
            st.plotly_chart(style_fig(fig_metrics), use_container_width=True)

        with col_graph2:
            # Confusion Matrix for Best Model
            best_result = next((r for r in results if r['model_name'] == best_model_name), None)
            if best_result:
                cm = best_result['confusion_matrix']
                labels = best_result['labels']
                fig_cm = px.imshow(
                    cm, text_auto=True, aspect='auto',
                    title=f'🎯 Confusion Matrix - {best_model_name}',
                    labels=dict(x='Predicted', y='Actual'),
                    x=labels, y=labels,
                    color_continuous_scale=px.colors.sequential.Blues
                )
                st.plotly_chart(style_fig(fig_cm), use_container_width=True)

        # Detailed Report Expander
        if best_result:
             with st.expander(f'📋 Detailed Classification Report for {best_model_name}'):
                 st.code(best_result['classification_report'])

        # Rationale: why best model is best vs others
        if results:
            best = max(results, key=lambda r: r['accuracy'])
            others = [r for r in results if r['model_name'] != best['model_name']]
            rationale = f"{best['model_name']} achieved the highest accuracy ({best['accuracy']:.3f}) and F1 ({best['f1_score']:.3f}), indicating better generalization on TF-IDF features."
            if others:
                weaknesses = []
                for r in others:
                    reasons = []
                    if r['accuracy'] + 0.02 < best['accuracy']:
                        reasons.append('lower accuracy')
                    if r['f1_score'] + 0.02 < best['f1_score']:
                        reasons.append('lower F1')
                    model_note = ', '.join(reasons) if reasons else 'slightly less consistent performance'
                    weaknesses.append(f"{r['model_name']}: {model_note} on sparse high-dimensional text features.")
                rationale += "\n\nWhy others are not the best:\n- " + "\n- ".join(weaknesses)
            st.markdown("### 🧾 Model Selection Rationale")
            st.write(rationale)

            # Get AI-generated theoretical comparison
            st.markdown("### 🤖 Model Analysis")
            with st.spinner("Generating analysis of models..."):
                model_theory = get_ai_model_comparison(selected_models)
                
            if model_theory:
                # Create DataFrame for display
                theory_rows = []
                for model_name, details in model_theory.items():
                    theory_rows.append({
                        'Model': model_name,
                        **details
                    })
                theory_df = pd.DataFrame(theory_rows)

                # Attach accuracy as a row (after transpose) via a column before transpose
                acc_map = {r['model_name']: r['accuracy'] for r in results}
                # Resolve potential name mismatches between AI labels and trained model names
                def _normalize(name):
                    try:
                        s = str(name).lower()
                        s = re.sub(r"[^a-z0-9 ]+", " ", s)
                        s = re.sub(r"\s+", " ", s).strip()
                        return s
                    except Exception:
                        return str(name).lower()

                # Build alias lookup including common synonyms
                alias = { _normalize(k): k for k in acc_map.keys() }
                # Add manual aliases for frequent AI wording
                for kn_alias in [
                    'knn', 'k nn', 'k nearest neighbors', 'k nearest neighbour', 'k nearest neighbor'
                ]:
                    alias.setdefault(kn_alias, 'KNN (k=5)')

                def _map_model_name(ai_label):
                    n = _normalize(ai_label)
                    if n in alias:
                        return alias[n]
                    # token contains matching (e.g., contains 'knn')
                    if 'knn' in n or 'k nearest' in n:
                        return 'KNN (k=5)'
                    return ai_label

                # Format as percentage with one decimal place, e.g., 82.8%
                def _fmt_pct(v):
                    try:
                        if v is None or (isinstance(v, float) and pd.isna(v)):
                            return None
                        return f"{float(v) * 100:.1f}%"
                    except Exception:
                        return None
                theory_df['Accuracy or R2 score'] = theory_df['Model'].map(_map_model_name).map(acc_map).map(_fmt_pct)

                # Clean symbols/brackets and truncate to short phrases (<= 7 words)
                def _clean_and_truncate(val, max_words=7):
                    try:
                        s = str(val)
                        # Replace colons with space to avoid fragments like key: value
                        s = re.sub(r"\s*:\s*", " ", s)
                        # Remove any remaining non-alphanumeric (keep spaces, %, +, -)
                        s = re.sub(r"[^A-Za-z0-9%+\- ]+", " ", s)
                        # Collapse whitespace
                        s = re.sub(r"\s+", " ", s).strip()
                        words = s.split()
                        return ' '.join(words[:max_words])
                    except Exception:
                        return val

                for col in theory_df.columns:
                    if col not in ['Model', 'accuracy or r2 score', 'Accuracy or R2 score']:
                        theory_df[col] = theory_df[col].apply(_clean_and_truncate)

                # Display the AI-generated comparison (transpose: fields as rows, models as columns)
                display_df = theory_df.set_index('Model').T
                # Dynamic height to avoid extra empty rows
                _row_height = 38
                _header_extra = 40
                _rows = int(display_df.shape[0])
                _height = max(120, min(900, _rows * _row_height + _header_extra))
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    height=_height
                )
                
                # Add download button for the theoretical comparison
                theory_csv = theory_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    '📥 Download AI Analysis',
                    theory_csv,
                    file_name='model_theoretical_analysis.csv',
                    mime='text/csv'
                )

        # Predicted vs Actual bar charts for each model (counts per class)
        st.markdown('### 📊 Actual vs Predicted Counts (per model)')
        def _ensure_predictions_for_result(result_item):
            """Backfill y_test/y_pred if missing by re-splitting and predicting with the saved pipeline.
            Uses the same split logic as training (random_state=42, stratify) and excludes 'neutral'."""
            try:
                if 'y_test' in result_item and 'y_pred' in result_item and len(result_item['y_test']) and len(result_item['y_pred']):
                    return result_item
                # Recompute using current base_df
                df_filtered_tmp = base_df[base_df['sentiment'] != 'neutral'].copy()
                if text_col not in df_filtered_tmp.columns:
                    return result_item
                X_tmp = df_filtered_tmp[text_col].fillna('')
                y_tmp = df_filtered_tmp['sentiment']
                if y_tmp.nunique() < 2:
                    return result_item
                X_tr, X_te, y_tr, y_te = train_test_split(X_tmp, y_tmp, test_size=0.2, random_state=42, stratify=y_tmp)
                pipe = result_item.get('pipeline')
                if pipe is None:
                    return result_item
                y_pred_te = pipe.predict(X_te)
                result_item['y_test'] = pd.Series(y_te).reset_index(drop=True)
                result_item['y_pred'] = pd.Series(y_pred_te).reset_index(drop=True)
                return result_item
            except Exception:
                # Keep original result if any failure occurs
                return result_item

        for r in results:
            r = _ensure_predictions_for_result(r)
            if 'y_test' in r and 'y_pred' in r and len(r['y_test']) and len(r['y_pred']):
                try:
                    # Align lengths and sanitize
                    y_true = pd.Series(r['y_test']).astype(str).reset_index(drop=True)
                    y_hat = pd.Series(r['y_pred']).astype(str).reset_index(drop=True)
                    n = int(min(len(y_true), len(y_hat)))
                    if n == 0:
                        raise ValueError('empty predictions')
                    y_true = y_true.iloc[:n]
                    y_hat = y_hat.iloc[:n]

                    # Count positives/negatives
                    classes = ['positive', 'negative']
                    actual_counts = [int((y_true.str.lower() == c).sum()) for c in classes]
                    pred_counts = [int((y_hat.str.lower() == c).sum()) for c in classes]

                    # Prepare dataframes for prettier bars
                    df_actual = pd.DataFrame({'Class': classes, 'Count': actual_counts})
                    df_pred = pd.DataFrame({'Class': classes, 'Count': pred_counts})

                    left_col, right_col = st.columns(2)

                    with left_col:
                        fig_actual = px.bar(
                            df_actual, x='Class', y='Count', color='Class',
                            title=f"Actual Counts - {r['model_name']}",
                            color_discrete_map={'positive': '#2ecc71', 'negative': '#e74c3c'}
                        )
                        fig_actual.update_traces(text=df_actual['Count'], textposition='outside', marker_line_width=1, marker_line_color='#444')
                        fig_actual.update_layout(
                            xaxis_title='Class', yaxis_title='Count',
                            xaxis=dict(categoryorder='array', categoryarray=classes),
                            bargap=0.25
                        )
                        st.plotly_chart(style_fig(fig_actual), use_container_width=True)

                    with right_col:
                        fig_pred = px.bar(
                            df_pred, x='Class', y='Count', color='Class',
                            title=f"Predicted Counts - {r['model_name']}",
                            color_discrete_map={'positive': '#2ecc71', 'negative': '#e74c3c'}
                        )
                        fig_pred.update_traces(text=df_pred['Count'], textposition='outside', marker_line_width=1, marker_line_color='#444')
                        fig_pred.update_layout(
                            xaxis_title='Class', yaxis_title='Count',
                            xaxis=dict(categoryorder='array', categoryarray=classes),
                            bargap=0.25
                        )
                        st.plotly_chart(style_fig(fig_pred), use_container_width=True)
                except Exception:
                    st.info(f"Could not plot Predicted vs Actual for {r['model_name']}.")
            else:
                st.info(f"Predicted vs Actual unavailable for {r['model_name']}. Please retrain to view this plot.")

        # Download Button for Results
        csv_data = comparison_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            '📥 Download Comparison Report', csv_data,
            file_name='model_comparison.csv', mime='text/csv'
        )

    elif train_btn: # Show if training was attempted but failed
         st.warning("No model results to display. Training might have failed.")


# --- TAB 3: Insights & Exports ---
with tab3:
    st.markdown('<h2 class="section-header">💡 Insights & Exports</h2>', unsafe_allow_html=True)
    st.write(f"Displaying insights for **{len(filtered_df)}** posts based on current filters.")

    if not len(filtered_df):
        st.warning("No data matches the current filters for insights.")
    else:
        # Word Clouds
        st.markdown('### ☁️ Word Clouds (Top 100 words)')
        col_wc1, col_wc2 = st.columns(2)

        # Function to generate word cloud
        @st.cache_data(show_spinner=False)
        def generate_wordcloud(text_blob, sentiment):
             if len(text_blob.strip()) > 10:
                  # Use clean_text again to ensure only words remain
                  # Create a single-item series for vectorized operation
                  cleaned_blob = clean_text(pd.Series([text_blob])).iloc[0]
                  wc = WordCloud(
                       width=600, height=300,
                       # --- MODIFIED: Force black background, white text for consistency ---
                       background_color='#000000',
                       colormap='viridis' if sentiment == 'positive' else 'Reds',
                       max_words=100 # Limit words
                  ).generate(cleaned_blob)
                  return wc.to_array()
             return None

        for sentiment, col in [('positive', col_wc1), ('negative', col_wc2)]:
            with col:
                subset = filtered_df[filtered_df['sentiment'].str.lower() == sentiment]
                if len(subset) > 0 and text_col in subset.columns:
                    text_blob = ' '.join(subset[text_col].astype(str).fillna('').tolist())
                    wc_image = generate_wordcloud(text_blob, sentiment)
                    if wc_image is not None:
                         st.image(wc_image, caption=f'{sentiment.title()} WordCloud', use_container_width=True)
                    else:
                         st.info(f'Not enough {sentiment} text for wordcloud.')
                else:
                    st.info(f'No {sentiment} data available.')

        # Keyword Analysis
        st.markdown('### 🔍 Keyword Analysis')
        if text_col in filtered_df.columns:
            # Use clean_text from preprocessor
            cleaned_text_list = clean_text(filtered_df[text_col].astype(str)).tolist()
            all_text = ' '.join(cleaned_text_list)
            # Basic word count, filter short words
            words = [w for w in all_text.split() if len(w) > 3]
            if words:
                word_freq = pd.Series(words).value_counts().nlargest(20) # Get top 20
                # Use a treemap for variety
                kw_df = pd.DataFrame({'keyword': word_freq.index.tolist(), 'freq': word_freq.values.tolist()})
                fig_words_tree = px.treemap(
                    kw_df, path=['keyword'], values='freq', title='🔤 Top Keywords (Treemap)'
                )
                st.plotly_chart(style_fig(fig_words_tree), use_container_width=True)
            else:
                st.info("Not enough text data to generate keyword analysis.")


        # Engagement Analysis
        if 'score' in filtered_df.columns and 'num_comments' in filtered_df.columns:
            st.markdown('### 📊 Engagement Analysis')
            # Filter out extreme outliers for better visualization if needed (optional)
            df_engagement = filtered_df.dropna(subset=['score', 'num_comments'])
            if len(df_engagement):
                 # --- MODIFIED: Added 3D Scatter Plot ---
                 st.markdown("#### 3D Engagement Plot")
                 fig_engagement_3d = px.scatter_3d(
                     df_engagement, 
                     x='score', y='num_comments', z='sentiment', # Use sentiment for Z-axis
                     color='sentiment',
                     hover_data=['title'] if 'title' in df_engagement.columns else None,
                     title='📣 3D Engagement: Score vs Comments vs Sentiment',
                     color_discrete_map={'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#f39c12'},
                     opacity=0.7
                 )
                 # Apply black background styling
                 st.plotly_chart(style_fig(fig_engagement_3d), use_container_width=True)
                 
                 # --- Original 2D Scatter Plot ---
                 st.markdown("#### 2D Engagement Plot")
                 fig_engagement_2d = px.scatter(
                     df_engagement, x='score', y='num_comments', color='sentiment',
                     hover_data=['title'] if 'title' in df_engagement.columns else None,
                     title='📣 2D Engagement: Score vs Comments',
                     color_discrete_map={'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#f39c12'},
                     opacity=0.7
                 )
                 st.plotly_chart(style_fig(fig_engagement_2d), use_container_width=True)
            else:
                 st.info("Not enough engagement data (score, num_comments) available.")


        # Data Export
        st.markdown('### 📥 Data Export')
        csv_export = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            '📊 Download Filtered Data', csv_export,
            file_name='filtered_reddit_data.csv', mime='text/csv'
        )

        # Export Model Results (if available)
        if 'model_results' in st.session_state and st.session_state['model_results']:
             comparison_data = []
             for result in st.session_state['model_results']:
                   comparison_data.append({
                       'Model': result['model_name'], 'Accuracy': result['accuracy'],
                       'Precision': result['precision'], 'Recall': result['recall'],
                       'F1-Score': result['f1_score']
                   })
             comparison_df_export = pd.DataFrame(comparison_data)
             csv_model_export = comparison_df_export.to_csv(index=False).encode('utf-8')
             st.download_button(
                  '🤖 Download Model Results', csv_model_export,
                  file_name='model_results.csv', mime='text/csv'
             )


# --- TAB 4: Real-time Comment Analyzer ---
with tab4:
    st.markdown('<h2 class="section-header">💬 Real-time Comment Analyzer</h2>', unsafe_allow_html=True)
    st.markdown("Paste the URL of any Reddit post to fetch all its comments and analyze their sentiment in real-time.")

    # 1. Get Reddit Instance (reuse the cached one)
    reddit_instance_for_comments = get_reddit_instance()
    if not reddit_instance_for_comments:
        st.error("Could not connect to Reddit API for comment analysis. Check credentials.")
    else:
        # 2. User Input
        url = st.text_input("Enter a Reddit Post URL:", placeholder="https://www.reddit.com/r/...", key="comment_url_input") # Unique key
        
        # Initialize session state for comment analysis
        if 'last_analyzed_url' not in st.session_state:
            st.session_state['last_analyzed_url'] = None
        if 'comment_analysis_results' not in st.session_state:
            st.session_state['comment_analysis_results'] = None
        if 'comment_analysis_error' not in st.session_state:
            st.session_state['comment_analysis_error'] = None
        
        # Clear previous results if URL changed
        if url != st.session_state.get('last_analyzed_url'):
            st.session_state['comment_analysis_results'] = None
            st.session_state['comment_analysis_error'] = None

        if st.button("Analyze Comments", key="analyze_comment_button"):
            if url:
                # Basic local validation to avoid calling the fetch routine for non-Reddit URLs
                url_lower = url.strip().lower()
                if ('reddit.com' not in url_lower) and ('redd.it' not in url_lower):
                    # Immediately store the invalid-url error and clear any previous results
                    st.session_state['last_analyzed_url'] = url
                    st.session_state['comment_analysis_results'] = None
                    st.session_state['comment_analysis_error'] = "❌ Invalid Reddit URL. Please paste a Reddit post URL (e.g. https://www.reddit.com/r/...)."
                else:
                    # 3. Run Analysis (using the function we integrated)
                    analyzed_df, error = fetch_and_analyze_comments(reddit_instance_for_comments, url)

                    # Store results in session state (always, even if error)
                    st.session_state['last_analyzed_url'] = url
                    st.session_state['comment_analysis_results'] = analyzed_df
                    st.session_state['comment_analysis_error'] = error
            else:
                st.warning("⚠️ Please enter a Reddit Post URL.")

        # Display results from session state (only if URL matches and is not empty)
        if url and st.session_state.get('last_analyzed_url') == url:
            error = st.session_state.get('comment_analysis_error')
            analyzed_df = st.session_state.get('comment_analysis_results')
            
            if error:
                st.error(error)
            elif analyzed_df is not None:
                    st.success(f"Analysis complete! Found and analyzed {len(analyzed_df)} comments.")

                    # 4. Display Results

                    # --- Overall Sentiment ---
                    st.subheader("Overall Comment Sentiment")
                    sentiment_counts = analyzed_df['sentiment'].value_counts()
                    sentiment_df = pd.DataFrame({'sentiment': sentiment_counts.index, 'count': sentiment_counts.values})
                    # Pie chart with count and percentage
                    total_comments = int(sentiment_df['count'].sum())
                    fig_comment_pie = px.pie(
                        sentiment_df, values='count', names='sentiment',
                        title='Comment Sentiment Distribution',
                        hole=0.3,
                        color='sentiment',
                        color_discrete_map={'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#f39c12'}
                    )
                    # Show percentages as labels
                    fig_comment_pie.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(style_fig(fig_comment_pie), use_container_width=True)

                    # Numeric summary: counts and percentages
                    pos = int(sentiment_df.loc[sentiment_df['sentiment']=='positive', 'count'].sum()) if 'positive' in sentiment_df['sentiment'].values else 0
                    neg = int(sentiment_df.loc[sentiment_df['sentiment']=='negative', 'count'].sum()) if 'negative' in sentiment_df['sentiment'].values else 0
                    neu = int(sentiment_df.loc[sentiment_df['sentiment']=='neutral', 'count'].sum()) if 'neutral' in sentiment_df['sentiment'].values else 0
                    pos_p = (pos/total_comments*100) if total_comments else 0
                    neg_p = (neg/total_comments*100) if total_comments else 0
                    neu_p = (neu/total_comments*100) if total_comments else 0

                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.markdown(f'<div class="kpi-box"><div class="kpi-title">Positive</div><div class="kpi-value">{pos} ({pos_p:.1f}%)</div></div>', unsafe_allow_html=True)
                    with c2:
                        st.markdown(f'<div class="kpi-box"><div class="kpi-title">Negative</div><div class="kpi-value">{neg} ({neg_p:.1f}%)</div></div>', unsafe_allow_html=True)
                    with c3:
                        st.markdown(f'<div class="kpi-box"><div class="kpi-title">Neutral</div><div class="kpi-value">{neu} ({neu_p:.1f}%)</div></div>', unsafe_allow_html=True)

                    # --- Top Comments ---
                    st.subheader("Top Positive & Negative Comments")
                    col_top1, col_top2 = st.columns(2)
                    with col_top1:
                        st.markdown("#### Top Positive Comments (by Score)")
                        pos_comments = analyzed_df[analyzed_df['sentiment'] == 'positive'].sort_values(by='score', ascending=False)
                        if not pos_comments.empty:
                             for i, row in pos_comments.head(3).iterrows():
                                 st.success(f"**Score: {row['score']}** | `{row['text'][:150].strip()}...`")
                        else:
                             st.info("No positive comments found.")

                    with col_top2:
                        st.markdown("#### Top Negative Comments (by Score)")
                        neg_comments = analyzed_df[analyzed_df['sentiment'] == 'negative'].sort_values(by='score', ascending=False)
                        if not neg_comments.empty:
                             for i, row in neg_comments.head(3).iterrows():
                                 st.error(f"**Score: {row['score']}** | `{row['text'][:150].strip()}...`")
                        else:
                             st.info("No negative comments found.")


                    # --- Data Table ---
                    with st.expander("View all analyzed comments"):
                        st.dataframe(analyzed_df[['sentiment', 'score', 'author', 'text']])
            else:
                st.warning("No data returned. The post may have no comments or analysis failed.")


# -------------------------------
# Footer (From Original Dashboard)
# -------------------------------
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #7f8c8d; padding: 2rem;'>
        <p>🤖 Next-Gen Reddit Sentiment AI Dashboard | Powered by Streamlit, Plotly & Scikit-learn</p>
        <p>Built for AI Influencer Sentiment Analysis | Real-time ML & Comment Analysis</p>
    </div>
    """,
    unsafe_allow_html=True
)

