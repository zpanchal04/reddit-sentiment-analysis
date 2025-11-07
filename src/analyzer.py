# --- VADER Sentiment Analyzer ---
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# --- ML / sklearn Imports ---
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline
import pandas as pd

# Classifiers
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier, PassiveAggressiveClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


def get_vader_sentiment(text):
    """
    Analyzes a text string and returns a sentiment label ('positive', 'negative', 'neutral')
    based on VADER's compound score.
    """
    if not isinstance(text, str) or not text.strip():
        return 'neutral'  # Return neutral for empty strings

    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = analyzer.polarity_scores(text)
    compound_score = sentiment_scores['compound']

    if compound_score >= 0.05:
        return 'positive'
    elif compound_score <= -0.05:
        return 'negative'
    else:
        return 'neutral'


def run_model_pipeline(df, model_name="naive_bayes_model"):
    """
    Filters data, builds a model pipeline, trains, and evaluates it.

    Args:
        df (pd.DataFrame): The DataFrame with 'tokens_joined' and 'vader_sentiment'
        model_name (str): The name of the model to run. Must match one of the known keys.

    Returns:
        dict: A dictionary containing accuracy, f1_score, and the full report.
    """

    # --- 1. Filter Data ---
    # We only train on 'positive' and 'negative' to get a clear accuracy metric
    if 'vader_sentiment' not in df.columns or 'tokens_joined' not in df.columns:
        return {
            "model_name": model_name,
            "accuracy": 0,
            "f1_score": 0,
            "report": "Required columns ('vader_sentiment' and 'tokens_joined') not found in dataframe."
        }

    df_filtered = df[df['vader_sentiment'] != 'neutral'].copy()

    if len(df_filtered) < 10:
        return {
            "model_name": model_name,
            "accuracy": 0,
            "f1_score": 0,
            "report": "Not enough positive/negative data to train a model."
        }

    # --- 2. Define X and y ---
    X = df_filtered['tokens_joined']
    y = df_filtered['vader_sentiment']

    if y.nunique() < 2:
        return {
            "model_name": model_name,
            "accuracy": 0,
            "f1_score": 0,
            "report": f"Only one class ('{y.unique()[0]}') found. Cannot train."
        }

    # --- 3. Train/Test Split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- 4. Select Model (Model Factory) ---
    name = model_name.lower()
    if name in ["naive_bayes_model", "naive_bayes_multinomial_model"]:
        model = MultinomialNB()
    elif name == "logistic_regression_model":
        model = LogisticRegression(max_iter=1000)
    elif name in ["support_vector_machine_svm_model", "support_vector_machine_(svm)_model"]:
        model = SVC(kernel='linear', probability=True)
    elif name == "support_vector_machine_linearsvc_model":
        model = LinearSVC()
    elif name == "svc_rbf_kernel_model":
        model = SVC(kernel='rbf', probability=True)
    elif name == "random_forest_model":
        model = RandomForestClassifier()
    elif name == "extra_trees_model":
        model = ExtraTreesClassifier()
    elif name == "gradient_boosting_model":
        model = GradientBoostingClassifier()
    elif name == "adaboost_model":
        model = AdaBoostClassifier()
    elif name == "decision_tree_model":
        model = DecisionTreeClassifier()
    elif name == "knn_k=5_model":
        model = KNeighborsClassifier(n_neighbors=5)
    elif name == "ridge_classifier_model":
        model = RidgeClassifier()
    elif name == "sgd_classifier_model":
        model = SGDClassifier(random_state=42, max_iter=1000)
    elif name == "passive_aggressive_model":
        model = PassiveAggressiveClassifier()
    else:
        return {
            "model_name": model_name,
            "accuracy": 0,
            "f1_score": 0,
            "report": f"Unknown model name provided: {model_name}"
        }

    # --- 5. Build pipeline (TF-IDF + classifier) ---
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=2000, stop_words='english', ngram_range=(1, 2))),
        ('classifier', model)
    ])

    # --- 6. Train ---
    pipeline.fit(X_train, y_train)

    # --- 7. Evaluate ---
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    report = classification_report(y_test, y_pred, zero_division=0)

    return {
        "model_name": model_name,
        "accuracy": accuracy,
        "f1_score": f1,
        "report": report
    }

