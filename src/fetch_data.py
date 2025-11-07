import os
import json
import praw
import praw.exceptions
from dotenv import load_dotenv
from datetime import datetime
import time

# Load environment variables from a .env file
load_dotenv()

# --- Configuration ---
# Define the path where the raw JSON data will be saved
RAW_DATA_PATH = 'data/raw/reddit-posts.json'

# Search terms for Reddit submissions
SEARCH_QUERIES = ['lilmiquela', 'imma-gram', 'AI Influencer']

# --- NEW REQUIREMENT: Set target for NEW posts ---
POST_LIMIT = 5000

# Subreddit to search within ('all' searches across all of Reddit)
SUBREDDIT = 'all'

# --- Reddit Authentication ---

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
        if reddit.read_only:
             print("Warning: PRAW is in read-only mode. Check credentials.")
        else:
             print(f"✅ Successfully connected and authenticated as: {reddit.user.me()}")
        return reddit
    except Exception as e:
        print(f"❌ Error connecting to Reddit API. Check environment variables (.env): {e}")
        return None

# --- Data Handling ---

def load_existing_data(filepath):
    """Loads existing posts from the JSON file and returns a list of posts and a set of IDs."""
    if not os.path.exists(filepath):
        print("No existing data file found. Starting fresh.")
        return [], set() # Return empty list (data) and empty set (IDs)
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Create a set of all existing post IDs for fast lookup
        existing_ids = set(post['id'] for post in data if 'id' in post)
        print(f"Loaded {len(data)} existing posts. Found {len(existing_ids)} unique IDs.")
        return data, existing_ids
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Could not read {filepath}. Starting fresh. Error: {e}")
        return [], set()


def save_data_to_json(new_posts, all_posts, filepath):
    """Saves the combined list of all posts (old + new) to the JSON file."""
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(all_posts, f, ensure_ascii=False, indent=4)
            
        print(f"\n💾 Successfully saved {len(new_posts)} new posts.")
        print(f"📦 Total posts in file: {len(all_posts)}")
    except Exception as e:
        print(f"❌ Error saving data to JSON: {e}")

# --- Data Fetching (UPDATED FOR NEW REQUIREMENT) ---

def fetch_reddit_data(reddit_instance, queries, limit, subreddit_name, existing_ids):
    """
    Fetches NEW Reddit submissions based on multiple queries, sorting by 'new'.
    Skips any IDs that are already in the existing_ids set.
    """
    if not reddit_instance:
        return []

    print(f"\n--- Starting data fetch (target: {limit} new posts) ---")
    print(f"--- Skipping {len(existing_ids)} posts already in database ---")
    
    submissions = set() # Use a set to track unique submission IDs found *in this run*
    data = []
    total_fetched = 0 # This will count NEW posts

    for query in queries:
        print(f"🔍 Searching for new posts matching: '{query}'")
        
        try:
            # --- MODIFIED: We only sort by 'new' to get the latest posts ---
            # We set limit=None to get as many as the API allows,
            # so we can find 'limit' number of *new* ones.
            results = reddit_instance.subreddit(subreddit_name).search(
                query,
                limit=None,
                sort='new'
            )

            for submission in results:
                # --- NEW REQUIREMENT: Check against existing IDs ---
                if submission.id not in existing_ids:
                    submissions.add(submission.id)
                    existing_ids.add(submission.id) # Add to set to avoid duplicates *in this run*
                    total_fetched += 1

                    # Collect key metadata
                    data.append({
                        'id': submission.id,
                        'title': submission.title,
                        'text': submission.selftext if submission.selftext else submission.url,
                        'subreddit': submission.subreddit.display_name,
                        'score': submission.score,
                        'num_comments': submission.num_comments,
                        'created_utc': submission.created_utc,
                        'created_date': str(datetime.fromtimestamp(submission.created_utc)),
                        'search_query': query,
                        'sort_type': 'new' # Hardcoded
                    })
                    
                    if total_fetched % 100 == 0:
                        print(f"   ... fetched {total_fetched} new posts")
                    
                    if total_fetched >= limit:
                        break # Stop searching this query

        except praw.exceptions.APIException as api_e:
            print(f"⚠️ PRAW API Error during search: {api_e}. Waiting 5 seconds...")
            time.sleep(5)
        except Exception as e:
            print(f"❌ An unexpected error occurred while processing query '{query}': {e}")
        
        if total_fetched >= limit:
            print(f"Hit total post limit of {limit}. Stopping all searches.")
            break # Stop searching other queries
            
        # Sleep briefly between queries to be kind to the API
        time.sleep(1) 

    print(f"--- Finished fetching. Total new posts collected: {len(data)} ---")
    return data # This function now ONLY returns the NEW posts

# --- Main Execution (UPDATED) ---

def main():
    """Orchestrates the incremental data collection process."""
    
    # 1. Get the Reddit instance
    reddit = get_reddit_instance()
    
    if not reddit:
        print("🛑 Cannot proceed without a valid Reddit instance.")
        return

    # 2. Load existing data
    existing_posts_list, existing_post_ids = load_existing_data(RAW_DATA_PATH)

    # 3. Fetch only NEW data
    new_posts_list = fetch_reddit_data(
        reddit,
        SEARCH_QUERIES,
        POST_LIMIT,
        SUBREDDIT,
        existing_post_ids # Pass in the set of IDs to skip
    )

    # 4. Save the combined data
    if new_posts_list:
        # Combine old and new lists
        all_posts_list = existing_posts_list + new_posts_list
        # Save
        save_data_to_json(new_posts_list, all_posts_list, RAW_DATA_PATH)
    else:
        print("🤷 No new data was collected in this run.")

if __name__ == "__main__":
    main()

