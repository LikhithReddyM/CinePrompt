from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from llama_cpp import Llama
import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm
import pickle
import argparse
from dotenv import load_dotenv
import os, json, openai, logging
from werkzeug.security import generate_password_hash, check_password_hash
import random
import string
from datetime import datetime
from personalization import MoviePersonalizer
import requests
import re

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session management
DATA_FILE = "data.json"
USERS_FILE = "users.json"
logging.basicConfig(level=logging.INFO)

# ---------- Data helpers ----------
def load_data():
    if not os.path.exists(DATA_FILE) or os.path.getsize(DATA_FILE)==0:
        with open(DATA_FILE,"w") as f: json.dump([],f)
    with open(DATA_FILE) as f: return json.load(f)

def save_or_update(record):
    data = load_data()
    # Always append new record instead of updating
    data.append(record)
    with open(DATA_FILE,"w") as f: json.dump(data,f,indent=4)

# ---------- User Management ----------
def load_users():
    if not os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'w') as f:
            json.dump([], f)
    with open(USERS_FILE) as f:
        return json.load(f)

def save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=4)

def generate_user_id():
    while True:
        # Generate 8-digit user ID
        user_id = ''.join(random.choices(string.digits, k=8))
        users = load_users()
        # Check if ID is unique
        if not any(user['user_id'] == user_id for user in users):
            return user_id

def get_user_by_username(username):
    users = load_users()
    return next((user for user in users if user['username'] == username), None)

# Initialize the embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize FAISS index
dimension = 384  # Dimension of all-MiniLM-L6-v2 embeddings
index = faiss.IndexFlatL2(dimension)

# Initialize the personalization system
personalizer = MoviePersonalizer.get_instance()
try:
    personalizer.load_model()
except:
    print("No saved personalization model found. Creating new one...")
    personalizer.load_data()
    personalizer.save_model()

def movie_row_to_document(row):
    """
    Converts a single movie row (as a dictionary) into a clean text document for embedding.
    """

    # Helper to safely get names from list of dicts
    def extract_names(items, key='name'):
        if isinstance(items, list):
            return ', '.join(item.get(key, '') for item in items if item.get(key))
        return ''
    
    # Extract fields
    title = row.get('title', '')
    poster_path = row.get('poster_path', '')
    id = row.get('id', '')
    original_title = row.get('original_title', '')
    genres = extract_names(row.get('genres', []))
    keywords = extract_names(row.get('keywords', []))
    overview = row.get('overview', '')
    tagline = row.get('tagline', '')
    release_date = row.get('release_date', '')
    runtime = row.get('runtime', '')
    production_companies = extract_names(row.get('production_companies', []))
    production_countries = extract_names(row.get('production_countries', []))
    spoken_languages = extract_names(row.get('spoken_languages', []))
    popularity = row.get('popularity', '')
    vote_average = row.get('vote_average', '')
    vote_count = row.get('vote_count', '')

    # Format into a document string
    doc = f"""
ID: {id}
Poster Path: {poster_path}
Title: {title}
Original Title: {original_title}
Release Date: {release_date}
Runtime: {runtime} minutes
Genres: {genres}
Keywords: {keywords}
Tagline: {tagline}
Overview: {overview}
Production Companies: {production_companies}
Production Countries: {production_countries}
Languages: {spoken_languages}
Popularity Score: {popularity}
Average Rating: {vote_average} (based on {vote_count} votes)
"""
    # Remove extra whitespace
    return doc.strip()

def save_embeddings_and_index(embeddings, index, df, documents):
    """Save embeddings, index, and documents to disk"""
    print("Saving embeddings and index...")
    # Create directory if it doesn't exist
    os.makedirs('saved_data', exist_ok=True)
    
    # Save embeddings
    with open('saved_data/embeddings.pkl', 'wb') as f:
        pickle.dump(embeddings, f)
    
    # Save FAISS index
    faiss.write_index(index, 'saved_data/faiss_index.bin')
    
    # Save documents and dataframe
    with open('saved_data/documents.pkl', 'wb') as f:
        pickle.dump(documents, f)
    
    df.to_pickle('saved_data/dataframe.pkl')
    print("Data saved successfully!")

def load_embeddings_and_index():
    """Load embeddings, index, and documents from disk"""
    print("Loading saved data...")
    try:
        # Load embeddings
        with open('saved_data/embeddings.pkl', 'rb') as f:
            embeddings = pickle.load(f)
        
        # Load FAISS index
        index = faiss.read_index('saved_data/faiss_index.bin')
        
        # Load documents and dataframe
        with open('saved_data/documents.pkl', 'rb') as f:
            documents = pickle.load(f)
        
        df = pd.read_pickle('saved_data/dataframe.pkl')
        print("Data loaded successfully!")
        return df, documents, embeddings, index
    
    except FileNotFoundError:
        print("No saved data found. Will generate new embeddings.")
        return None

# Load and process the movies dataset
def load_and_process_data():
    # Try to load saved data first
    saved_data = load_embeddings_and_index()
    if saved_data is not None:
        df, documents, embeddings, loaded_index = saved_data
        # Update the global index
        global index
        index = loaded_index
        return df, documents
    
    print("Loading dataset...")
    # Assuming the dataset is in the same directory
    df = pd.read_csv('dataset/merged_data.csv')
    
    print("Processing movies...")
    # Create documents for each movie
    documents = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Creating movie documents"):
        try:
            # Convert row to dictionary and create document
            doc = movie_row_to_document(row.to_dict())
            if doc and doc.strip():  # Check if document is not empty
                documents.append(doc)
        except Exception as e:
            print(f"Error processing row: {e}")
            continue
    
    if not documents:
        raise ValueError("No valid documents were created from the dataset")
    
    print("Creating embeddings...")
    # Create embeddings with progress bar
    embeddings = []
    for doc in tqdm(documents, desc="Generating embeddings"):
        try:
            embedding = embedding_model.encode([doc])[0]
            embeddings.append(embedding)
        except Exception as e:
            print(f"Error creating embedding: {e}")
            continue
    
    if not embeddings:
        raise ValueError("No valid embeddings were created")
    
    print("Building FAISS index...")
    # Convert to numpy array and add to index
    embeddings_array = np.array(embeddings).astype('float32')
    with tqdm(total=len(embeddings_array), desc="Indexing embeddings") as pbar:
        index.add(embeddings_array)
        pbar.update(len(embeddings_array))
    
    # Save the generated data
    save_embeddings_and_index(embeddings, index, df, documents)
    
    return df, documents

# Load data and create embeddings
df, documents = load_and_process_data()

def extract_keywords(query: str) -> str:
    """
    Use LLM to extract relevant keywords from the user query.
    """
    prompt = f"""Extract the most relevant keywords and phrases for finding movies based on this request: "{query}"
    Consider genres, themes, plot elements, emotions, time periods, cast(if any), and other relevant movie characteristics.
    Format your response as a comma-separated list of keywords and phrases.
    
    Example 1 - For query "I want a heartwarming movie about family":
    heartwarming, family relationships, emotional, family bonds, family drama, feel-good, domestic life
    
    Example 2 - For query "Show me sci-fi movies with time travel":
    science fiction, time travel, temporal paradox, futuristic, sci-fi action, alternate timelines
    
    Keywords for the given query:"""
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    
    keywords = response["choices"][0]["message"]["content"].strip()
    # print("\nExtracted keywords:", keywords)
    return keywords

# ---------- Helper function for TMDB poster ----------
def get_poster_by_tmdb_id(movie_id, api_key):
    if not api_key:
        print("Warning: TMDB_API_KEY not found.")
        return "No poster found (API key missing)"
    if not movie_id or not isinstance(movie_id, (int, str)) or not str(movie_id).isdigit():
         print(f"Warning: Invalid movie_id for TMDB lookup: {movie_id}")
         return "No poster found (Invalid ID)"
         
    url = f"https://api.themoviedb.org/3/movie/{movie_id}"
    params = {
        "api_key": api_key
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status() # Raise an exception for bad status codes
        data = response.json()

        if 'poster_path' in data and data['poster_path']:
            poster_path = data['poster_path']
            # Use w342 size for better performance
            return f"https://image.tmdb.org/t/p/w342{poster_path}"
        else:
            return "No poster found"
    except requests.exceptions.RequestException as e:
        print(f"Error fetching TMDB data for ID {movie_id}: {e}")
        return "No poster found (API error)"
    except Exception as e:
        print(f"Error processing TMDB data for ID {movie_id}: {e}")
        return "No poster found (Processing error)"

def recommend(query: str) -> list:
    """
    Recommend movies based on user query using FAISS and LLM.
    Returns a list of movie dictionaries.
    """
    # First, extract relevant keywords using LLM
    keywords = extract_keywords(query)
    
    # Combine original query with keywords for better semantic search
    enhanced_query = f"{query} {keywords}"
    print("\nEnhanced query:", enhanced_query)
    
    # Get query embedding with enhanced query
    query_embedding = embedding_model.encode([enhanced_query])
    
    # Search in FAISS
    k = 50  # Number of similar documents to retrieve
    distances, indices = index.search(np.array(query_embedding).astype('float32'), k)
    
    # Get relevant documents and movie IDs
    relevant_docs = []
    movie_ids = []
    doc_id_map = {}
    for i in indices[0]:
        doc = documents[i]
        lines = [line for line in doc.split('\n') if line.strip()]
        
        movie_id = None
        simplified_doc_lines = []
        title = "Unknown Title"
        for line in lines:
            if line.startswith('ID:'):
                try:
                    movie_id = int(line.split(':')[1].strip())
                except (ValueError, IndexError):
                    print(f"Warning: Could not parse ID from line: {line}")
                    continue
            if line.startswith('Title:'):
                title = line.split(':', 1)[1].strip()
                
            # Keep essential info for LLM context
            if any(key in line for key in ['ID:', 'Title:', 'Release Date:', 'Genres:', 'Overview:', 'Average Rating:']):
                 simplified_doc_lines.append(line)
                
        if movie_id is not None:
            if movie_id not in movie_ids: # Avoid duplicates
                movie_ids.append(movie_id)
                simplified_doc = '\n'.join(simplified_doc_lines)
                relevant_docs.append(simplified_doc)
                doc_id_map[simplified_doc] = movie_id # Map doc back to ID
            
    # Get personalized scores for the movies
    top_k_personalized = 15 # Consider top K after personalization
    if 'user_id' in session:
        user_id = session['user_id']
        personalized_scores = personalizer.get_personalized_scores(user_id, movie_ids)
        
        # Sort documents by personalized scores
        # Ensure scores and docs align correctly before sorting
        if len(personalized_scores) == len(relevant_docs):
             sorted_indices = np.argsort(personalized_scores)[::-1]
             relevant_docs = [relevant_docs[i] for i in sorted_indices[:top_k_personalized]]
        else:
             print(f"Warning: Mismatch between score count ({len(personalized_scores)}) and doc count ({len(relevant_docs)}). Skipping personalization sort.")
             relevant_docs = relevant_docs[:top_k_personalized]
    else:
        # If no user preferences, just take the top K by relevance
        relevant_docs = relevant_docs[:top_k_personalized]
    
    print(f"Number of relevant docs sent to LLM: {len(relevant_docs)}")
    # Create prompt for LLM
    prompt = f"""Based on the following movie information and the user's preference for "{query}" (with relevant aspects: {keywords}), \
    recommend the top 10 most relevant movies. For each movie, provide the ID, Title, Release Date, and a brief 1-2 line Summary based *only* on the provided 'Overview' field.\
    \
    Format EACH recommendation STRICTLY on a new line like this:\
    ID - Title, Release Date - Summary\
    \
    Example format:\
    278 - The Shawshank Redemption, 1994-09-23 - Framed in the 1940s for the double murder of his wife and her lover, upstanding banker Andy Dufresne begins a new life at the Shawshank prison...\
    238 - The Godfather, 1972-03-14 - Spanning the years 1945 to 1955, a chronicle of the fictional Italian-American Corleone crime family...\
    27205 - Inception, 2010-07-16 - Cobb, a skilled thief who commits corporate espionage by infiltrating the subconscious of his targets is offered a chance to regain his old life...\
    \
    Relevant movie information:\
    {chr(10).join(relevant_docs)}\
    \
    Recommendations:"""
    
    # Generate response from LLM
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role":"user","content": prompt}],
            temperature=0.5, # Slightly lower temperature for more consistent format
            max_tokens=500 # Increase max tokens if summaries are long
        )
        llm_output = response["choices"][0]["message"]["content"].strip()
        print("\nRecommendation output from LLM:")
        print(llm_output)
    except Exception as e:
        print(f"Error calling OpenAI: {e}")
        return [] # Return empty list on error

    # Parse LLM output and fetch posters
    recommendations_list = []
    for line in llm_output.split('\n'):
        if line.strip():
            match = re.match(r'^(\d+)\s*-\s*(.*?)(?:,\s*(\d{4}-\d{2}-\d{2}|Unknown year))?\s*-\s*(.*)$', line.strip())
            if match:
                movie_id_str, title, release_date, summary = match.groups()
                movie_id = int(movie_id_str) # Convert ID to int
                title = title.strip()
                release_date = release_date.strip() if release_date else 'Unknown Date'
                summary = summary.strip()
                
                print(f"Fetching poster for ID: {movie_id}")
                poster_url = get_poster_by_tmdb_id(movie_id, TMDB_API_KEY)
                
                recommendations_list.append({
                    'id': movie_id,
                    'title': title,
                    'release_date': release_date,
                    'summary': summary,
                    'poster_url': poster_url
                })
            else:
                print(f"Warning: Could not parse recommendation line: {line}")

    return recommendations_list

# ---------- Routes ----------
@app.route("/")
def home():
    if 'username' not in session:
        return redirect(url_for('login_page'))
    return render_template("index.html", username=session['username'])

@app.route("/login", methods=["GET"])
def login_page():
    if 'username' in session:
        return redirect(url_for('home'))
    return render_template("login.html")

@app.route("/login", methods=["POST"])
def login():
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        
        if not username or not password:
            return jsonify(status="error", message="Username and password are required"), 400
        
        users = load_users()
        user = next((user for user in users if user['username'] == username), None)
        
        if not user:
            return jsonify(status="error", message="Username not found"), 401
            
        if not check_password_hash(user['password'], password):
            return jsonify(status="error", message="Incorrect password"), 401
            
        session['username'] = username
        session['user_id'] = user['user_id']
        return jsonify(status="success", message="Login successful", redirect="/")
        
    except Exception as e:
        app.logger.error(f"Login error: {str(e)}")
        return jsonify(status="error", message="An error occurred during login"), 500

@app.route("/register", methods=["POST"])
def register():
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        
        if not username or not password:
            return jsonify(status="error", message="Username and password are required"), 400
            
        if len(username) < 3:
            return jsonify(status="error", message="Username must be at least 3 characters long"), 400
            
        if len(password) < 6:
            return jsonify(status="error", message="Password must be at least 6 characters long"), 400
        
        users = load_users()
        if any(user['username'] == username for user in users):
            return jsonify(status="error", message="Username already exists"), 400
        
        user_id = generate_user_id()
        new_user = {
            'username': username,
            'password': generate_password_hash(password),
            'user_id': user_id
        }
        
        users.append(new_user)
        save_users(users)
        
        # Verify the user was saved
        saved_users = load_users()
        if not any(user['username'] == username for user in saved_users):
            return jsonify(status="error", message="Failed to save user data"), 500
        
        # Automatically log in the user after registration
        session['username'] = username
        session['user_id'] = user_id
        
        # Redirect to preferences page for first-time users
        return jsonify(status="success", message="Registration successful", redirect=f"/preferences/{username}")
        
    except Exception as e:
        app.logger.error(f"Registration error: {str(e)}")
        return jsonify(status="error", message="An error occurred during registration"), 500

@app.route("/logout")
def logout():
    if 'user_id' in session:
        # Save user preferences before logging out
        try:
            personalizer.save_user_preferences(session['user_id'])
        except Exception as e:
            app.logger.error(f"Error saving preferences: {str(e)}")
    
    session.pop('username', None)
    session.pop('user_id', None)
    return redirect(url_for('home'))

@app.route("/preferences/<username>")
def preferences(username):
    if 'username' not in session or session['username'] != username:
        return redirect(url_for('login_page'))
    return render_template("preferences.html", username=username)

@app.route("/save-preferences", methods=["POST"])
def save_preferences():
    try:
        if 'username' not in session:
            return jsonify(status="error", message="User not logged in"), 401
            
        p = request.get_json(force=True)
        user_id = session['user_id']
        
        # Update user preferences in memory only (not saved to disk)
        if any(key in p for key in ['genres', 'favorite_movies', 'worst_movies', 'era']):
            preferences = {
                'genres': p.get('genres', []),
                'favorite_movies': p.get('favorite_movies', []),
                'worst_movies': p.get('worst_movies', []),
                'era': p.get('era', 'new')
            }
            personalizer.update_user_preferences(user_id, preferences)
            personalizer.save_user_preferences(user_id)  # Save to disk
        
        return jsonify(status="success", message="Preferences saved successfully")

    except Exception as e:
        app.logger.error("save-preferences error: %s", e)
        import traceback
        traceback.print_exc()
        return jsonify(status="error", message=str(e)), 500

@app.route("/submit-preferences", methods=["POST"])
def submit_preferences():
    try:
        if 'username' not in session:
            return jsonify(status="error", message="User not logged in"), 401
            
        p = request.get_json(force=True)
        username = session['username']
        user_id = session['user_id']
        
        # Get recommendations based on the prompt
        prompt = p.get("prompt", "").strip() or "Give me five great movies."
        suggestions_list = recommend(prompt) # This now returns a list of dicts
        
        print(f"\nUser: {username}")
        print(f"Prompt: {prompt}")
        print("Suggestions returned to frontend (structure):", [item.keys() for item in suggestions_list] if suggestions_list else [])

        # Save to history (save the structured list)
        record = {
            "username": username,
            "prompt": prompt,
            "suggestions": suggestions_list, # Save the list of dicts
            "timestamp": datetime.now().isoformat()
        }
        save_or_update(record)

        return jsonify(status="success",
                      message="Recommendations generated.",
                      suggestions=suggestions_list) # Return the list of dicts

    except Exception as e:
        app.logger.error("submit-preferences error: %s", e)
        import traceback
        traceback.print_exc()
        return jsonify(status="error", message=str(e)), 500

@app.route("/history")
def history():
    if 'username' not in session:
        return redirect(url_for('login_page'))
    return render_template("history.html", username=session['username'])

@app.route("/api/history")
def get_history():
    if 'username' not in session:
        return jsonify(status="error", message="User not logged in"), 401
        
    data = load_data()
    user_history = [item for item in data if item.get("username") == session['username']]
    
    # Sort history by timestamp descending (latest first)
    try:
        user_history.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    except Exception as e:
        app.logger.error(f"Error sorting history: {e}")
        # Continue without sorting if error occurs

    # Ensure suggestions are in a consistent format (list of titles for older entries)
    for item in user_history:
        if isinstance(item.get('suggestions'), str):
             # Attempt to parse old string format into titles
             item['suggestions_titles'] = [line.split(',')[0].strip() for line in item['suggestions'].split('\n') if line.strip()]
        elif isinstance(item.get('suggestions'), list):
             # Extract titles from the list of dicts
             item['suggestions_titles'] = [movie.get('title', 'Unknown Title') for movie in item['suggestions']]
        else:
             item['suggestions_titles'] = [] # Default to empty list

    # We only need to send prompt, timestamp, and titles to the frontend for history display
    frontend_history = [
        {
            "prompt": item.get("prompt", "N/A"),
            "timestamp": item.get("timestamp", "N/A"),
            "suggestions_titles": item.get("suggestions_titles", [])
        }
        for item in user_history
    ]
    
    return jsonify(frontend_history)

@app.route("/api/get-preferences")
def get_preferences():
    print("\n=== Starting preferences fetch ===")
    if 'username' not in session:
        print("Error: User not logged in")
        return jsonify(status="error", message="User not logged in"), 401
    
    print(f"Fetching preferences for username: {session['username']}")
    user_id = session.get('user_id')
    if not user_id:
        print("Error: User ID not found in session")
        return jsonify(status="error", message="User ID not found"), 404
    
    print(f"User ID: {user_id}")
    # Get user preferences from the personalizer
    preferences = personalizer.get_user_preferences(user_id)
    if preferences:
        print(f"Found preferences: {preferences}")
        return jsonify(status="success", preferences=preferences)
    else:
        print("No preferences found, returning defaults")
        return jsonify(status="success", preferences={
            'genres': [],
            'favorite_movies': [],
            'worst_movies': [],
            'era': 'new'
        })

@app.route("/search-movies", methods=["GET"])
def search_movies():
    try:
        if 'username' not in session:
            return jsonify(status="error", message="User not logged in"), 401
            
        query = request.args.get('q', '').lower()
        if not query:
            return jsonify([])
            
        # Search in both title and original_title columns
        matches = df[
            (df['title'].str.lower().str.contains(query, na=False)) |
            (df['original_title'].str.lower().str.contains(query, na=False))
        ]
        
        # Get the first 10 matches with title, original_title, and release_date
        results = matches[['title', 'original_title', 'release_date']].head(10)
        
        # Format the results
        formatted_results = []
        for _, row in results.iterrows():
            formatted_results.append({
                'title': row['title'],
                'original_title': row['original_title'],
                'release_date': row['release_date']
            })
            
        return jsonify(formatted_results)
    except Exception as e:
        app.logger.error(f"Search error: {str(e)}")
        return jsonify([])

if __name__ == "__main__":
    # Disable Flask's reloader in debug mode
    app.run(debug=True, use_reloader=False)
