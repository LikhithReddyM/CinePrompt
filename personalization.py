import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pickle
import os
from pathlib import Path
from scipy.sparse import csr_matrix
import warnings
import re
from collections import defaultdict
warnings.filterwarnings('ignore')

class MoviePersonalizer:
    _instance = None
    MODEL_FILE = 'saved_data/personalization_model.pkl'
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.user_preferences = {}
        self.movie_features = None
        self.movie_ids = None
        self.user_movie_matrix = None
        self.movie_similarity = None
        self.new_users_since_update = 0
        self.BATCH_UPDATE_THRESHOLD = 10  # Update model after 10 new users
        
        # Create saved_data directory if it doesn't exist
        os.makedirs('saved_data', exist_ok=True)
        
        # Try to load existing model
        try:
            self.load_model()
            print("Successfully loaded existing personalization model")
        except Exception as e:
            print(f"Error loading model: {str(e)}. Creating new model...")
            self.load_data()
            self.save_model()
    
    def load_data(self):
        """Load and process the movies and ratings data"""
        print("Loading and processing data...")
        
        # Load movies data
        movies = pd.read_csv('dataset/merged_data.csv')
        
        # Load ratings data in chunks to handle large files
        ratings_chunks = pd.read_csv('dataset/ratings.csv', chunksize=1000000)
        ratings = pd.concat([chunk[chunk['rating'] >= 3.0] for chunk in ratings_chunks])
        
        # Create user-movie matrix
        print("Creating user-movie matrix...")
        self.user_movie_matrix = csr_matrix(
            (ratings['rating'], 
             (ratings['userId'] - 1, ratings['movieId'] - 1))
        )
        
        # Extract movie features
        print("Extracting movie features...")
        self.movie_features, self.movie_ids = self._extract_movie_features(movies)
        
        # Calculate movie similarities in chunks to avoid memory issues
        print("Calculating movie similarities...")
        chunk_size = 1000
        n_movies = len(self.movie_ids)
        self.movie_similarity = np.zeros((n_movies, n_movies))
        
        for i in range(0, n_movies, chunk_size):
            end_i = min(i + chunk_size, n_movies)
            for j in range(0, n_movies, chunk_size):
                end_j = min(j + chunk_size, n_movies)
                self.movie_similarity[i:end_i, j:end_j] = cosine_similarity(
                    self.movie_features[i:end_i], 
                    self.movie_features[j:end_j]
                )
                print(f"Processed chunk {i//chunk_size + 1}, {j//chunk_size + 1}")
        
        print("Data processing complete!")
    
    def save_model(self):
        """Save the current model state"""
        print("Saving personalization model...")
        model_data = {
            'movie_features': self.movie_features,
            'movie_ids': self.movie_ids,
            'user_movie_matrix': self.user_movie_matrix,
            'movie_similarity': self.movie_similarity,
            'user_preferences': self.user_preferences,
            'new_users_since_update': self.new_users_since_update
        }
        
        with open(self.MODEL_FILE, 'wb') as f:
            pickle.dump(model_data, f)
        print("Model saved successfully!")
    
    def load_model(self):
        """Load the saved model state"""
        if not os.path.exists(self.MODEL_FILE):
            raise FileNotFoundError("No saved model found")
            
        print("Loading personalization model...")
        with open(self.MODEL_FILE, 'rb') as f:
            model_data = pickle.load(f)
        
        self.movie_features = model_data['movie_features']
        self.movie_ids = model_data['movie_ids']
        self.user_movie_matrix = model_data['user_movie_matrix']
        self.movie_similarity = model_data['movie_similarity']
        self.user_preferences = model_data['user_preferences']
        self.new_users_since_update = model_data['new_users_since_update']
        print("Model loaded successfully!")
    
    def _check_and_update_model(self):
        """Check if model needs to be updated based on new users"""
        if self.new_users_since_update >= self.BATCH_UPDATE_THRESHOLD:
            print("Threshold reached, updating model...")
            self.load_data()  # Reload and reprocess all data
            self.new_users_since_update = 0
            self.save_model()
    
    def _extract_movie_features(self, movies):
        """Extract relevant features from movies data"""
        # Store movie IDs
        movie_ids = movies['id'].values
        
        # Convert genres to one-hot encoding
        genres = movies['genres'].str.get_dummies(sep='|')
        self.genre_names = genres.columns.tolist()  # Store genre names
        
        # Normalize numerical features
        numerical_features = movies[['vote_average', 'popularity']].fillna(0)
        numerical_features = self.scaler.fit_transform(numerical_features)
        
        # Combine features
        features = np.hstack([genres.values, numerical_features])
        return features, movie_ids
        
    def _get_movie_code(self, movie_id):
        """Convert movie ID to matrix code"""
        return self.reverse_movie_id_map.get(movie_id, None)
        
    def update_user_preferences(self, user_id, preferences):
        """Update user preferences without saving the model"""
        print(f"\n=== Updating preferences for user {user_id} ===")
        print(f"New preferences: {preferences}")
        self.user_preferences[user_id] = preferences
        print("Preferences updated in memory")

    def save_user_preferences(self, user_id):
        """Save user preferences to disk without updating the model"""
        print(f"\n=== Saving preferences to disk for user {user_id} ===")
        try:
            # Load existing preferences if any
            prefs_file = 'saved_data/user_preferences.pkl'
            if os.path.exists(prefs_file):
                print("Loading existing preferences file")
                with open(prefs_file, 'rb') as f:
                    all_preferences = pickle.load(f)
            else:
                print("No existing preferences file, creating new")
                all_preferences = {}
            
            # Update preferences for this user
            all_preferences[user_id] = self.user_preferences.get(user_id, {})
            print(f"Updated preferences for user {user_id}: {all_preferences[user_id]}")
            
            # Save back to disk
            with open(prefs_file, 'wb') as f:
                pickle.dump(all_preferences, f)
            
            print("Successfully saved preferences to disk")
        except Exception as e:
            print(f"Error saving user preferences: {str(e)}")

    def load_user_preferences(self, user_id):
        """Load user preferences from disk"""
        print(f"\n=== Loading preferences from disk for user {user_id} ===")
        try:
            prefs_file = 'saved_data/user_preferences.pkl'
            if os.path.exists(prefs_file):
                print("Found preferences file")
                with open(prefs_file, 'rb') as f:
                    all_preferences = pickle.load(f)
                    if user_id in all_preferences:
                        print(f"Found preferences for user {user_id}")
                        self.user_preferences[user_id] = all_preferences[user_id]
                        print(f"Loaded preferences: {all_preferences[user_id]}")
                        return True
                print(f"No preferences found for user {user_id}")
            else:
                print("No preferences file exists")
            return False
        except Exception as e:
            print(f"Error loading user preferences: {str(e)}")
            return False

    def get_user_preferences(self, user_id):
        """Get user preferences from memory or load from disk if not in memory"""
        print(f"\n=== Getting preferences for user {user_id} ===")
        # First try to get from memory
        if user_id in self.user_preferences:
            print("Found preferences in memory")
            print(f"Preferences: {self.user_preferences[user_id]}")
            return self.user_preferences[user_id]
        
        print("Preferences not in memory, trying to load from disk")
        # If not in memory, try to load from disk
        if self.load_user_preferences(user_id):
            print("Successfully loaded preferences from disk")
            return self.user_preferences[user_id]
        
        print("No preferences found anywhere")
        # If no preferences found, return None
        return None

    def track_user_interaction(self, user_id, prompt, recommended_movies, selected_movies=None):
        """Track user interaction and check for batch update"""
        # Store the prompt
        self.user_prompts[user_id].append(prompt)
        
        # Extract keywords from prompt
        keywords = self._extract_keywords(prompt)
        
        # Update keyword weights based on frequency
        for keyword in keywords:
            self.keyword_weights[keyword] += 1
            
        # If user selected specific movies, track those interactions
        if selected_movies:
            self.user_interactions[user_id].extend(selected_movies)
            self.total_interactions_since_update += len(selected_movies)
            
            # Update genre weights based on selected movies
            for movie_id in selected_movies:
                movie_idx = np.where(self.movie_ids == movie_id)[0]
                if len(movie_idx) > 0:
                    movie_genres = self.movie_features[movie_idx[0], :len(self.genre_weights)]
                    for genre_idx, weight in enumerate(movie_genres):
                        if weight > 0:
                            genre_name = list(self.genre_weights.keys())[genre_idx]
                            self.genre_weights[genre_name] += 1
            
        # Check if we need to update the model
        self._check_and_update_model()
        
    def _extract_keywords(self, prompt):
        """Extract relevant keywords from user prompt"""
        # Convert to lowercase and remove special characters
        clean_prompt = re.sub(r'[^\w\s]', '', prompt.lower())
        
        # Split into words and remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = [word for word in clean_prompt.split() if word not in stop_words]
        
        # Extract potential keywords (nouns, adjectives)
        keywords = []
        for word in words:
            if len(word) > 3:  # Filter out short words
                keywords.append(word)
                
        return keywords
        
    def get_personalized_scores(self, user_id, movie_ids):
        """Calculate personalized scores for given movies"""
        if user_id not in self.user_preferences:
            # Try to load preferences from disk
            if not self.load_user_preferences(user_id):
                return np.ones(len(movie_ids))
            
        preferences = self.user_preferences[user_id]
        
        # Content-based filtering score
        content_scores = self._calculate_content_scores(preferences, movie_ids)
        
        # Collaborative filtering score
        collab_scores = self._calculate_collab_scores(user_id, movie_ids)
        
        # Combine scores with weights
        final_scores = (
            0.6 * content_scores +  # Content-based filtering
            0.4 * collab_scores    # Collaborative filtering
        )
        
        return final_scores
        
    def _calculate_content_scores(self, preferences, movie_ids):
        """Calculate content-based filtering scores"""
        scores = np.zeros(len(movie_ids))
        
        if 'genres' in preferences:
            genre_scores = self._calculate_genre_scores(preferences['genres'], movie_ids)
            scores += genre_scores
            
        if 'favorite_movies' in preferences:
            fav_scores = self._calculate_favorite_similarity(preferences['favorite_movies'], movie_ids)
            scores += fav_scores
            
        if 'era' in preferences:
            era_scores = self._calculate_era_scores(preferences['era'], movie_ids)
            scores += era_scores
            
        return scores / 3
        
    def _calculate_genre_scores(self, preferred_genres, movie_ids):
        """Calculate scores based on genre preferences"""
        scores = np.zeros(len(movie_ids))
        
        # If genre_names is not available, use the length of preferred_genres
        if not hasattr(self, 'genre_names'):
            self.genre_names = list(set(preferred_genres))  # Create a list of unique genres
        
        # Create a mapping of genre names to their indices in the feature matrix
        genre_indices = {genre: idx for idx, genre in enumerate(self.genre_names)}
        
        # Create a one-hot encoded array for preferred genres
        preferred_genres_one_hot = np.zeros(len(self.genre_names))
        for genre in preferred_genres:
            if genre in genre_indices:
                preferred_genres_one_hot[genre_indices[genre]] = 1.0
        
        for i, movie_id in enumerate(movie_ids):
            movie_idx = np.where(self.movie_ids == movie_id)[0]
            if len(movie_idx) > 0:
                movie_genres = self.movie_features[movie_idx[0], :len(self.genre_names)]
                scores[i] = np.sum(movie_genres * preferred_genres_one_hot)
        
        return scores
        
    def _calculate_favorite_similarity(self, favorite_movies, movie_ids):
        """Calculate scores based on similarity to favorite movies"""
        scores = np.zeros(len(movie_ids))
        for i, movie_id in enumerate(movie_ids):
            movie_idx = np.where(self.movie_ids == movie_id)[0]
            if len(movie_idx) > 0:
                for fav_movie in favorite_movies:
                    fav_idx = np.where(self.movie_ids == fav_movie)[0]
                    if len(fav_idx) > 0:
                        scores[i] += self.movie_similarity[movie_idx[0], fav_idx[0]]
        return scores / len(favorite_movies) if favorite_movies else scores
        
    def _calculate_era_scores(self, preferred_era, movie_ids):
        """Calculate scores based on movie era preferences"""
        scores = np.zeros(len(movie_ids))
        for i, movie_id in enumerate(movie_ids):
            movie_idx = np.where(self.movie_ids == movie_id)[0]
            if len(movie_idx) > 0:
                release_year = self.movie_features[movie_idx[0], -1]
                if preferred_era == 'old' and release_year < 1970:
                    scores[i] = 1
                elif preferred_era == 'middle' and 1970 <= release_year < 2000:
                    scores[i] = 1
                elif preferred_era == 'new' and release_year >= 2000:
                    scores[i] = 1
        return scores
        
    def _calculate_collab_scores(self, user_id, movie_ids):
        """Calculate collaborative filtering scores"""
        if self.user_movie_matrix is None:
            return np.zeros(len(movie_ids))
        
        try:
            # Convert user_id to matrix index (assuming user_ids start from 1)
            user_idx = int(user_id) - 1
            
            # Check if user exists in the matrix
            if user_idx >= self.user_movie_matrix.shape[0]:
                return np.zeros(len(movie_ids))
            
            scores = np.zeros(len(movie_ids))
            for i, movie_id in enumerate(movie_ids):
                # Convert movie_id to matrix index (assuming movie_ids start from 1)
                movie_idx = int(movie_id) - 1
                if movie_idx < self.user_movie_matrix.shape[1]:
                    scores[i] = self.user_movie_matrix[user_idx, movie_idx]
                
            return scores / 5  # Normalize to [0,1]
        except (ValueError, TypeError) as e:
            print(f"Error in collaborative filtering: {str(e)}")
            return np.zeros(len(movie_ids))
