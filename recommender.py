import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

class MovieRecommender:
    def __init__(self, data_path='data/tmdb_5000_movies.csv'):
        self.data_path = data_path
        self.df = None
        self.similarity = None
        self.movies = None
        
    def load_data(self):
        """Load and preprocess the dataset"""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset not found at {self.data_path}. Download from Kaggle.")
        
        self.df = pd.read_csv(self.data_path)
        
        # Select important columns
        self.movies = self.df[['id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
        
        # Clean and combine features
        self.movies = self.movies.copy()
        self.movies['overview'] = self.movies['overview'].fillna('')
        
        self.movies['genres'] = self.movies['genres'].apply(self._convert)
        self.movies['keywords'] = self.movies['keywords'].apply(self._convert)
        self.movies['cast'] = self.movies['cast'].apply(self._convert_cast)
        self.movies['crew'] = self.movies['crew'].apply(self._fetch_director)
        
        # Create tags
        self.movies['tags'] = (self.movies['overview'] + ' ' +
                             self.movies['genres'] + ' ' +
                             self.movies['keywords'] + ' ' +
                             self.movies['cast'] + ' ' +
                             self.movies['crew'])
        
        print(f"Loaded {len(self.movies)} movies")
        
    def _convert(self, text):
        """Convert JSON string to list of names"""
        import ast
        try:
            L = []
            for i in ast.literal_eval(text):
                L.append(i['name'])
            return " ".join(L)
        except:
            return ""
    
    def _convert_cast(self, text):
        """Take top 3 cast members"""
        import ast
        try:
            L = []
            counter = 0
            for i in ast.literal_eval(text):
                if counter < 3:
                    L.append(i['name'])
                    counter += 1
            return " ".join(L)
        except:
            return ""
    
    def _fetch_director(self, text):
        """Extract director name"""
        import ast
        try:
            for i in ast.literal_eval(text):
                if i['job'] == 'Director':
                    return i['name']
            return ""
        except:
            return ""
    
    def build_model(self):
        """Build CountVectorizer + Cosine Similarity"""
        cv = CountVectorizer(max_features=5000, stop_words='english')
        vectors = cv.fit_transform(self.movies['tags']).toarray()
        
        self.similarity = cosine_similarity(vectors)
        print("Model built successfully!")
        
    def recommend(self, movie_title, top_n=10):
        """Recommend movies"""
        if self.df is None or self.similarity is None:
            self.load_data()
            self.build_model()
        
        try:
            movie_idx = self.movies[self.movies['title'].str.lower() == movie_title.lower()].index[0]
            distances = self.similarity[movie_idx]
            movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:top_n+1]
            
            recommendations = []
            for i in movies_list:
                recommendations.append(self.movies.iloc[i[0]].title)
            return recommendations
        except IndexError:
            return ["Movie not found in database. Try another title."]
    
    def save_model(self, path='models/'):
        """Save processed data and similarity matrix"""
        os.makedirs(path, exist_ok=True)
        pickle.dump(self.movies, open(f'{path}movies.pkl', 'wb'))
        pickle.dump(self.similarity, open(f'{path}similarity.pkl', 'wb'))
        print("Model saved!")

# For direct usage
if __name__ == "__main__":
    recommender = MovieRecommender()
    recommender.load_data()
    recommender.build_model()
    recommender.save_model()
    
    # Test
    print(recommender.recommend("The Dark Knight", 5))
