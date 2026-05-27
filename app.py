import streamlit as st
import pickle
import pandas as pd
import requests
from recommender import MovieRecommender

st.set_page_config(page_title=" Movie Recommender", layout="wide")

st.title("AI Movie Recommendation System")
st.markdown("### Find your next favorite movie!")

# Load model
@st.cache_resource
def load_recommender():
    try:
        movies = pickle.load(open('models/movies.pkl', 'rb'))
        similarity = pickle.load(open('models/similarity.pkl', 'rb'))
        return movies, similarity
    except:
        st.error("Model not found. Please run recommender.py first.")
        return None, None

movies, similarity = load_recommender()

if movies is not None:
    movie_list = movies['title'].values
    selected_movie = st.selectbox("Search for a movie you liked:", movie_list)
    
    if st.button("Get Recommendations"):
        with st.spinner("Finding similar movies..."):
            recommender = MovieRecommender()
            recommender.movies = movies
            recommender.similarity = similarity
            
            recommendations = recommender.recommend(selected_movie, 8)
            
            cols = st.columns(4)
            for idx, movie in enumerate(recommendations):
                with cols[idx % 4]:
                    st.subheader(movie)
