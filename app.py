import pandas as pd
import streamlit as st
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 🔹 PUT YOUR OMDb API KEY HERE
API_KEY = "5de16792"

# Load dataset
movies = pd.read_csv("movies.csv")
movies = movies[['title', 'genres', 'keywords', 'overview']]
movies.fillna('', inplace=True)

# Combine text columns
movies['combined'] = movies['genres'] + " " + movies['keywords'] + " " + movies['overview']

# Convert text to numbers
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['combined']).toarray()

# Calculate similarity
similarity = cosine_similarity(vectors)

# 🔹 Function to fetch poster from OMDb
def fetch_poster(movie_name):
    url = f"http://www.omdbapi.com/?t={movie_name}&apikey={API_KEY}"
    data = requests.get(url).json()
    
    if data['Response'] == 'True':
        return data['Poster']
    else:
        return None

# Recommendation function
def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = similarity[index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended_movies = []
    recommended_posters = []

    for i in movie_list:
        movie_name = movies.iloc[i[0]].title
        recommended_movies.append(movie_name)
        recommended_posters.append(fetch_poster(movie_name))

    return recommended_movies, recommended_posters

# Streamlit UI
st.title("🎬 Movie Recommendation System with Posters")

selected_movie = st.selectbox("Select a movie", movies['title'].values)

if st.button("Recommend"):
    names, posters = recommend(selected_movie)

    col1, col2, col3, col4, col5 = st.columns(5)

    for idx, col in enumerate([col1, col2, col3, col4, col5]):
        with col:
            st.text(names[idx])
            if posters[idx] and posters[idx] != "N/A":
                st.image(posters[idx])