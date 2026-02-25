import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
movies = pd.read_csv("movies.csv")

# Select only needed columns
movies = movies[['title', 'genres', 'keywords', 'overview']]

# Fill empty values
movies.fillna('', inplace=True)

# Combine all text
movies['combined'] = movies['genres'] + " " + movies['keywords'] + " " + movies['overview']

# Convert text to numbers
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['combined']).toarray()

# Calculate similarity
similarity = cosine_similarity(vectors)

# Recommendation function
def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = similarity[index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    return [movies.iloc[i[0]].title for i in movie_list]

# Streamlit UI
st.title("🎬 Movie Recommendation System")

selected_movie = st.selectbox("Select a movie", movies['title'].values)

if st.button("Recommend"):
    recommendations = recommend(selected_movie)
    st.write("### Recommended Movies:")
    for movie in recommendations:
        st.write(movie)