import streamlit as st
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse # Required to load the compressed file

st.title("Movie Recommender System")

# 1. Load the dictionary and the COMPRESSED vectors
movies_dict = pickle.load(open('movie_dict.pkl','rb'))
movies = pd.DataFrame(movies_dict)
vectors = sparse.load_npz('vectors.npz') # Load the .npz file

selected_movie = st.selectbox("Select a movie:", movies['title'].values)

if st.button('Recommend'):
    idx = movies[movies['title'] == selected_movie].index[0]
    
    # Calculate similarity on the fly using the sparse vector
    distances = cosine_similarity(vectors[idx], vectors)[0]
    
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    
    for i in movie_list:
        st.write(f"🎥 {movies.iloc[i[0]].title}")
