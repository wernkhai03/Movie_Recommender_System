import streamlit as st
import pandas as pd
import ast
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- STREAMLIT UI SETUP ---
st.set_page_config(page_title="Movie Recommender")
st.title("🎬 Movie Recommender System")

# 1. Load the datasets (Using the exact names from your GitHub screenshot)
@st.cache_data
def load_data():
    # Note: We added .zip to the movies filename to match your GitHub
    movies = pd.read_csv('tmdb_5000_movies.csv')
    credits = pd.read_csv('tmdb_5000_credits.zip', compression='zip')
    return movies, credits

try:
    movies, credits = load_data()
    # 2. Merge datasets
    movies = movies.merge(credits, on='title')
except Exception as e:
    st.error(f"Error loading files: {e}")
    st.stop()

# 3. Select relevant columns
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

# 4. Data Cleaning Functions
def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

def convert_cast(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter += 1
        else:
            break
    return L

def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L

# Apply cleaning
movies.dropna(inplace=True)
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(convert_cast)
movies['crew'] = movies['crew'].apply(fetch_director)

# Remove spaces
movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ", "") for i in x])

# 5. Create "Tags" Column
movies['overview'] = movies['overview'].apply(lambda x: x.split())
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

new_df = movies[['movie_id', 'title', 'tags']]
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x).lower())

# 6. Vectorization
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()

# 7. Similarity
similarity = cosine_similarity(vectors)

# --- STREAMLIT INTERACTIVE PART ---
movie_list = new_df['title'].values
selected_movie = st.selectbox(
    "Type or select a movie to get recommendations:",
    movie_list
)

if st.button('Show Recommendation'):
    movie_index = new_df[new_df['title'] == selected_movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    
    st.subheader(f"Because you liked '{selected_movie}', you might also like:")
    for i in movies_list:
        st.write(f"🎥 {new_df.iloc[i[0]].title}")
