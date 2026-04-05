import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- STEP 1: DATA PREPROCESSING ---
@st.cache_data # Cache so it doesn't reload every time you click a button
def load_and_clean_data():
    movies = pd.read_csv('tmdb_5000_movies.csv')
    credits = pd.read_csv('tmdb_5000_credits.zip')
    
    # Merge datasets on title
    movies = movies.merge(credits, on='title')
    
    # Select important columns
    movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
    movies.dropna(inplace=True)
    
    # Simple cleaning (extracting names from JSON-like strings)
    # For a quick start, we'll just use 'overview' and 'title' as tags
    movies['tags'] = movies['overview'] + " " + movies['genres']
    
    new_df = movies[['movie_id', 'title', 'tags']]
    new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())
    
    return new_df

df = load_and_clean_data()

# --- STEP 2: VECTORIZATION ---
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(df['tags']).toarray()
similarity = cosine_similarity(vectors)

# --- STEP 3: RECOMMENDATION LOGIC ---
def recommend(movie):
    try:
        movie_index = df[df['title'] == movie].index[0]
        distances = similarity[movie_index]
        movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
        
        return [df.iloc[i[0]].title for i in movies_list]
    except:
        return ["Movie not found!"]

# --- STEP 4: STREAMLIT UI ---
st.set_page_config(page_title="Movie Recommender", page_icon="🍿")
st.title("Movie Recommender System")

selected_movie = st.selectbox(
    "Search for a movie you've watched:",
    df['title'].values
)

if st.button('Show Recommendations'):
    recommendations = recommend(selected_movie)
    st.write("### You might also like:")
    for i in recommendations:
        st.success(i)
