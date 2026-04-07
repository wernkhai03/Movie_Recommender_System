import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("🎬 Movie Recommendation System")

# 1. Load Data with Caching
@st.cache_data
def load_data():
    try:
        # Ensure these files are in your GitHub repo folder
        movies = pd.read_csv('movies.csv')
        ratings = pd.read_csv('ratings.csv')
        return movies, ratings
    except FileNotFoundError:
        return None, None

movies, ratings = load_data()

if movies is None or ratings is None:
    st.error("Missing Data: Please upload 'movies.csv' and 'ratings.csv' to your GitHub repository.")
else:
    # 2. Content-Based Pre-calculation (Genres)
    # We use cache_resource for the similarity matrix to save RAM
    @st.cache_resource
    def get_content_sim(movies_df):
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(movies_df['genres'])
        return cosine_similarity(tfidf_matrix, tfidf_matrix)

    content_sim = get_content_sim(movies)

    # --- UI Logic ---
    option = st.sidebar.selectbox("Recommendation Type", ["By Movie Title", "By User ID"])

    if option == "By Movie Title":
        st.header("Similar Movie Recommendations")
        movie_list = movies['title'].values
        selected_movie = st.selectbox("Type or select a movie", movie_list)
        
        if st.button('Show Recommendations'):
            idx = movies[movies['title'] == selected_movie].index[0]
            distances = sorted(list(enumerate(content_sim[idx])), reverse=True, key=lambda x: x[1])
            
            st.subheader(f"Because you liked {selected_movie}:")
            for i in distances[1:7]:
                st.write(f"🎥 {movies.iloc[i[0]].title}")

    else:
        st.header("Personalized Recommendations")
        user_id = st.number_input("Enter User ID", min_value=1, step=1)
        
        if st.button('Show User Recommendations'):
            # Filter ratings for this user
            user_ratings = ratings[ratings['userId'] == user_id]
            
            if user_ratings.empty:
                st.warning(f"User ID {user_id} not found in dataset. Try IDs like 1, 2, or 5.")
            else:
                # Optimized logic: find what other people who liked the same movies also watched
                watched_ids = user_ratings['movieId'].unique()
                
                # Find users who liked at least one movie this user watched
                peers = ratings[ratings['movieId'].isin(watched_ids)]['userId'].unique()
                
                # Filter ratings to these peers and exclude movies the user already saw
                recommendations = (
                    ratings[(ratings['userId'].isin(peers)) & (~ratings['movieId'].isin(watched_ids))]
                    .groupby('movieId')['rating']
                    .mean()
                    .sort_values(ascending=False)
                    .head(5)
                )
                
                if not recommendations.empty:
                    st.subheader(f"Top 5 Picks for User {user_id}:")
                    for m_id in recommendations.index:
                        name = movies[movies['movieId'] == m_id]['title'].values[0]
                        st.write(f"⭐ {name}")
                else:
                    st.info("Not enough data to make a recommendation for this user.")
