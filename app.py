import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

st.title("🎬 Movie Recommendation System")

# 1. Load Data
@st.cache_data
def load_data():
    # Make sure these files are in your GitHub repo!
    movies = pd.read_csv('movies.csv')
    ratings = pd.read_csv('ratings.csv')
    return movies, ratings

try:
    movies, ratings = load_data()
    
    # 2. Content-Based Filtering (Genres)
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['genres'])
    content_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # 3. Collaborative Filtering (User-Item Matrix)
    # Creating a matrix where rows are users and columns are movies
    user_movie_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    user_sim = cosine_similarity(user_movie_matrix)
    user_sim_df = pd.DataFrame(user_sim, index=user_movie_matrix.index, columns=user_movie_matrix.index)

    # --- UI Logic ---
    option = st.selectbox("How do you want recommendations?", ["By Movie Title", "By User ID"])

    if option == "By Movie Title":
        movie_list = movies['title'].values
        selected_movie = st.selectbox("Type or select a movie", movie_list)
        
        if st.button('Show Recommendations'):
            idx = movies[movies['title'] == selected_movie].index[0]
            distances = sorted(list(enumerate(content_sim[idx])), reverse=True, key=lambda x: x[1])
            for i in distances[1:6]:
                st.write(movies.iloc[i[0]].title)

    else:
        user_id = st.number_input("Enter User ID", min_value=1, step=1)
        if st.button('Show User Recommendations'):
            # Find similar users
            similar_users = user_sim_df[user_id].sort_values(ascending=False).index[1:6]
            # Get movies those users liked
            recommended_movies = ratings[ratings['userId'].isin(similar_users)].sort_values(by='rating', ascending=False)
            top_movies = recommended_movies['movieId'].unique()[:5]
            
            for m_id in top_movies:
                name = movies[movies['movieId'] == m_id]['title'].values[0]
                st.write(name)

except Exception as e:
    st.error(f"Please ensure movies.csv and ratings.csv are uploaded. Error: {e}")
