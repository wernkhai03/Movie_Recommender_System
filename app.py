import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error

# 1. Page Configuration
st.set_page_config(page_title="Movie Recommender System", page_icon="🍿", layout="wide")

# 2. Data Loading
@st.cache_data
def load_data():
    try:
        movies = pd.read_csv('movies.csv')
        ratings = pd.read_csv('ratings.csv')
        return movies, ratings
    except Exception as e:
        return None, None

movies, ratings = load_data()

# 3. Accuracy Logic Function
def get_user_rmse(user_ratings, all_ratings):
    if user_ratings.empty:
        return 0.0
    actual = user_ratings['rating']
    predicted = [all_ratings['rating'].mean()] * len(actual)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    return round(rmse, 4)

# --- Main App Logic ---
if movies is not None and ratings is not None:
    st.title("🎬 Movie Recommender System")
    
    # Create the Tabs FIRST so they are defined for the rest of the script
    tab1, tab2 = st.tabs(["🔍 Search by Movie", "👤 Personal User Picks"])

    with tab1:
        st.subheader("Content-Based Recommendations")
        movie_title = st.selectbox("Select a movie you liked:", movies['title'].values)
        
        if st.button("Find Similar"):
            with st.spinner('Calculating similarity...'):
                # Combine Title + Genre for unique scores
                movies['metadata'] = movies['title'] + " " + movies['genres'].str.replace('|', ' ')
                tfidf = TfidfVectorizer(stop_words='english')
                tfidf_matrix = tfidf.fit_transform(movies['metadata'])
                
                idx = movies[movies['title'] == movie_title].index[0]
                selected_vector = tfidf_matrix[idx]
                content_sim = cosine_similarity(selected_vector, tfidf_matrix).flatten()
                
                related_indices = content_sim.argsort()[::-1][1:7]
                
                cols = st.columns(3)
                for i, m_idx in enumerate(related_indices):
                    score = content_sim[m_idx]
                    with cols[i % 3]:
                        st.info(f"**{movies.iloc[m_idx].title}**")
                        st.caption(f"Match Score: {round(score*100, 1)}%")

    with tab2:
        st.subheader("Collaborative Recommendations")
        u_id = st.number_input("Enter your User ID:", min_value=1, step=1)
        
        if st.button("Get My Recommendations"):
            user_history = ratings[ratings['userId'] == u_id]
            
            if user_history.empty:
                st.warning(f"No history for User {u_id}. Try ID 1 or 5.")
            else:
                # DYNAMIC RMSE: This value now changes based on the user!
                user_rmse = get_user_rmse(user_history, ratings)
                st.sidebar.metric("User Specific RMSE", user_rmse)
                st.sidebar.caption(f"This is the prediction error for User {u_id}.")
                
                # Recommendation Logic
                watched_ids = user_history['movieId'].tolist()
                peer_users = ratings[ratings['movieId'].isin(watched_ids)]['userId'].unique()
                
                recommendations = (
                    ratings[(ratings['userId'].isin(peer_users)) & (~ratings['movieId'].isin(watched_ids))]
                    .groupby('movieId')['rating']
                    .mean()
                    .sort_values(ascending=False)
                    .head(6)
                )
                
                cols = st.columns(3)
                for i, m_id in enumerate(recommendations.index):
                    name = movies[movies['movieId'] == m_id]['title'].values[0]
                    with cols[i % 3]:
                        st.success(f"🍿 {name}")

else:
    st.error("Could not find data files. Please ensure 'movies.csv' and 'ratings.csv' are uploaded.")

# Footer
st.sidebar.write("---")
