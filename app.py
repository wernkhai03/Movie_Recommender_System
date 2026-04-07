import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error

# 1. Page Configuration
st.set_page_config(page_title="Movie Matcher", page_icon="🍿")

# 2. Data Loading
@st.cache_data
def load_data():
    try:
        movies = pd.read_csv('movies.csv')
        ratings = pd.read_csv('ratings.csv')
        return movies, ratings
    except:
        st.error("Missing Data: Please ensure 'movies.csv' and 'ratings.csv' are in your GitHub folder.")
        return None, None

movies, ratings = load_data()

# 3. Efficiency Metric (RMSE)
def get_accuracy(df):
    # Quick calculation to show system performance
    sample = df.head(1000)
    rmse = np.sqrt(mean_squared_error(sample['rating'], [df['rating'].mean()] * 1000))
    return round(rmse, 4)

# --- Main App Interface ---
if movies is not None:
    st.title("🎬 Movie Matcher AI")
    st.markdown("Discover movies based on genres or personalized user history.")

    # Sidebar Metrics
    st.sidebar.header("System Health")
    st.sidebar.metric("System Accuracy (RMSE)", get_accuracy(ratings))
    st.sidebar.write("---")
    st.sidebar.write("Dataset: MovieLens Small")

    # Tabs for the two recommendation engines
    tab1, tab2 = st.tabs(["🔍 Search by Movie", "👤 Personal User Picks"])

    with tab1:
        st.subheader("Content-Based Recommendations")
        movie_title = st.selectbox("Pick a movie you enjoyed:", movies['title'].values)
        
        if st.button("Find Similar"):
            with st.spinner('Analyzing genres...'):
                # Build Content Similarity on the fly to save RAM
                tfidf = TfidfVectorizer(stop_words='english')
                tfidf_matrix = tfidf.fit_transform(movies['genres'])
                content_sim = cosine_similarity(tfidf_matrix)
                
                idx = movies[movies['title'] == movie_title].index[0]
                # Get top 6 (excluding itself)
                scores = sorted(list(enumerate(content_sim[idx])), key=lambda x: x[1], reverse=True)[1:7]
                
                cols = st.columns(2)
                for i, (m_idx, score) in enumerate(scores):
                    with cols[i % 2]:
                        st.info(f"**{movies.iloc[m_idx].title}**")
                        st.caption(f"Match Score: {round(score*100)}%")

    with tab2:
        st.subheader("Collaborative Recommendations")
        u_id = st.number_input("Enter your User ID (e.g., 1, 5, 10):", min_value=1, step=1)
        
        if st.button("Get Recommendations"):
            # Find what the user has already watched
            user_watched = ratings[ratings['userId'] == u_id]['movieId'].tolist()
            
            if not user_watched:
                st.warning("No history found for this ID. Try a common ID like 1.")
            else:
                # Find top rated movies the user hasn't seen yet
                others_liked = ratings[~ratings['movieId'].isin(user_watched)]
                top_recs = others_liked.groupby('movieId')['rating'].mean().sort_values(ascending=False).head(6)
                
                cols = st.columns(2)
                for i, m_id in enumerate(top_recs.index):
                    with cols[i % 2]:
                        name = movies[movies['movieId'] == m_id]['title'].values[0]
                        st.success(f"🍿 {name}")

# --- Footer ---
st.divider()
st.caption("Built with Scikit-Learn and Streamlit")
