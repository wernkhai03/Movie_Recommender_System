import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Load the Datasets
# Assuming the files are named exactly as follows:
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

# 2. Merge and Select Relevant Columns
movies = movies.merge(credits, on='title')
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

# Drop rows with missing data (like missing overviews)
movies.dropna(inplace=True)

# 3. Data Preprocessing Helper Functions
def convert(obj):
    """Extracts 'name' values from JSON-like string columns."""
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

def convert_cast(obj):
    """Extracts the first 3 cast members."""
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
    """Extracts the director's name from the crew column."""
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L

# 4. Apply Preprocessing
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(convert_cast)
movies['crew'] = movies['crew'].apply(fetch_director)

# Split overview text into a list of words
movies['overview'] = movies['overview'].apply(lambda x: x.split())

# Remove spaces from tags (e.g., 'Johnny Depp' -> 'JohnnyDepp')
movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ", "") for i in x])

# 5. Create the 'Tags' Column
# Combine all metadata into a single list of strings
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

# Create a simplified DataFrame
new_df = movies[['movie_id', 'title', 'tags']]

# Convert tags list back to a string and lowercase everything
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x).lower())

# 6. Vectorization and Similarity Calculation
# Convert text tags into numerical vectors
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()

# Calculate Cosine Similarity between all movie vectors
similarity = cosine_similarity(vectors)

# 7. Recommendation Function
def recommend(movie_title):
    try:
        # Find the index of the movie in the DataFrame
        movie_index = new_df[new_df['title'] == movie_title].index[0]
        distances = similarity[movie_index]
        
        # Sort movies by similarity and pick the top 5 (excluding the movie itself)
        movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
        
        print(f"Recommendations for '{movie_title}':")
        for i in movies_list:
            print(f"- {new_df.iloc[i[0]].title}")
            
    except IndexError:
        print(f"Movie '{movie_title}' not found in the database. Check the spelling.")

# 8. Test the system
# Example Usage:
recommend('Avatar')
print("-" * 30)
recommend('The Dark Knight Rises')
