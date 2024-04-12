import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# Load data
def load_data():
    credits_df = pd.read_csv(r"C:\Users\X1 CARBON\Downloads\credits (1).csv")
    movies_df = pd.read_csv(r"C:\Users\X1 CARBON\Downloads\movies (1).csv")
    return credits_df, movies_df

credits_df, movies_df = load_data()

# Merge dataframes on 'movie_id'
merged_df = pd.merge(movies_df, credits_df, left_on='id', right_on='movie_id')

# Preprocess data
# Drop unnecessary columns
merged_df.drop(['movie_id', 'crew'], axis=1, inplace=True)

# Fill missing values
merged_df['genres'] = merged_df['genres'].fillna('[]')  # Fill missing genres with empty list
merged_df['cast'] = merged_df['cast'].fillna('') 
merged_df['keywords'] = merged_df['keywords'].fillna('') 
merged_df['overview'] = merged_df['overview'].fillna('') 

# Combine selected features into a single column
merged_df['combined_features'] = merged_df['genres'] + ' ' + \
                                 merged_df['cast'] + ' ' + \
                                 merged_df['keywords'] + ' ' + \
                                 merged_df['overview']

# Create TF-IDF matrix
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(merged_df['combined_features'])

# Fit KNN model
knn_model = NearestNeighbors(n_neighbors=10, metric='cosine')
knn_model.fit(tfidf_matrix)

# Define function to get movie recommendations
def get_recommendations(title):
    # Transform input title into TF-IDF vector
    title_tfidf = tfidf_vectorizer.transform([title])

    # Find k-nearest neighbors
    distances, indices = knn_model.kneighbors(title_tfidf)

    # Get indices of similar movies
    similar_indices = indices[0][1:]  # Exclude the first index which is the input movie itself

    # Get movie titles based on indices
    similar_movies = merged_df.iloc[similar_indices]['original_title']

    return similar_movies

# Streamlit UI
st.title('Movie Recommendation System')

# Input field for user to enter movie title
user_input = st.text_input('Enter a movie title:', 'The Dark Knight')

# Button to trigger recommendation
if st.button('Get Recommendations'):
    recommendations = get_recommendations(user_input)
    st.subheader('Recommended Movies:')
    st.write(recommendations)
