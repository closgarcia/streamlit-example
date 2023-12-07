# music_recommender_app.py
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline

# Load your dataset (if not already loaded)
tracks_df = pd.read_csv('./spotify-dataset-19212020-600k-tracks/tracks.csv')

# Your existing code for clustering, input_preprocessor, and Music_Recommender functions
# Fill in your existing code here...

# Example for clustering
def perform_clustering(data):
    # Implement your clustering logic here
    # Example using KMeans
    kmeans = KMeans(n_clusters=8)
    clusters = kmeans.fit_predict(data)
    return clusters

# Example for input preprocessing
def input_preprocessor(user_songs):
    # Implement your input preprocessing logic here
    # Example: Extract relevant features from user input
    user_features = [song['valence'] for song in user_songs]
    return user_features

# Example for Music_Recommender function
def Music_Recommender(user_songs):
    # Implement your recommendation logic here
    # Example: Use clustering and input preprocessing
    # Assume 'tracks_df' is available globally

    # Extract relevant features from the dataset
    data_for_clustering = tracks_df[features]

    # Perform clustering
    clusters = perform_clustering(data_for_clustering)

    # Preprocess user input
    user_features = input_preprocessor(user_songs)

    # Identify the cluster of the user input
    user_cluster = perform_clustering([user_features])[0]

    # Filter songs in the same cluster as the user input
    cluster_songs = tracks_df[clusters == user_cluster]

    # Recommend top songs from the cluster
    recommendations = cluster_songs.head(10)  # Adjust as needed
    return recommendations


# Define features and metadata columns
features = ['valence', 'year', 'acousticness', 'danceability', 'duration_ms', 'energy',
            'explicit', 'instrumentalness', 'key', 'liveness', 'loudness', 'mode',
            'popularity', 'speechiness', 'tempo']

metadata_cols = ['year', 'name', 'artists']

# Create a pipeline with StandardScaler and KMeans
song_cluster_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('kmeans', KMeans(n_clusters=8, verbose=2))
], verbose=True)

# Function to preprocess input and recommend songs
def Music_Recommender(song_list, n_songs=10):
    # Your existing code for recommendation
    # Fill in your existing code here...

# Streamlit app code
def main():
    st.title("Music Recommender App")
    st.sidebar.header("Input Songs")

    # Get user input for songs
    song_name = st.sidebar.text_input("Enter Song Name:")
    song_year = st.sidebar.number_input("Enter Year:", min_value=1920, max_value=2023, step=1)
    
    # Create a list of user-input songs
    user_songs = [{'name': song_name, 'year': int(song_year)}]

    # Call the Music_Recommender function
    recommendations = Music_Recommender(user_songs)

    # Display the recommendations
    st.subheader("Recommendations:")
    st.table(recommendations)

if __name__ == "__main__":
    main()


