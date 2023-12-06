
# streamlit_app.py
import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np

# Load the dataset (assuming it's already loaded in your notebook)
# Replace 'your_dataset.csv' with the actual dataset filename
tracks_df = pd.read_csv('./spotify-dataset-19212020-600k-tracks')

# Define features and metadata columns
features = ['valence', 'year', 'acousticness',
            'danceability', 'duration_ms', 'energy',
            'explicit', 'instrumentalness', 'key',
            'liveness', 'loudness', 'mode',
            'popularity', 'speechiness', 'tempo']

metadata_cols = ['year', 'name', 'artists']

# Create a pipeline with StandardScaler and KMeans
song_cluster_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('kmeans', KMeans(n_clusters=8, verbose=2))  # Adjust the number of clusters as needed
], verbose=True)

# Extract features for clustering
X = tracks_df[features]

# Fit the pipeline to the data
song_cluster_pipeline.fit(X)

def input_preprocessor(song_list, dataset, features):
    song_vectors = []

    for song in song_list:
        try:
            song_data = dataset[(dataset['name'] == song['name']) &
                                (dataset['year'] == song['year'])].iloc[0]

        except IndexError:
            song_data = None

        if song_data is None:
            print(f'Warning: {song["name"]} from {song["year"]} not found in the dataset.')
            continue

        song_vectors.append(song_data[features].values)

    return np.mean(np.array(list(song_vectors)), axis=0)

def music_recommender(song_list, dataset, pipeline, features, n_songs=10):
    groupby_input_tracks = tracks_groupby(song_list)
    song_center = input_preprocessor(song_list, dataset, features)

    if song_center is None:
        print("No valid song data found. Recommendation cannot be generated.")
        return None

    scaler = pipeline.steps[0][1]
    scaled_data = scaler.transform(dataset[features])
    scaled_song_center = scaler.transform(song_center.reshape(1, -1))

    ed_dist = euclidean_distances(scaled_song_center, scaled_data)

    index = list(np.argsort(ed_dist)[:,:n_songs][0])
    rec_output = dataset.iloc[index]

    return rec_output[metadata_cols]

def tracks_groupby(song_list):
    # Placeholder for tracks_groupby function
    pass

# Streamlit UI
st.title("Music Recommender App")

# User input
user_songs = st.text_area("Enter songs (format: 'Toosie Slide, 2020\nOutta Time (feat. Drake), 2020'):")

if user_songs:
    # Parse user input
    user_songs_list = [dict(zip(['name', 'year'], song.split(', '))) for song in user_songs.split('\n')]
    # Call the music_recommender function
    results = music_recommender(user_songs_list, tracks_df, song_cluster_pipeline, features)
    
    # Display recommendations
    if results is not None and not results.empty:
        st.subheader("Recommended Songs:")
        st.table(results)
    else:
        st.warning("No valid song data found. Recommendation cannot be generated.")


