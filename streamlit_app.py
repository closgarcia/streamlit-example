import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import euclidean_distances

# Load your dataset (if not already loaded)
tracks_df = pd.read_csv('./spotify-dataset-19212020-600k-tracks/tracks.csv')

# Your existing code for clustering, input_preprocessor, and Music_Recommender functions

# Create a pipeline with StandardScaler and KMeans
scaler_with_names = StandardScaler().fit(tracks_df[features])
song_cluster_pipeline = Pipeline([
    ('scaler', scaler_with_names),
    ('kmeans', KMeans(n_clusters=8, verbose=2))
], verbose=True)

def main():
    print("Welcome to the Music Recommender CLI!")
    
    # Get user input for songs
    song_name = input("Enter Song Name: ")
    song_year = int(input("Enter Year: "))
    
    # Create a list of user-input songs
    user_songs = [{'name': song_name, 'year': song_year}]

    # Call the Music_Recommender function
    recommendations = Music_Recommender(user_songs, tracks_df)

    # Display the recommendations
    print("\nRecommendations:")
    print(recommendations.to_string(index=False))

if __name__ == "__main__":
    main()


