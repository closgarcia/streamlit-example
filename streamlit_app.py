# Your existing code for clustering, input_preprocessor, and Music_Recommender functions

# Fit the clustering model to your data
song_cluster_pipeline.fit(tracks_df[features])

# Function to preprocess input and recommend songs
def Music_Recommender(song_list, n_songs=10):
    # Assuming input_preprocessor is a function that processes the user's input
    # and returns a DataFrame with the same columns as your dataset
    user_input_df = input_preprocessor(song_list)

    # Use the trained pipeline to transform user input and predict clusters
    user_cluster = song_cluster_pipeline.predict(user_input_df[features])

    # Filter songs that belong to the same cluster as the user
    cluster_mask = song_cluster_pipeline.predict(tracks_df[features]) == user_cluster
    cluster_songs = tracks_df[cluster_mask]

    # Sort and select top n_songs based on some criteria (e.g., popularity)
    recommendations = cluster_songs.sort_values(by='popularity', ascending=False).head(n_songs)

    return recommendations

# Your existing code...

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



