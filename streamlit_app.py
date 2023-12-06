import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

"""
# Welcome to Streamlit!

Edit `/streamlit_app.py` to customize this app to your heart's desire :heart:.
If you have any questions, checkout our [documentation](https://docs.streamlit.io) and [community
forums](https://discuss.streamlit.io).

In the meantime, below is an example of what you can do with just a few lines of code:
"""

num_points = st.slider("Number of points in spiral", 1, 10000, 1100)
num_turns = st.slider("Number of turns in spiral", 1, 300, 31)

indices = np.linspace(0, 1, num_points)
theta = 2 * np.pi * num_turns * indices
radius = indices

x = radius * np.cos(theta)
y = radius * np.sin(theta)

df = pd.DataFrame({
    "x": x,
    "y": y,
    "idx": indices,
    "rand": np.random.randn(num_points),
})

st.altair_chart(alt.Chart(df, height=700, width=700)
    .mark_point(filled=True)
    .encode(
        x=alt.X("x", axis=None),
        y=alt.Y("y", axis=None),
        color=alt.Color("idx", legend=None, scale=alt.Scale()),
        size=alt.Size("rand", legend=None, scale=alt.Scale(range=[1, 150])),
    ))


!pip install opendatasets

import matplotlib.pyplot as plt


import numpy as np
import opendatasets as od
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import os
dataset_url = 'https://www.kaggle.com/datasets/yamaerenay/spotify-dataset-19212020-600k-tracks'
od.download('https://www.kaggle.com/datasets/yamaerenay/spotify-dataset-19212020-600k-tracks')
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
%matplotlib inline

dataset = './spotify-dataset-19212020-600k-tracks'

import pandas as pd
tracks_df = pd.read_csv(dataset + '/tracks.csv')
artists_df = pd.read_csv(dataset + '/artists.csv')

# Merge dataframes based on 'id_artists'
merged_df = pd.merge(tracks_df, artists_df, left_on='id_artists', right_on='id', how='left')

# Drop unnecessary columns
merged_df = merged_df.drop(columns=['id_x', 'id_y', 'followers', 'genres', 'name_y'])

# Handle missing values if any
merged_df = merged_df.dropna()

print(merged_df.columns)





print(tracks_df.columns)


# Merge relevant columns based on common keys ('id_artists' column)
merged_df = pd.merge(tracks_df, artists_df, left_on='id_artists', right_on='id', how='left')

# Check the merged DataFrame to ensure everything looks good
print(merged_df.head())

# Verify the columns in the merged DataFrame
print(merged_df.columns)


# Convert lists in 'id_artists' column to tuples
tracks_df['id_artists'] = tracks_df['id_artists'].apply(tuple)

# Now, you can get unique values
unique_id_artists = tracks_df['id_artists'].unique()

# Print the unique values
print(unique_id_artists)


# Convert lists in 'id_artists' column to tuples
tracks_df['id_artists'] = tracks_df['id_artists'].apply(lambda x: tuple(x) if isinstance(x, list) else (x,))

# Keep only the first artist ID in tuples with multiple elements
tracks_df['id_artists'] = tracks_df['id_artists'].apply(lambda x: x[0])

# Now, you can get unique values
unique_id_artists = tracks_df['id_artists'].unique()

# Print the unique values
print(unique_id_artists)


# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Set the title for the entire plot
fig.suptitle('Count Plots')

# Plot count plots using merged_df DataFrame
sns.countplot(ax=axes[0], x='explicit', data=merged_df, palette='coolwarm')
sns.countplot(ax=axes[1], x='mode', data=merged_df, palette='coolwarm')

# Show the plots
plt.show()


tracks_df['release_date'] = pd.to_datetime(tracks_df['release_date'])
tracks_df['year'] = tracks_df['release_date'].apply(lambda time: time.year)

import seaborn as sns
import matplotlib.pyplot as plt
zerotoone = tracks_df[['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'valence', 'year']]
dfn = zerotoone[zerotoone['year'] > 1945].reset_index(drop=True)
dfn.set_index('year', inplace=True)

# Your DataFrame dfn should already be prepared
sns.set_style('whitegrid')
plt.figure(figsize=(12, 10))

# Plot the data with specific columns and style
sns.lineplot(data=dfn[['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'valence']], dashes=True)

# Customize the plot
plt.xlabel('Year', fontsize=14)
plt.ylabel('Values', fontsize=14)
plt.title('Trends Over Years', fontsize=16)
sns.despine(left=True, bottom=True)

# Show the plot
plt.show()


plt.figure(figsize=(8,6))
sns.histplot(x='popularity',data=tracks_df,color="olive")

df = tracks_df[tracks_df['popularity']>85][['name','artists','popularity','year','time_signature']]
fig = px.scatter(df,x='year', y='popularity',color='name',size='time_signature')
fig.show()

plt.figure(figsize=(10,6))

corr = tracks_df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr,mask=mask,vmin=-1,cmap='viridis',annot=False)

corr[abs(corr['popularity']) > 0.25]['popularity']

plt.figure(figsize=(10,8))
sns.set_style('whitegrid')
sns.scatterplot(x="danceability", y="popularity",
                hue="year", size="key",
                palette="ch:r=-.4,d=.2_r",
                sizes=(50, 300), linewidth=0,
                data=tracks_df,legend=True).set(title='danceability')
sns.despine(left = True,bottom=True)



# Check column names and data types
print(tracks_df.columns)
print(tracks_df['artists'].head())
print(tracks_df['popularity'].head())

# Convert 'artists' column to strings if it's not already
tracks_df['artists'] = tracks_df['artists'].astype(str)

# Select data for the specified artists
post = tracks_df[tracks_df['artists'].str.contains('Post Malone')]
ed = tracks_df[tracks_df['artists'].str.contains('Ed Sheeran')]
kw = tracks_df[tracks_df['artists'].str.contains('Kanye West')]
dra = tracks_df[tracks_df['artists'].str.contains('Drake')]
cb = tracks_df[tracks_df['artists'].str.contains('Chris Brown')]

# Create a figure and axis
fig, ax = plt.subplots(figsize=(20,10))

# Remove spines
sns.despine(fig, left=True, bottom=True)
sns.set_context("talk", font_scale=1, rc={"lines.linewidth": 2.5})

# Plot histograms for each artist
sns.histplot(post['popularity'], color='y', label="Post Malone")
sns.histplot(ed['popularity'], color='b', label="Ed Sheeran")
sns.histplot(kw['popularity'], color='m', label="Kanye West")
sns.histplot(dra['popularity'], color='g', label="Drake")
sns.histplot(cb['popularity'], color='r', label="Chris Brown")

# Add legend
ax.legend(fontsize=14)

# Show the plot
plt.show()


Drake has the highest popularity from these 5 artists


# Group by 'loudness' and calculate mean popularity for the top 20 loudness values
ld = tracks_df.groupby("loudness")["popularity"].mean().sort_values(ascending=False).head(20).reset_index()

# Group by 'energy' and calculate mean popularity for the top 20 energy values
en = tracks_df.groupby("energy")["popularity"].mean().sort_values(ascending=False).head(20).reset_index()

# Create a figure with subplots
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))

# Plot popularity vs. loudness
sns.pointplot(data=ld, x="loudness", y="popularity", ax=axes[0], color='b')
axes[0].set_xlabel('Loudness', fontsize=12)
axes[0].set_ylabel('Popularity', fontsize=12)
axes[0].set_title('Popularity vs Loudness', fontsize=15)
axes[0].tick_params(axis='x', rotation=90)

# Plot popularity vs. energy
sns.pointplot(data=en, x="energy", y="popularity", ax=axes[1], color='g')
axes[1].set_xlabel('Energy', fontsize=12)
axes[1].set_ylabel('Popularity', fontsize=12)
axes[1].set_title('Popularity vs Energy', fontsize=15)
axes[1].tick_params(axis='x', rotation=90)

# Adjust layout
plt.tight_layout()

# Show the plots
plt.show()


# Group by 'acousticness' and calculate mean popularity for the top 20 acousticness values
ac = tracks_df.groupby("acousticness")["popularity"].mean().sort_values(ascending=False).head(20).reset_index()

# Group by 'instrumentalness' and calculate mean popularity for the top 20 instrumentalness values
ins = tracks_df.groupby("instrumentalness")["popularity"].mean().sort_values(ascending=False).head(20).reset_index()

# Plot popularity vs. acousticness
plt.figure(figsize=(12, 6))
sns.pointplot(data=ac, x="acousticness", y="popularity", color='y')
plt.xlabel('Acousticness', fontsize=12)
plt.ylabel('Popularity', fontsize=12)
plt.title('Popularity vs Acousticness', fontsize=15)
plt.xticks(rotation='vertical')
plt.show()

# Plot popularity vs. instrumentalness
plt.figure(figsize=(12, 6))
sns.pointplot(data=ins, x="instrumentalness", y="popularity", color='r')
plt.xlabel('Instrumentalness', fontsize=12)
plt.ylabel('Popularity', fontsize=12)
plt.title('Popularity vs Instrumentalness', fontsize=15)
plt.xticks(rotation='vertical')
plt.show()


# Assuming you have loaded your dataset into a DataFrame named 'dataset'
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import metrics

# Select numerical columns from the dataset
numeric_cols = tracks_df.select_dtypes(include=[np.number])

# Standardize the data
scaler = StandardScaler()
scaled_X = scaler.fit_transform(numeric_cols)
scaled_df = pd.DataFrame(scaled_X, columns=numeric_cols.columns)

# Perform PCA with 2 components
pca = PCA(n_components=2)
pca.fit(scaled_df)
pca_df = pca.transform(scaled_df)

# Apply KMeans clustering with 8 clusters
kmeans = KMeans(n_clusters=8, verbose=2)
kmeans.fit(pca_df)

# Assign cluster labels to the original dataset
tracks_df['cluster_label'] = kmeans.predict(pca_df)

# Calculate Calinski-Harabasz score
calinski_harabasz_score = metrics.calinski_harabasz_score(pca_df, tracks_df['cluster_label'])

# Print the Calinski-Harabasz score
print("Calinski-Harabasz Score:", calinski_harabasz_score)


The Calinski-Harabasz score is a measure of clustering quality that evaluates both the separation between clusters and the compactness of clusters. A higher Calinski-Harabasz score indicates better-defined clusters.

In this case, the Calinski-Harabasz score for the KMeans clustering with *k*
=
8.
k=8 clusters is approximately
441405.12. This suggests that the data points are well-clustered into distinct and compact groups.







from sklearn.cluster import Birch
from sklearn import metrics
import pandas as pd

# Assuming pca_df is your PCA-transformed data
brc = Birch(n_clusters=20)
brc.fit(pca_df)

# Predict cluster labels
spotify_dataBirch = tracks_df.copy()
spotify_dataBirch['cluster_label'] = brc.predict(pca_df)

# Calculate Calinski-Harabasz score
calinski_harabasz_score = metrics.calinski_harabasz_score(pca_df, spotify_dataBirch['cluster_label'])
print(calinski_harabasz_score)  # Output the Calinski-Harabasz score for Birch clustering

# Initialize dict1 as an empty dictionary
dict1 = {}

# Add the score to dict1
dict1['BRICH'] = calinski_harabasz_score

# Optionally, if you want to see the number of samples in each cluster
cluster_counts = spotify_dataBirch['cluster_label'].value_counts()
print(cluster_counts)


from sklearn.cluster import MiniBatchKMeans
from sklearn import metrics

# Assuming pca_df is your PCA-transformed data
minikmeans = MiniBatchKMeans(n_clusters=6, random_state=23, batch_size=32)
minikmeans.fit(pca_df)

# Predict cluster labels
spotify_minikmeans = tracks_df.copy()
spotify_minikmeans['cluster_label'] = minikmeans.predict(pca_df)

# Calculate Calinski-Harabasz score
calinski_harabasz_score = metrics.calinski_harabasz_score(pca_df, spotify_minikmeans['cluster_label'])
print(calinski_harabasz_score)  # Output the Calinski-Harabasz score for MiniBatchKMeans clustering

# Add the score to dict1
dict1['minibatch kmeans'] = calinski_harabasz_score

# Optionally, if you want to see the number of samples in each cluster
cluster_counts = spotify_minikmeans['cluster_label'].value_counts()
print(cluster_counts)


# Install the fuzzy-c-means package
!pip install fuzzy-c-means --quiet

# Import necessary libraries
from fcmeans import FCM
from sklearn import metrics

# Assuming pca_df is your PCA-transformed data
fcm = FCM(n_clusters=8)
fcm.fit(pca_df)

# Predict cluster labels using Fuzzy C-means
spotify_dataFuzzy = tracks_df.copy()
spotify_dataFuzzy['cluster_label'] = fcm.predict(pca_df)

# Calculate Calinski-Harabasz score
calinski_harabasz_score = metrics.calinski_harabasz_score(pca_df, spotify_dataFuzzy['cluster_label'])
print(calinski_harabasz_score)  # Output the Calinski-Harabasz score for Fuzzy C-means clustering

# Add the score to dict1
dict1['fuzzy c'] = calinski_harabasz_score


from sklearn.mixture import GaussianMixture
from sklearn import metrics

# Assuming pca_df is your PCA-transformed data
gm = GaussianMixture(n_components=7, random_state=23)
gm.fit(pca_df)

# Predict cluster labels using Gaussian Mixture
spotify_gm = tracks_df.copy()  # Assuming tracks_df is your original dataset
spotify_gm['cluster_label'] = gm.predict(pca_df)

# Calculate Calinski-Harabasz score
calinski_harabasz_score = metrics.calinski_harabasz_score(pca_df, spotify_gm['cluster_label'])
print(calinski_harabasz_score)  # Output the Calinski-Harabasz score for Gaussian Mixture clustering

# Add the score to dict1
dict1['Gaussian Mixture'] = calinski_harabasz_score


import pandas as pd
import plotly.express as px

# Assuming dict1 contains your cluster labels as keys and corresponding scores as values
dict1 = {
    'KMeans': 441405.1171084726,
    'BRICH': 183096.66712907582,
    'MiniBatch KMeans': 525157.673555993,
    'Fuzzy C-means': 566236.4181428673,
    'Gaussian Mixture': 553023.4780984044
}

# Create a DataFrame from dict1 and set the index
m = pd.DataFrame(list(dict1.items()), columns=['cluster', 'score'])
m.set_index('cluster', inplace=True)

# Plot the data using Plotly Express
fig = px.bar(m, orientation='h')
fig.show()


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import euclidean_distances
from collections import defaultdict

# Define features and metadata columns
features = ['valence', 'year', 'acousticness',
            'danceability', 'duration_ms', 'energy',
            'explicit','instrumentalness', 'key',
            'liveness', 'loudness', 'mode',
            'popularity','speechiness', 'tempo']

metadata_cols = ['year', 'name',  'artists']

# Create a pipeline with StandardScaler and KMeans
song_cluster_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('kmeans', KMeans(n_clusters=8, verbose=2))  # Adjust the number of clusters as needed
], verbose=True)

# Extract features for clustering
X = tracks_df[features]

# Fit the pipeline to the data
song_cluster_pipeline.fit(X)


def input_preprocessor(song_list, dataset):

    song_vectors = []

    for song in song_list:
        try:
            song_data = tracks_df[(dataset['name'] == song['name']) &
                                (dataset['year'] == song['year'])].iloc[0]

        except IndexError:
            song_data = None

        if song_data is None:
            print('Warning: {} does not exist in our database'.format(song['name']))
            continue

        song_vectors.append(song_data[features].values)

    return np.mean(np.array(list(song_vectors)), axis=0)

def Music_Recommender(song_list, dataset, n_songs=10):

    groupby_input_tracks = tracks_groupby(song_list)
    song_center = input_preprocessor(song_list, dataset)


    scaler = song_cluster_pipeline.steps[0][1]
    scaled_data = scaler.transform(dataset[features])
    scaled_song_center = scaler.transform(song_center.reshape(1, -1))


    ed_dist = euclidean_distances(scaled_song_center, scaled_data)


    index = list(np.argsort(ed_dist)[:,:n_songs][0])
    rec_output = dataset.iloc[index]


    return rec_output[metadata_cols]

import numpy as np

def input_preprocessor(song_list, tracks_df, features):
    """
    Preprocesses the input song list and creates feature vectors.

    Args:
        song_list (list): List of dictionaries containing song information.
        dataset (pd.DataFrame): DataFrame containing the dataset.
        features (list): List of feature column names in the dataset.

    Returns:
        np.array: Mean feature vector of the input songs.
    """
    song_vectors = []

    for song in song_list:
        try:
            song_data = tracks_df[(tracks_df['name'] == song['name']) &
                                (tracks_df['year'] == song['year'])].iloc[0]

        except IndexError:
            song_data = None

        if song_data is None:
            print('Warning: {} does not exist in our database'.format(song['name']))
            continue

        song_vectors.append(song_data[features].values)

    return np.mean(np.array(list(song_vectors)), axis=0)


import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

def tracks_groupby(song_list):
    # Implementation of tracks_groupby function (Assumed)
    pass

def input_preprocessor(song_list, tracks_df, features):
    # Implementation of input_preprocessor function (Assumed)
    pass

def Music_Recommender(song_list, tracks_df, n_songs=10):
    groupby_input_tracks = tracks_groupby(song_list)
    song_center = input_preprocessor(song_list, tracks_df, features)

    scaler = song_cluster_pipeline.steps[0][1]
    scaled_data = scaler.transform(tracks_df[features])
    scaled_song_center = scaler.transform(song_center.reshape(1, -1))

    ed_dist = euclidean_distances(scaled_song_center, scaled_data)

    index = list(np.argsort(ed_dist)[:,:n_songs][0])
    rec_output = tracks_df.iloc[index]

    return rec_output[metadata_cols]

def Music_Recommender(song_list, tracks_df, n_songs=10):
    groupby_input_tracks = tracks_groupby(song_list)
    song_center = input_preprocessor(song_list, tracks_df, features)

    if song_center is None:
        print("No valid song data found. Recommendation cannot be generated.")
        return None

    scaler = song_cluster_pipeline.steps[0][1]
    scaled_data = scaler.transform(tracks_df[features])
    scaled_song_center = scaler.transform(song_center.reshape(1, -1))

    ed_dist = euclidean_distances(scaled_song_center, scaled_data)

    index = list(np.argsort(ed_dist)[:,:n_songs][0])
    rec_output = tracks_df.iloc[index]

    return rec_output[metadata_cols]


def input_preprocessor(song_list, tracks_df, features):
    song_vectors = []

    for song in song_list:
        try:
            song_data = tracks_df[(tracks_df['name'] == song['name']) & (tracks_df['year'] == song['year'])].iloc[0]

        except IndexError:
            song_data = None

        if song_data is None:
            print(f'Warning: {song["name"]} from {song["year"]} not found in the dataset.')
            continue

        song_vectors.append(song_data[features].values)

    return np.mean(np.array(list(song_vectors)), axis=0)


results = Music_Recommender([
    {'name': 'Toosie Slide', 'year': 2020},
    {'name': 'Outta Time (feat. Drake)', 'year': 2020},
    {'name': 'Chicago Freestyle (feat. Giveon)', 'year': 2020}
], tracks_df)
print(results)


# Assuming 'results' is the DataFrame containing the recommendations
formatted_results = results.to_string(index=False, header=True)

print(formatted_results)

results = Music_Recommender([
    {'name': 'Toosie Slide', 'year': 2020},
    {'name': 'Outta Time (feat. Drake)', 'year': 2020},
    {'name': 'Chicago Freestyle (feat. Giveon)', 'year': 2020}
], tracks_df)
print(results)

deduplicated_df = tracks_df.drop_duplicates(subset=['name', 'year'])


deduplicated_df = tracks_df.drop_duplicates(subset=['name', 'year'])

from sklearn.preprocessing import StandardScaler
import pandas as pd

# Create a DataFrame with the features
features_df = tracks_df[features]

# Create a new StandardScaler instance and fit it with the feature DataFrame
scaler_with_names = StandardScaler().fit(features_df)

# Modify the pipeline to use the new scaler instance
song_cluster_pipeline = Pipeline([
    ('scaler', scaler_with_names),
    ('kmeans', KMeans(n_clusters=8, verbose=2))
], verbose=True)

# Rest of your code remains unchanged

# Modify the input dictionary for Sublime's songs
sublime_songs = [
    {'name': 'Badfish', 'year': 1992},
    {'name': 'Santeria', 'year': 1996}
]

# Call Music_Recommender function with Sublime's songs
results_sublime = Music_Recommender(sublime_songs, tracks_df)

# Print the formatted recommendations
formatted_results_sublime = results_sublime.to_string(index=False, header=True)
print(formatted_results_sublime)
