import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

st.title("🔍 Debug Mode: App Started")

try:
    data = pd.read_csv("finalDataset.csv")
    st.success("✅ finalDataset.csv loaded")
except Exception as e:
    st.error(f"❌ Failed to load finalDataset.csv: {e}")
    st.stop()

try:
    tracks = pd.read_csv("tracks.csv")
    st.success("✅ tracks.csv loaded")
except Exception as e:
    st.error(f"❌ Failed to load tracks.csv: {e}")
    st.stop()

# Calculate similarity matrix
features = data[['track_popularity','playlist_id','playlist_genre','playlist_subgenre',
                 'danceability','energy','key','loudness','mode','speechiness',
                 'acousticness','instrumentalness','liveness','valence','tempo','clusters']]
similarity_matrix = cosine_similarity(features)

# Recommendation function
def recommend_song(track_id, top_n=5):
    idx = data.index[data['track_id'] == track_id][0]
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n + 1]
    song_indices = [i[0] for i in sim_scores]
    return tracks.iloc[song_indices][['track_name', 'track_artist']]

st.set_page_config(page_title="Music Recommender", page_icon="🎶", layout="centered")

st.title("🎵 Music Recommendation System")

# Prepare dropdown
cleaned_tracks = tracks[['track_id', 'track_name', 'track_artist']].dropna()
track_options = cleaned_tracks.apply(lambda x: f"{x['track_name']} - {x['track_artist']}", axis=1)
track_map = dict(zip(track_options, cleaned_tracks['track_id']))

selected_track = st.selectbox("Select a song to get recommendations:", track_options)

if st.button("Get Recommendations"):
    track_id = track_map[selected_track]
    recommendations = recommend_song(track_id)

    st.subheader("Recommended Songs:")
    for i, row in recommendations.iterrows():
        st.write(f"**{row['track_name']}** by *{row['track_artist']}*")