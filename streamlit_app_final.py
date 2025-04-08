
import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Spotify Song Recommender", layout="centered")
st.title("🎵 Spotify Song Recommender")

# Debug block for loading datasets
try:
    data = pd.read_csv("finalDataset.csv")
    st.success("✅ finalDataset.csv loaded")
except Exception as e:
    st.error(f"❌ Error loading finalDataset.csv: {e}")
    st.stop()

try:
    tracks = pd.read_csv("tracks.csv")
    st.success("✅ tracks.csv loaded")
except Exception as e:
    st.error(f"❌ Error loading tracks.csv: {e}")
    st.stop()

# Filter tracks to match only track_ids present in data
valid_ids = set(data['track_id'])
tracks = tracks[tracks['track_id'].isin(valid_ids)]

# Load similarity matrix
try:
    with open("similarity.pkl", "rb") as f:
        similarity_matrix = pickle.load(f)
    st.success("✅ similarity.pkl loaded")
except Exception as e:
    st.error(f"❌ Error loading similarity.pkl: {e}")
    st.stop()

# Recommendation function
def recommend_song(track_id, top_n=5):
    if track_id not in data['track_id'].values:
        st.error("❌ The selected track is not available in the dataset for recommendations.")
        return pd.DataFrame(columns=['track_name', 'track_artist'])
    
    idx = data.index[data['track_id'] == track_id][0]
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n + 1]
    song_indices = [i[0] for i in sim_scores]
    return tracks.iloc[song_indices][['track_name', 'track_artist']]

# User input
try:
    selected_track = st.selectbox("🎧 Select a track:", tracks['track_name'].values)

    if selected_track:
        track_id = tracks[tracks['track_name'] == selected_track]['track_id'].values[0]
        recommendations = recommend_song(track_id)

        if not recommendations.empty:
            st.subheader("🔁 Recommended Songs:")
            for i, row in recommendations.iterrows():
                st.markdown(f"**{row['track_name']}** by *{row['track_artist']}*")
        else:
            st.info("No recommendations found.")
except Exception as e:
    st.error("❌ An unexpected error occurred:")
    st.exception(e)
