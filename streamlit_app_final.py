
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from scipy.sparse import load_npz


st.set_page_config(page_title="Spotify Song Recommender", layout="centered")
st.title("🎵 Spotify Song Recommender")

data = pd.read_csv("finalDataset.csv")
tracks = pd.read_csv("tracks.csv")

valid_ids = set(data['track_id'])
tracks = tracks[tracks['track_id'].isin(valid_ids)]

similarity = load_npz("similarity_sparse.npz").toarray()

def recommend_song(track_id, top_n=5):
    if track_id not in data['track_id'].values:
        st.error("The selected track is not available in the dataset for recommendations.")
        return pd.DataFrame(columns=['track_name', 'track_artist'])

    idx = data.index[data['track_id'] == track_id][0]
    sim_scores = list(enumerate(similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:]

    song_indices = [i[0] for i in sim_scores if i[1] > 0][:top_n]

    if not song_indices:
        return pd.DataFrame(columns=['track_name', 'track_artist'])

    return tracks.iloc[song_indices][['track_name', 'track_artist']]
if __name__ == "__main__":
    
    selected_track = st.selectbox("🎧 Select a track:", tracks['track_name'].values)

    if selected_track:
        track_id = tracks[tracks['track_name'] == selected_track]['track_id'].values[0]
        recommendations = recommend_song(track_id)

        if not recommendations.empty:
            st.subheader("Recommended Songs:")
            for i, row in recommendations.iterrows():
                st.markdown(f"**{row['track_name']}** by *{row['track_artist']}*")
        else:
            st.info("No recommendations found.")