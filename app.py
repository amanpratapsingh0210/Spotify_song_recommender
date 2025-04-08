from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import os

# Load the dataset
data = pd.read_csv(r'finalDataSet.csv')
tracks = pd.read_csv(r'tracks.csv')

# Calculate similarity matrix (using selected features)
features = data[['track_popularity','playlist_id','playlist_genre','playlist_subgenre','danceability','energy','key','loudness','mode','speechiness','acousticness','instrumentalness','liveness','valence','tempo','clusters']]
similarity_matrix = cosine_similarity(features)

# Recommendation logic
def recommend_song(track_id, top_n=5):
    idx = data.index[data['track_id'] == track_id][0]
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n + 1]
    song_indices = [i[0] for i in sim_scores]
    return tracks.iloc[song_indices][['track_name', 'track_artist']].to_dict(orient='records')

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')  # Serve the HTML page

@app.route('/songs', methods=['GET'])
def get_songs():
    # Clean the data by filling or dropping NaNs
    cleaned_data = tracks[['track_id', 'track_name','track_artist']].dropna()

    # Convert to JSON-compatible format
    songs = cleaned_data.to_dict(orient='records')
    return jsonify(songs)  # Send song list as JSON

@app.route('/recommend', methods=['POST'])
def recommend():
    track_id = int(request.form['track_id'])
    recommendations = recommend_song(track_id)
    return jsonify(recommendations)

port = int(os.environ.get("PORT", 5000))
app.run(host="0.0.0.0", port=port, debug=False)