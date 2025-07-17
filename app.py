from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import os

# Initialize Flask app
app = Flask(__name__)

# Load the dataset
# Ensure these CSV files are in the same directory as app.py or provide full paths
try:
    data = pd.read_csv('finalDataSet.csv')
    tracks = pd.read_csv('tracks.csv')
except FileNotFoundError:
    print("Error: 'finalDataSet.csv' or 'tracks.csv' not found. Please ensure they are in the same directory.")
    # Exit or handle the error appropriately
    exit()

# Calculate similarity matrix (using selected features)
# Ensure 'clusters' column exists or adjust features as needed
features = data[['track_popularity','playlist_id','playlist_genre','playlist_subgenre','danceability','energy','key','loudness','mode','speechiness','acousticness','instrumentalness','liveness','valence','tempo','clusters']]
similarity_matrix = cosine_similarity(features)

# Recommendation logic
def recommend_song(track_id, top_n=5):
    # Ensure track_id exists in data before proceeding
    if track_id not in data['track_id'].values:
        return [] # Return empty list if track_id is not found

    idx = data.index[data['track_id'] == track_id][0]
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n + 1]
    song_indices = [i[0] for i in sim_scores]
    return tracks.iloc[song_indices][['track_name', 'track_artist']].to_dict(orient='records')

@app.route('/')
def index():
    return render_template('index.html')  # Serve the HTML page

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"}), 200

@app.route('/songs', methods=['GET'])
def get_songs():
    # Clean the data by filling or dropping NaNs
    cleaned_data = tracks[['track_id', 'track_name','track_artist']].dropna()

    # Convert to JSON-compatible format
    songs = cleaned_data.to_dict(orient='records') # type: ignore
    return jsonify(songs)  # Send song list as JSON

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        track_id = int(request.form.get('track_id', 0))
        recommendations = recommend_song(track_id)
        return jsonify(recommendations)
    except ValueError:
        return jsonify({"error": "Invalid track ID"}), 400
    except IndexError:
        return jsonify({"error": "Track ID not found in dataset"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(debug=False, host='0.0.0.0', port=port)
