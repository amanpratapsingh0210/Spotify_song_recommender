import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix, save_npz

# Load dataset
df = pd.read_csv("finalDataset.csv")

# Select features
features = [
    'track_popularity','playlist_id','playlist_genre','playlist_subgenre',
                 'danceability','energy','key','loudness','mode','speechiness',
                 'acousticness','instrumentalness','liveness','valence','tempo','clusters'
]

# Scale features
scaler = StandardScaler()
scaled = scaler.fit_transform(df[features])

# Compute top-10 cosine similarity using NearestNeighbors
k = 10
nn = NearestNeighbors(n_neighbors=k+1, metric='cosine', algorithm='brute')
nn.fit(scaled)
distances, indices = nn.kneighbors(scaled)

# Convert to similarity scores
similarities = 1 - distances

# Build sparse matrix
rows, cols, vals = [], [], []
for i in range(len(indices)):
    for j in range(1, k+1):  # skip self-similarity at j=0
        rows.append(i)
        cols.append(indices[i][j])
        vals.append(similarities[i][j])

sparse_sim = csr_matrix((vals, (rows, cols)), shape=(len(df), len(df)))

# Save
save_npz("similarity_sparse.npz", sparse_sim)
print("✅ Sparse similarity matrix saved as 'similarity_sparse.npz'")