import pandas as pd
from sklearn.decomposition import TruncatedSVD
import numpy as np

# Load mock events
events = pd.read_csv('mock_events.csv', parse_dates=['timestamp'])

# Ensure 'vehicle' and 'weight' exist
assert 'vehicle' in events.columns, "'vehicle' column missing"
assert 'weight' in events.columns, "'weight' column missing"

# Sample first 5 users
users = events['user_id'].unique()[:5]

# Table 1: Raw interactions (no weight)
raw = events[events['user_id'].isin(users)][['user_id', 'event_type', 'vehicle']].drop_duplicates()
print("\n=== Raw Interactions (no weight) ===")
print(raw)

# Table 2: Weighted interactions
weighted = events[events['user_id'].isin(users)][['user_id', 'event_type', 'vehicle', 'weight']]
print("\n=== Weighted Interactions ===")
print(weighted)

# Table 3: Recommendations split (3 known + 2 new)
# Build user-item matrix
pivot = weighted.pivot_table(index='user_id', columns='vehicle', values='weight', fill_value=0)

# Apply SVD
svd = TruncatedSVD(n_components=2, random_state=42)
user_factors = svd.fit_transform(pivot)
item_factors = svd.components_.T
scores = np.dot(user_factors, item_factors.T)
scores_df = pd.DataFrame(scores, index=pivot.index, columns=pivot.columns)

# Generate recommendations
recos = []
for user in users:
    known = set(pivot.columns[pivot.loc[user] > 0])
    sorted_items = scores_df.loc[user].sort_values(ascending=False).index.tolist()
    rec_known = [v for v in sorted_items if v in known][:3]
    rec_new = [v for v in sorted_items if v not in known][:2]
    assert len(rec_known) <= 3, "Too many known recs"
    assert len(rec_new) <= 2, "Too many new recs"
    recos.append({'user_id': user, 'rec_known': rec_known, 'rec_new': rec_new})

rec_df = pd.DataFrame(recos)
print("\n=== Recommendations Split ===")
print(rec_df)
