import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb
import shap
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

# --- Load data ---
events = pd.read_csv('mock_events.csv', parse_dates=['timestamp'])
ymmt = pd.read_csv('cars_ymmt.csv')
ymmt['vehicle'] = ymmt.apply(lambda r: f"{int(r.year)} {r.make} {r.model} {r.trim}", axis=1)

# Sample 5 users
users = events['user_id'].unique()[:5]

# --- Table 1: Raw Interactions ---
raw = events[events['user_id'].isin(users)][['user_id','event_type','vehicle_id','weight']].drop_duplicates()
raw = raw.merge(ymmt[['vehicle_id','vehicle']], on='vehicle_id')[['user_id','event_type','vehicle','weight']]
print("\nRaw Interactions:")
print(raw)

# --- Table 2: Weighted Interactions ---
print("\nWeighted Interactions:")
print(raw)

# --- Feature Engineering ---
df = events.copy()
df['vehicle_id'] = df['vehicle_id']
agg = df.groupby(['user_id','vehicle_id']).agg(
    weight_sum=('weight','sum'),
    weight_max=('weight','max'),
    event_type_count=('event_type','nunique'),
    session_count=('timestamp','nunique')
).reset_index()
agg['hours_since_last'] = (pd.Timestamp.now() - df.groupby(['user_id','vehicle_id'])['timestamp'].max()).dt.total_seconds()/3600
agg = agg.merge(ymmt[['vehicle_id','year']], on='vehicle_id')
agg['vehicle_age'] = pd.Timestamp.now().year - agg['year']

# Label encoding
le_u = LabelEncoder().fit(agg['user_id'])
le_i = LabelEncoder().fit(agg['vehicle_id'])
agg['u_idx'] = le_u.transform(agg['user_id'])
agg['i_idx'] = le_i.transform(agg['vehicle_id'])

# Train XGB for feature importance
X = agg[['u_idx','i_idx','weight_sum','weight_max','event_type_count','session_count','hours_since_last','vehicle_age']]
y = (agg['weight_max'] > 0.5).astype(int)
X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.2,random_state=42)
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
bst = xgb.train({'objective':'binary:logistic','eval_metric':'auc'}, dtrain, num_boost_round=20, evals=[(dval,'val')])

# --- Plot Feature Importance ---
xgb.plot_importance(bst, max_num_features=8, title='Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()

# --- SHAP Summary ---
explainer = shap.TreeExplainer(bst)
shap_vals = explainer.shap_values(dval)
shap.summary_plot(shap_vals, X_val, show=False)
plt.tight_layout()
plt.savefig('shap_summary.png')
plt.show()

# --- Collaborative Filtering SVD ---
pivot = agg.pivot_table(index='user_id', columns='vehicle_id', values='weight_sum', fill_value=0)
svd = TruncatedSVD(n_components=2, random_state=42)
item_factors = svd.fit_transform(pivot.T)
labels = pivot.columns.tolist()

# Choose first user for known/new
user0 = users[0]
known0 = set(df[df['user_id']==user0]['vehicle_id'])
colors = ['red' if vid in known0 else 'blue' for vid in labels]

# --- Embedding Scatter ---
plt.figure()
for x,y,c,label in zip(item_factors[:,0], item_factors[:,1], colors, labels):
    plt.scatter(x, y, c=c)
    plt.text(x, y, label, fontsize=8)
plt.title(f'Embeddings Scatter (Red=Known, Blue=New) for {user0}')
plt.tight_layout()
plt.savefig('embeddings_scatter.png')
plt.show()

# --- Score Distribution Histogram ---
preds = bst.predict(xgb.DMatrix(X_val))
plt.figure()
plt.hist(preds, bins=20)
plt.title('Score Distribution Histogram')
plt.xlabel('Score'); plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('score_distribution.png')
plt.show()

# --- Coverage / Popularity Curve ---
# Generate top-5 for each user
scores_all = bst.predict(xgb.DMatrix(X))
agg['score'] = scores_all
top5 = agg.sort_values(['user_id','score'], ascending=[True,False]).groupby('user_id').head(5)
pop = top5['vehicle_id'].value_counts().reset_index()
pop.columns = ['vehicle_id','count']
pop = pop.merge(ymmt[['vehicle_id','vehicle']], on='vehicle_id').reset_index(drop=True)
pop['cum_count'] = pop['count'].cumsum()
plt.figure()
plt.plot(range(1,len(pop)+1), pop['cum_count'], marker='o')
plt.xticks(range(1,len(pop)+1), pop['vehicle'], rotation=45, ha='right')
plt.title('Coverage / Popularity Curve')
plt.ylabel('Cumulative Count')
plt.tight_layout()
plt.savefig('coverage_curve.png')
plt.show()
