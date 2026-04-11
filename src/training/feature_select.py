import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

X_train = np.load('data/X_train.npy')
y_train = np.load('data/y_train.npy')
X_test  = np.load('data/X_test.npy')

# Find most important features
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

importances = rf.feature_importances_
indices = importances.argsort()[::-1]

print("Top 20 most important features:")
for i in range(20):
    print(f"  Feature {indices[i]:2d}: {importances[indices[i]]:.4f}")

# Keep top 20 features
selector = SelectFromModel(rf, max_features=20, prefit=True)
X_train_sel = selector.transform(X_train)
X_test_sel  = selector.transform(X_test)

print(f"\nReduced from 41 → {X_train_sel.shape[1]} features")
np.save('data/X_train_selected.npy', X_train_sel)
np.save('data/X_test_selected.npy',  X_test_sel)

import pickle
with open('data/feature_selector.pkl', 'wb') as f:
    pickle.dump(selector, f)
print("Saved selected features and selector")