import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pickle

cols = [
    'duration','protocol_type','service','flag','src_bytes','dst_bytes',
    'land','wrong_fragment','urgent','hot','num_failed_logins','logged_in',
    'num_compromised','root_shell','su_attempted','num_root','num_file_creations',
    'num_shells','num_access_files','num_outbound_cmds','is_host_login',
    'is_guest_login','count','srv_count','serror_rate','srv_serror_rate',
    'rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate',
    'srv_diff_host_rate','dst_host_count','dst_host_srv_count',
    'dst_host_same_srv_rate','dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate',
    'dst_host_serror_rate','dst_host_srv_serror_rate',
    'dst_host_rerror_rate','dst_host_srv_rerror_rate','label','difficulty'
]

print("Loading dataset...")
train_df = pd.read_csv('data/KDDTrain+.txt', names=cols)
test_df  = pd.read_csv('data/KDDTest+.txt',  names=cols)

print(f"Train samples : {len(train_df)}")
print(f"Test  samples : {len(test_df)}")

train_df.drop(columns=['difficulty'], inplace=True)
test_df.drop(columns=['difficulty'],  inplace=True)

print("\nEncoding categorical columns...")
for col in ['protocol_type', 'service', 'flag']:
    le = LabelEncoder()
    le.fit(pd.concat([train_df[col], test_df[col]]))
    train_df[col] = le.transform(train_df[col])
    test_df[col]  = le.transform(test_df[col])

print("Converting labels to binary...")
train_df['label'] = (train_df['label'] != 'normal').astype(int)
test_df['label']  = (test_df['label']  != 'normal').astype(int)

print(f"\nTrain label distribution:\n{train_df['label'].value_counts()}")
print(f"\nTest  label distribution:\n{test_df['label'].value_counts()}")

X_train = train_df.drop('label', axis=1).values.astype(np.float32)
y_train = train_df['label'].values
X_test  = test_df.drop('label', axis=1).values.astype(np.float32)
y_test  = test_df['label'].values

print("\nNormalizing features...")
scaler  = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

print("Saving preprocessed data...")
np.save('data/X_train.npy', X_train)
np.save('data/y_train.npy', y_train)
np.save('data/X_test.npy',  X_test)
np.save('data/y_test.npy',  y_test)

with open('data/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("\nDone. Files saved:")
print("  data/X_train.npy")
print("  data/y_train.npy")
print("  data/X_test.npy")
print("  data/y_test.npy")
print("  data/scaler.pkl")