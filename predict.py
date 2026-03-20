import torch
import numpy as np
import pickle
from model import build_model

def load_model(bits=8):
    model = build_model(bits)
    model.load_state_dict(torch.load(f'models/model_{bits}bit.pt', map_location='cpu'))
    model.eval()
    return model

def load_scaler():
    with open('data/scaler.pkl', 'rb') as f:
        return pickle.load(f)

def predict(sample_dict, model, scaler):
    cols = [
        'duration','protocol_type','service','flag','src_bytes','dst_bytes',
        'land','wrong_fragment','urgent','hot','num_failed_logins','logged_in',
        'num_compromised','root_shell','su_attempted','num_root',
        'num_file_creations','num_shells','num_access_files','num_outbound_cmds',
        'is_host_login','is_guest_login','count','srv_count','serror_rate',
        'srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate',
        'diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count',
        'dst_host_same_srv_rate','dst_host_diff_srv_rate',
        'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate',
        'dst_host_serror_rate','dst_host_srv_serror_rate',
        'dst_host_rerror_rate','dst_host_srv_rerror_rate'
    ]
    features = np.array([[sample_dict[c] for c in cols]], dtype=np.float32)
    features = scaler.transform(features)
    tensor   = torch.tensor(features)

    with torch.no_grad():
        out    = model(tensor)
        pred   = out.argmax(dim=1).item()
        probs  = torch.softmax(out, dim=1).numpy()[0]

    label      = "ATTACK" if pred == 1 else "NORMAL"
    confidence = probs[pred] * 100
    return label, confidence, probs

model  = load_model(bits=8)
scaler = load_scaler()

print("=" * 50)
print("  IDS Live Prediction — NSL-KDD features")
print("=" * 50)

normal_traffic = {
    'duration': 0, 'protocol_type': 2, 'service': 20, 'flag': 10,
    'src_bytes': 215, 'dst_bytes': 45076, 'land': 0, 'wrong_fragment': 0,
    'urgent': 0, 'hot': 1, 'num_failed_logins': 0, 'logged_in': 1,
    'num_compromised': 0, 'root_shell': 0, 'su_attempted': 0, 'num_root': 0,
    'num_file_creations': 0, 'num_shells': 0, 'num_access_files': 0,
    'num_outbound_cmds': 0, 'is_host_login': 0, 'is_guest_login': 0,
    'count': 1, 'srv_count': 1, 'serror_rate': 0.0, 'srv_serror_rate': 0.0,
    'rerror_rate': 0.0, 'srv_rerror_rate': 0.0, 'same_srv_rate': 1.0,
    'diff_srv_rate': 0.0, 'srv_diff_host_rate': 0.0, 'dst_host_count': 255,
    'dst_host_srv_count': 255, 'dst_host_same_srv_rate': 1.0,
    'dst_host_diff_srv_rate': 0.0, 'dst_host_same_src_port_rate': 0.0,
    'dst_host_srv_diff_host_rate': 0.0, 'dst_host_serror_rate': 0.0,
    'dst_host_srv_serror_rate': 0.0, 'dst_host_rerror_rate': 0.0,
    'dst_host_srv_rerror_rate': 0.0
}

dos_attack = {
    'duration': 0, 'protocol_type': 2, 'service': 20, 'flag': 10,
    'src_bytes': 0, 'dst_bytes': 0, 'land': 0, 'wrong_fragment': 0,
    'urgent': 0, 'hot': 0, 'num_failed_logins': 0, 'logged_in': 0,
    'num_compromised': 0, 'root_shell': 0, 'su_attempted': 0, 'num_root': 0,
    'num_file_creations': 0, 'num_shells': 0, 'num_access_files': 0,
    'num_outbound_cmds': 0, 'is_host_login': 0, 'is_guest_login': 0,
    'count': 511, 'srv_count': 511, 'serror_rate': 1.0, 'srv_serror_rate': 1.0,
    'rerror_rate': 0.0, 'srv_rerror_rate': 0.0, 'same_srv_rate': 1.0,
    'diff_srv_rate': 0.0, 'srv_diff_host_rate': 0.0, 'dst_host_count': 255,
    'dst_host_srv_count': 255, 'dst_host_same_srv_rate': 1.0,
    'dst_host_diff_srv_rate': 0.0, 'dst_host_same_src_port_rate': 1.0,
    'dst_host_srv_diff_host_rate': 0.0, 'dst_host_serror_rate': 1.0,
    'dst_host_srv_serror_rate': 1.0, 'dst_host_rerror_rate': 0.0,
    'dst_host_srv_rerror_rate': 0.0
}

port_scan = {
    'duration': 0, 'protocol_type': 1, 'service': 0, 'flag': 5,
    'src_bytes': 0, 'dst_bytes': 0, 'land': 0, 'wrong_fragment': 0,
    'urgent': 0, 'hot': 0, 'num_failed_logins': 0, 'logged_in': 0,
    'num_compromised': 0, 'root_shell': 0, 'su_attempted': 0, 'num_root': 0,
    'num_file_creations': 0, 'num_shells': 0, 'num_access_files': 0,
    'num_outbound_cmds': 0, 'is_host_login': 0, 'is_guest_login': 0,
    'count': 1, 'srv_count': 1, 'serror_rate': 0.0, 'srv_serror_rate': 0.0,
    'rerror_rate': 1.0, 'srv_rerror_rate': 1.0, 'same_srv_rate': 0.0,
    'diff_srv_rate': 1.0, 'srv_diff_host_rate': 0.0, 'dst_host_count': 30,
    'dst_host_srv_count': 10, 'dst_host_same_srv_rate': 0.03,
    'dst_host_diff_srv_rate': 0.97, 'dst_host_same_src_port_rate': 0.0,
    'dst_host_srv_diff_host_rate': 0.0, 'dst_host_serror_rate': 0.0,
    'dst_host_srv_serror_rate': 0.0, 'dst_host_rerror_rate': 1.0,
    'dst_host_srv_rerror_rate': 1.0
}

samples = {
    "Normal web browsing" : normal_traffic,
    "DoS attack"          : dos_attack,
    "Port scan"           : port_scan,
}

for name, sample in samples.items():
    label, confidence, probs = predict(sample, model, scaler)
    status = "✓" if label == "NORMAL" else "✗"
    print(f"\n  {status} {name}")
    print(f"    Prediction : {label}")
    print(f"    Confidence : {confidence:.1f}%")
    print(f"    Normal: {probs[0]*100:.1f}%  Attack: {probs[1]*100:.1f}%")

print("\n" + "=" * 50)
print("  Run your own sample:")
print("  Edit the sample dicts at the bottom of predict.py")
print("=" * 50)
