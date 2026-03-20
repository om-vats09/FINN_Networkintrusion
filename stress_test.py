import torch
import numpy as np
import pickle
import time
from model import build_model

def load_model(bits=8):
    model = build_model(bits)
    model.load_state_dict(torch.load(f'models/model_{bits}bit.pt', map_location='cpu'))
    model.eval()
    return model

def load_scaler():
    with open('data/scaler.pkl', 'rb') as f:
        return pickle.load(f)

def predict_sample(sample, model, scaler):
    features = np.array([list(sample.values())], dtype=np.float32)
    features = scaler.transform(features)
    tensor   = torch.tensor(features)
    with torch.no_grad():
        out    = model(tensor)
        pred   = out.argmax(dim=1).item()
        probs  = torch.softmax(out, dim=1).numpy()[0]
    return ("ATTACK" if pred == 1 else "NORMAL"), probs[pred] * 100

model  = load_model(8)
scaler = load_scaler()

BASE = {
    'duration': 0, 'protocol_type': 2, 'service': 20, 'flag': 10,
    'src_bytes': 0, 'dst_bytes': 0, 'land': 0, 'wrong_fragment': 0,
    'urgent': 0, 'hot': 0, 'num_failed_logins': 0, 'logged_in': 0,
    'num_compromised': 0, 'root_shell': 0, 'su_attempted': 0, 'num_root': 0,
    'num_file_creations': 0, 'num_shells': 0, 'num_access_files': 0,
    'num_outbound_cmds': 0, 'is_host_login': 0, 'is_guest_login': 0,
    'count': 1, 'srv_count': 1, 'serror_rate': 0.0, 'srv_serror_rate': 0.0,
    'rerror_rate': 0.0, 'srv_rerror_rate': 0.0, 'same_srv_rate': 1.0,
    'diff_srv_rate': 0.0, 'srv_diff_host_rate': 0.0, 'dst_host_count': 1,
    'dst_host_srv_count': 1, 'dst_host_same_srv_rate': 1.0,
    'dst_host_diff_srv_rate': 0.0, 'dst_host_same_src_port_rate': 0.0,
    'dst_host_srv_diff_host_rate': 0.0, 'dst_host_serror_rate': 0.0,
    'dst_host_srv_serror_rate': 0.0, 'dst_host_rerror_rate': 0.0,
    'dst_host_srv_rerror_rate': 0.0
}

def make(overrides):
    s = BASE.copy()
    s.update(overrides)
    return s

# ─────────────────────────────────────────────
# PART 1: Synthetic attack traffic
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("  PART 1 — Synthetic Attack Traffic")
print("="*60)

synthetic = {
    "Normal HTTP browsing"    : make({'src_bytes': 215, 'dst_bytes': 45076, 'logged_in': 1,
                                       'count': 1, 'dst_host_count': 255, 'dst_host_srv_count': 255,
                                       'same_srv_rate': 1.0, 'dst_host_same_srv_rate': 1.0}),

    "SYN flood (DoS)"         : make({'count': 511, 'srv_count': 511, 'serror_rate': 1.0,
                                       'srv_serror_rate': 1.0, 'dst_host_serror_rate': 1.0,
                                       'dst_host_srv_serror_rate': 1.0,
                                       'dst_host_same_src_port_rate': 1.0, 'dst_host_count': 255}),

    "UDP flood (DoS)"         : make({'protocol_type': 1, 'count': 511, 'srv_count': 511,
                                       'serror_rate': 0.0, 'same_srv_rate': 1.0,
                                       'dst_host_count': 255, 'dst_host_srv_count': 255,
                                       'dst_host_same_src_port_rate': 1.0}),

    "Port scan (probe)"       : make({'protocol_type': 1, 'flag': 5, 'count': 1,
                                       'rerror_rate': 1.0, 'srv_rerror_rate': 1.0,
                                       'diff_srv_rate': 1.0, 'dst_host_count': 30,
                                       'dst_host_srv_count': 10, 'dst_host_diff_srv_rate': 0.97,
                                       'dst_host_rerror_rate': 1.0}),

    "FTP brute force (R2L)"   : make({'service': 10, 'src_bytes': 105, 'dst_bytes': 146,
                                       'num_failed_logins': 5, 'logged_in': 0,
                                       'count': 1, 'dst_host_count': 1, 'duration': 2}),

    "Root shell (U2R)"        : make({'logged_in': 1, 'root_shell': 1, 'su_attempted': 1,
                                       'num_root': 5, 'num_file_creations': 3,
                                       'hot': 10, 'num_shells': 2, 'src_bytes': 1000,
                                       'dst_bytes': 5000, 'duration': 10}),

    "Ping sweep (probe)"      : make({'protocol_type': 0, 'service': 49, 'flag': 10,
                                       'count': 255, 'srv_count': 1, 'diff_srv_rate': 1.0,
                                       'dst_host_count': 255, 'dst_host_diff_srv_rate': 1.0}),

    "Normal FTP session"      : make({'service': 10, 'src_bytes': 2360, 'dst_bytes': 4580,
                                       'logged_in': 1, 'count': 2, 'same_srv_rate': 1.0,
                                       'dst_host_count': 10, 'dst_host_srv_count': 10}),

    "Normal SSH session"      : make({'service': 55, 'src_bytes': 3428, 'dst_bytes': 14728,
                                       'logged_in': 1, 'duration': 45, 'count': 1,
                                       'dst_host_count': 5, 'dst_host_srv_count': 5}),

    "ICMP flood (DoS)"        : make({'protocol_type': 0, 'count': 511, 'srv_count': 511,
                                       'same_srv_rate': 1.0, 'dst_host_count': 255,
                                       'dst_host_same_src_port_rate': 1.0,
                                       'dst_host_srv_count': 255}),
}

correct = 0
expected = {
    "Normal HTTP browsing"  : "NORMAL",
    "SYN flood (DoS)"       : "ATTACK",
    "UDP flood (DoS)"       : "ATTACK",
    "Port scan (probe)"     : "ATTACK",
    "FTP brute force (R2L)" : "ATTACK",
    "Root shell (U2R)"      : "ATTACK",
    "Ping sweep (probe)"    : "ATTACK",
    "Normal FTP session"    : "NORMAL",
    "Normal SSH session"    : "NORMAL",
    "ICMP flood (DoS)"      : "ATTACK",
}

for name, sample in synthetic.items():
    label, conf = predict_sample(sample, model, scaler)
    exp  = expected[name]
    ok   = label == exp
    correct += ok
    icon = "\u2705" if ok else "\u274c"
    print(f"\n  {icon} {name}")
    print(f"     Predicted : {label} ({conf:.1f}%)  |  Expected: {exp}")

print(f"\n  Score: {correct}/{len(synthetic)} correct")

# ─────────────────────────────────────────────
# PART 2: Edge cases and adversarial samples
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("  PART 2 — Edge Cases and Adversarial Samples")
print("="*60)

edge_cases = {
    "All zeros (blank packet)"        : make({}),
    "All max values"                  : make({'src_bytes': 1e9, 'dst_bytes': 1e9,
                                              'count': 511, 'srv_count': 511,
                                              'dst_host_count': 255, 'duration': 9999}),
    "Slow DoS (low rate)"             : make({'count': 2, 'serror_rate': 1.0,
                                              'duration': 300, 'src_bytes': 10}),
    "Disguised attack (normal bytes)" : make({'src_bytes': 215, 'dst_bytes': 45076,
                                              'logged_in': 1, 'root_shell': 1,
                                              'num_root': 3, 'su_attempted': 1}),
    "Single failed login"             : make({'num_failed_logins': 1, 'logged_in': 0,
                                              'service': 10, 'src_bytes': 100,
                                              'dst_bytes': 100}),
    "Borderline traffic (mixed)"      : make({'count': 5, 'serror_rate': 0.4,
                                              'rerror_rate': 0.3, 'same_srv_rate': 0.5,
                                              'diff_srv_rate': 0.5, 'dst_host_count': 50}),
    "Evasion: fragmented (wrong_frag)": make({'wrong_fragment': 3, 'src_bytes': 100,
                                              'urgent': 1, 'count': 1}),
    "Land attack (src=dst)"           : make({'land': 1, 'src_bytes': 0,
                                              'dst_bytes': 0, 'count': 1}),
}

for name, sample in edge_cases.items():
    label, conf = predict_sample(sample, model, scaler)
    print(f"\n  [{label:6s}] {conf:5.1f}%  |  {name}")

# ─────────────────────────────────────────────
# PART 3: Speed benchmark
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("  PART 3 — Speed Benchmark")
print("="*60)

X_test = np.load('data/X_test.npy')
y_test = np.load('data/y_test.npy')

print("\n  Single-sample inference (simulates real-time packet classification):")
sample  = torch.tensor(X_test[:1])
WARMUP  = 100
RUNS    = 10000

for _ in range(WARMUP):
    with torch.no_grad():
        _ = model(sample)

start = time.perf_counter()
for _ in range(RUNS):
    with torch.no_grad():
        _ = model(sample)
end = time.perf_counter()

elapsed   = end - start
per_pred  = (elapsed / RUNS) * 1000
per_sec   = RUNS / elapsed
print(f"    {RUNS} predictions in {elapsed:.3f}s")
print(f"    Per prediction : {per_pred:.4f} ms")
print(f"    Throughput     : {per_sec:,.0f} predictions/second")

print("\n  Batch inference (256 samples at a time):")
batch     = torch.tensor(X_test[:256])
RUNS_B    = 1000

for _ in range(100):
    with torch.no_grad():
        _ = model(batch)

start = time.perf_counter()
for _ in range(RUNS_B):
    with torch.no_grad():
        _ = model(batch)
end   = time.perf_counter()

elapsed_b  = end - start
per_batch  = (elapsed_b / RUNS_B) * 1000
throughput = (RUNS_B * 256) / elapsed_b
print(f"    {RUNS_B} batches (256 each) in {elapsed_b:.3f}s")
print(f"    Per batch  : {per_batch:.4f} ms")
print(f"    Throughput : {throughput:,.0f} samples/second")

print("\n  Per bit-width comparison:")
print(f"  {'Model':<10} {'ms/pred':>10} {'preds/sec':>14}")
print(f"  {'-'*36}")
for bits in [2, 4, 8]:
    m   = load_model(bits)
    s   = torch.tensor(X_test[:1])
    for _ in range(WARMUP):
        with torch.no_grad(): _ = m(s)
    t0  = time.perf_counter()
    for _ in range(RUNS):
        with torch.no_grad(): _ = m(s)
    t1  = time.perf_counter()
    ms  = ((t1 - t0) / RUNS) * 1000
    ps  = RUNS / (t1 - t0)
    print(f"  {bits}-bit{'':<6} {ms:>10.4f} {ps:>14,.0f}")

print("\nDone.")