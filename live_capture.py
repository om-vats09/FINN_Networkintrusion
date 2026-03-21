from scapy.all import sniff, IP, TCP, UDP
import numpy as np
import torch
import pickle
import time
from collections import defaultdict
import torch.nn as nn
import brevitas.nn as qnn

def build_model():
    return nn.Sequential(
        qnn.QuantLinear(41, 64,  bias=True, weight_bit_width=8),
        qnn.QuantReLU(bit_width=8),
        qnn.QuantLinear(64, 128, bias=True, weight_bit_width=8),
        qnn.QuantReLU(bit_width=8),
        qnn.QuantLinear(128, 64, bias=True, weight_bit_width=8),
        qnn.QuantReLU(bit_width=8),
        qnn.QuantLinear(64, 2,   bias=True, weight_bit_width=8),
    )

print("Loading model and scaler...")
model = build_model()
model.load_state_dict(torch.load('models/model_8bit.pt', map_location='cpu'))
model.eval()

with open('data/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# ── Connection tracker ────────────────────────────────────────────
connections = defaultdict(lambda: {
    'start'         : time.time(),
    'src_bytes'     : 0,
    'dst_bytes'     : 0,
    'src_pkts'      : 0,
    'dst_pkts'      : 0,
    'syn'           : 0,
    'fin'           : 0,
    'rst'           : 0,
    'duration'      : 0,
})

recent_conns = []

def extract_features(conn, proto, service):
    duration     = max(0, time.time() - conn['start'])
    src_bytes    = conn['src_bytes']
    dst_bytes    = conn['dst_bytes']
    count        = min(len(recent_conns), 511)
    serror_rate  = conn['syn'] / max(conn['src_pkts'], 1)
    rerror_rate  = conn['rst'] / max(conn['src_pkts'], 1)
    same_srv     = sum(1 for c in recent_conns[-100:]
                       if c.get('service') == service) / max(len(recent_conns[-100:]), 1)

    features = [
        duration,           # 0  duration
        proto,              # 1  protocol_type (0=icmp,1=udp,2=tcp)
        service,            # 2  service (simplified)
        10,                 # 3  flag (SF=10)
        src_bytes,          # 4  src_bytes
        dst_bytes,          # 5  dst_bytes
        0,                  # 6  land
        0,                  # 7  wrong_fragment
        0,                  # 8  urgent
        0,                  # 9  hot
        0,                  # 10 num_failed_logins
        1 if dst_bytes > 0 else 0,  # 11 logged_in
        0,                  # 12 num_compromised
        0,                  # 13 root_shell
        0,                  # 14 su_attempted
        0,                  # 15 num_root
        0,                  # 16 num_file_creations
        0,                  # 17 num_shells
        0,                  # 18 num_access_files
        0,                  # 19 num_outbound_cmds
        0,                  # 20 is_host_login
        0,                  # 21 is_guest_login
        count,              # 22 count
        count,              # 23 srv_count
        serror_rate,        # 24 serror_rate
        serror_rate,        # 25 srv_serror_rate
        rerror_rate,        # 26 rerror_rate
        rerror_rate,        # 27 srv_rerror_rate
        same_srv,           # 28 same_srv_rate
        1 - same_srv,       # 29 diff_srv_rate
        0,                  # 30 srv_diff_host_rate
        min(count, 255),    # 31 dst_host_count
        min(count, 255),    # 32 dst_host_srv_count
        same_srv,           # 33 dst_host_same_srv_rate
        1 - same_srv,       # 34 dst_host_diff_srv_rate
        0,                  # 35 dst_host_same_src_port_rate
        0,                  # 36 dst_host_srv_diff_host_rate
        serror_rate,        # 37 dst_host_serror_rate
        serror_rate,        # 38 dst_host_srv_serror_rate
        rerror_rate,        # 39 dst_host_rerror_rate
        rerror_rate,        # 40 dst_host_srv_rerror_rate
    ]
    return np.array(features, dtype=np.float32)

def predict(features):
    scaled = scaler.transform(features.reshape(1, -1))
    tensor = torch.tensor(scaled, dtype=torch.float32)
    with torch.no_grad():
        out   = model(tensor)
        pred  = out.argmax(dim=1).item()
        probs = torch.softmax(out, dim=1).numpy()[0]
    return pred, float(probs[pred]) * 100

# ── Packet handler ────────────────────────────────────────────────
packet_count = 0

def handle_packet(pkt):
    global packet_count
    if not pkt.haslayer(IP):
        return

    packet_count += 1
    src  = pkt[IP].src
    dst  = pkt[IP].dst
    proto = 0

    if pkt.haslayer(TCP):
        proto   = 2
        service = pkt[TCP].dport % 70
        key     = (src, dst, pkt[TCP].sport, pkt[TCP].dport)
        conn    = connections[key]
        conn['src_bytes'] += len(pkt)
        conn['src_pkts']  += 1
        flags = pkt[TCP].flags
        if flags & 0x02: conn['syn'] += 1
        if flags & 0x04: conn['rst'] += 1
        if flags & 0x01: conn['fin'] += 1

    elif pkt.haslayer(UDP):
        proto   = 1
        service = pkt[UDP].dport % 70
        key     = (src, dst, pkt[UDP].sport, pkt[UDP].dport)
        conn    = connections[key]
        conn['src_bytes'] += len(pkt)
        conn['src_pkts']  += 1
    else:
        return

    if packet_count % 10 == 0:
        features = extract_features(conn, proto, service)
        pred, conf = predict(features)
        label    = 'ATTACK' if pred == 1 else 'NORMAL'
        icon     = '🔴' if pred == 1 else '🟢'
        recent_conns.append({'service': service})
        if len(recent_conns) > 500:
            recent_conns.pop(0)

        print(f"{icon} [{packet_count:05d}] {label:6s} {conf:5.1f}%  "
              f"{src:15s} → {dst:15s}  proto={'TCP' if proto==2 else 'UDP'}")

print("\n" + "="*60)
print("  Live WiFi Traffic Classification")
print("="*60)
print("  Capturing on your WiFi interface...")
print("  Press Ctrl+C to stop\n")

try:
    sniff(filter="ip", prn=handle_packet, store=0, iface="en0")
except PermissionError:
    print("\nPermission denied — run with sudo:")
    print("  sudo python3 live_capture.py")
except KeyboardInterrupt:
    print(f"\n\nStopped. Processed {packet_count} packets.")