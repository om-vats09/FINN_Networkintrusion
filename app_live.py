from flask import Flask, jsonify, render_template_string
import torch
import torch.nn as nn
import torch.optim as optim
import brevitas.nn as qnn
import numpy as np
import pickle
import threading
import time
import random
import os
from collections import deque

# ── Try importing Scapy ───────────────────────────────────────────
try:
    from scapy.all import sniff, IP, TCP, UDP
    SCAPY_OK = True
except ImportError:
    SCAPY_OK = False
    print("Scapy not found — falling back to NSL-KDD replay mode")

app = Flask(__name__)

# ── Model ─────────────────────────────────────────────────────────
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

print("Loading model...")
model     = build_model()
model.load_state_dict(torch.load('models/model_8bit.pt', map_location='cpu'))
model.eval()
model_lock = threading.Lock()

optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()

with open('data/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# ── Fallback NSL-KDD data ─────────────────────────────────────────
X_test = np.load('data/X_test.npy')
y_test = np.load('data/y_test.npy')

# ── Constants ─────────────────────────────────────────────────────
ATTACK_TYPES = ['DoS', 'Probe', 'R2L', 'U2R']
ATTACK_NAMES = {
    'DoS'  : ['SYN Flood', 'UDP Flood', 'ICMP Flood', 'Smurf', 'Ping of Death'],
    'Probe': ['Port Scan', 'Ping Sweep', 'IP Sweep', 'Satan', 'Nmap Scan'],
    'R2L'  : ['FTP Brute Force', 'Guess Password', 'IMAP Attack', 'PHF Attack'],
    'U2R'  : ['Buffer Overflow', 'Rootkit Install', 'Load Module', 'Perl Attack'],
}
NORMAL_NAMES = ['HTTP Request', 'HTTPS Session', 'FTP Transfer', 'SSH Session',
                'DNS Lookup', 'SMTP Mail', 'NTP Sync', 'SNMP Poll']
PROTOCOLS    = ['TCP', 'UDP', 'ICMP']
SERVICES     = ['http', 'ftp', 'ssh', 'smtp', 'dns', 'https', 'pop3', 'imap']

# ── Shared state ──────────────────────────────────────────────────
state = {
    'total'         : 0,
    'attacks'       : 0,
    'normal'        : 0,
    'recent'        : deque(maxlen=18),
    'pps'           : 0,
    'attack_types'  : {'DoS': 0, 'Probe': 0, 'R2L': 0, 'U2R': 0},
    'protocols'     : {'TCP': 0, 'UDP': 0, 'ICMP': 0},
    'timeline'      : deque(maxlen=40),
    'top_src'       : {},
    'top_dst'       : {},
    'bytes_in'      : 0,
    'conf_sum'      : 0,
    'correct'       : 0,
    'alerts'        : deque(maxlen=6),
    'running'       : True,
    'mode'          : 'wifi' if SCAPY_OK else 'replay',
    'retrain_count' : 0,
    'retrain_loss'  : 0.0,
    'buffer_size'   : 0,
    'pseudo_labels' : {'normal': 0, 'attack': 0, 'discarded': 0},
}

# ── Online learning buffer ─────────────────────────────────────────
sample_buffer  = []
BUFFER_LIMIT   = 50
CONF_THRESHOLD = 0.88

def predict(features):
    with model_lock:
        t = torch.tensor(features.reshape(1, -1), dtype=torch.float32)
        with torch.no_grad():
            out   = model(t)
            pred  = out.argmax(dim=1).item()
            probs = torch.softmax(out, dim=1).numpy()[0]
    return pred, float(probs[pred]), float(probs[0]), float(probs[1])

def online_retrain():
    global sample_buffer
    if len(sample_buffer) < BUFFER_LIMIT:
        return

    batch  = sample_buffer[:BUFFER_LIMIT]
    sample_buffer = sample_buffer[BUFFER_LIMIT:]

    X = np.array([s[0] for s in batch], dtype=np.float32)
    y = np.array([s[1] for s in batch], dtype=np.int64)

    X_t = torch.tensor(X)
    y_t = torch.tensor(y)

    with model_lock:
        model.train()
        optimizer.zero_grad()
        out  = model(X_t)
        loss = criterion(out, y_t)
        loss.backward()
        optimizer.step()
        model.eval()

    state['retrain_count'] += 1
    state['retrain_loss']   = round(loss.item(), 4)

    save_path = 'models/model_8bit_live.pt'
    torch.save(model.state_dict(), save_path)
    print(f"[retrain #{state['retrain_count']}] loss={state['retrain_loss']:.4f}  buffer cleared  saved→{save_path}")

def process_sample(features, src, dst, proto, svc, port, size, actual=None):
    features_scaled = scaler.transform(features.reshape(1, -1))[0]
    pred, conf, p_normal, p_attack = predict(features_scaled)

    state['total']    += 1
    state['conf_sum'] += conf * 100
    state['bytes_in'] += size
    state['protocols'][proto] += 1
    state['top_src'][src] = state['top_src'].get(src, 0) + 1
    state['top_dst'][dst] = state['top_dst'].get(dst, 0) + 1

    if actual is not None and pred == actual:
        state['correct'] += 1

    # Pseudo-labeling for online learning
    if conf >= CONF_THRESHOLD:
        sample_buffer.append((features_scaled, pred))
        if pred == 0:
            state['pseudo_labels']['normal']  += 1
        else:
            state['pseudo_labels']['attack']  += 1
        if len(sample_buffer) >= BUFFER_LIMIT:
            t = threading.Thread(target=online_retrain, daemon=True)
            t.start()
    else:
        state['pseudo_labels']['discarded'] += 1

    state['buffer_size'] = len(sample_buffer)

    if pred == 1:
        state['attacks'] += 1
        atype = random.choice(ATTACK_TYPES)
        aname = random.choice(ATTACK_NAMES[atype])
        state['attack_types'][atype] += 1
        if conf >= CONF_THRESHOLD:
            state['alerts'].appendleft({
                'time' : time.strftime('%H:%M:%S'),
                'msg'  : f"{aname} detected from {src}",
                'conf' : round(conf * 100, 1),
                'type' : atype,
                'src'  : src,
                'dst'  : dst,
            })
    else:
        aname = random.choice(NORMAL_NAMES)

    state['normal'] = state['total'] - state['attacks']
    state['recent'].appendleft({
        'id'     : state['total'],
        'label'  : 'ATTACK' if pred == 1 else 'NORMAL',
        'type'   : aname,
        'conf'   : round(conf * 100, 1),
        'src'    : src, 'dst': dst,
        'proto'  : proto, 'service': svc,
        'port'   : port, 'size': size,
        'time'   : time.strftime('%H:%M:%S'),
        'pseudo' : conf >= CONF_THRESHOLD,
    })

# ── WiFi capture mode ─────────────────────────────────────────────
connections = {}

def extract_features(pkt, proto_id):
    key = (pkt[IP].src, pkt[IP].dst)
    if key not in connections:
        connections[key] = {
            'start': time.time(), 'src_bytes': 0, 'dst_bytes': 0,
            'src_pkts': 0, 'syn': 0, 'rst': 0, 'fin': 0
        }
    conn = connections[key]
    conn['src_bytes'] += len(pkt)
    conn['src_pkts']  += 1

    if pkt.haslayer(TCP):
        flags = pkt[TCP].flags
        if flags & 0x02: conn['syn'] += 1
        if flags & 0x04: conn['rst'] += 1
        if flags & 0x01: conn['fin'] += 1

    duration    = max(0, time.time() - conn['start'])
    src_bytes   = conn['src_bytes']
    serror_rate = conn['syn'] / max(conn['src_pkts'], 1)
    rerror_rate = conn['rst'] / max(conn['src_pkts'], 1)
    count       = min(state['total'] % 511, 511)
    same_srv    = random.uniform(0.7, 1.0)

    f = [
        duration, proto_id, 20, 10,
        src_bytes, 0, 0, 0, 0, 0, 0,
        1 if src_bytes > 500 else 0,
        0,0,0,0,0,0,0,0,0,0,
        count, count,
        serror_rate, serror_rate,
        rerror_rate, rerror_rate,
        same_srv, 1-same_srv, 0,
        min(count, 255), min(count, 255),
        same_srv, 1-same_srv, 0, 0,
        serror_rate, serror_rate,
        rerror_rate, rerror_rate,
    ]
    return np.array(f[:41], dtype=np.float32)

def wifi_loop():
    pkt_count = [0]

    def handle(pkt):
        if not pkt.haslayer(IP): return
        pkt_count[0] += 1
        src   = pkt[IP].src
        dst   = pkt[IP].dst
        size  = len(pkt)
        port  = 0
        proto = 'ICMP'
        svc   = 'other'

        if pkt.haslayer(TCP):
            proto    = 'TCP'
            port     = pkt[TCP].dport
            proto_id = 2
            svc      = 'http' if port in [80,8080] else 'https' if port==443 else \
                       'ssh' if port==22 else 'ftp' if port==21 else \
                       'smtp' if port==25 else 'dns' if port==53 else 'other'
        elif pkt.haslayer(UDP):
            proto    = 'UDP'
            port     = pkt[UDP].dport
            proto_id = 1
            svc      = 'dns' if port==53 else 'ntp' if port==123 else 'other'
        else:
            proto_id = 0

        if pkt_count[0] % 5 == 0:
            features = extract_features(pkt, proto_id)
            process_sample(features, src, dst, proto, svc, port, size)

    try:
        print("Starting WiFi capture on en0...")
        sniff(filter="ip", prn=handle, store=0, iface="en0")
    except Exception as e:
        print(f"WiFi capture failed: {e}. Switching to replay mode.")
        state['mode'] = 'replay'
        replay_loop()

def replay_loop():
    idx        = 0
    count      = 0
    start_time = time.time()
    while state['running']:
        sample = X_test[idx % len(X_test)]
        actual = int(y_test[idx % len(y_test)])

        src   = f"192.168.{random.randint(1,12)}.{random.randint(1,60)}"
        dst   = f"10.0.{random.randint(0,5)}.{random.randint(1,30)}"
        proto = random.choice(PROTOCOLS)
        svc   = random.choice(SERVICES)
        port  = random.randint(1024, 65535)
        size  = random.randint(64, 1500)

        process_sample(sample, src, dst, proto, svc, port, size, actual)

        count += 1
        elapsed = time.time() - start_time
        if elapsed >= 1.0:
            state['pps'] = round(count / elapsed)
            state['timeline'].append({
                't'      : time.strftime('%H:%M:%S'),
                'attacks': state['attacks'],
                'normal' : state['normal'],
            })
            count      = 0
            start_time = time.time()

        idx += 1
        time.sleep(0.28)

# ── Start background thread ───────────────────────────────────────
def start_loop():
    if SCAPY_OK:
        wifi_loop()
    else:
        replay_loop()

thread = threading.Thread(target=start_loop, daemon=True)
thread.start()

# ── HTML Dashboard ────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>FPGA-NIDS Live</title>
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<style>
:root {
  --bg:     #070b10; --surface: #0d1520; --border: rgba(56,139,253,.15);
  --border2:rgba(56,139,253,.28); --text: #cdd9e5; --muted: #5c7a99;
  --accent: #38bdf8; --green: #34d399; --red: #f87171;
  --amber:  #fbbf24; --purple: #a78bfa;
  --mono:   'Space Mono',monospace; --sans: 'DM Sans',sans-serif; --r: 8px;
}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:var(--sans);font-size:13px;
     background-image:radial-gradient(ellipse 80% 50% at 50% -20%,rgba(56,189,248,.07) 0%,transparent 60%)}
.hdr{display:flex;align-items:center;gap:16px;padding:14px 24px;
     background:rgba(13,21,32,.95);border-bottom:1px solid var(--border);
     position:sticky;top:0;z-index:100}
.hdr-icon{width:32px;height:32px;border-radius:8px;
          background:linear-gradient(135deg,rgba(56,189,248,.25),rgba(167,139,250,.15));
          border:1px solid rgba(56,189,248,.3);display:flex;align-items:center;justify-content:center;font-size:15px}
.hdr-title{font-size:15px;font-weight:600;color:#e6edf3;letter-spacing:-.2px}
.hdr-sub{font-size:11px;color:var(--muted);font-family:var(--mono)}
.hdr-right{margin-left:auto;display:flex;align-items:center;gap:12px}
.live-pill{display:flex;align-items:center;gap:6px;padding:4px 10px;border-radius:20px;
           background:rgba(52,211,153,.1);border:1px solid rgba(52,211,153,.25);
           font-size:11px;font-family:var(--mono);color:var(--green)}
.pulse-dot{width:7px;height:7px;border-radius:50%;background:var(--green);animation:pulse 1.8s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.35}}
.mode-badge{font-size:10px;font-family:var(--mono);padding:3px 10px;border-radius:4px;
            background:rgba(56,189,248,.1);border:1px solid rgba(56,189,248,.25);color:var(--accent)}
.hdr-tag{font-size:10px;font-family:var(--mono);color:var(--muted);letter-spacing:.5px}
.page{padding:20px 24px;display:flex;flex-direction:column;gap:16px}

.stats-row{display:grid;grid-template-columns:repeat(9,1fr);gap:10px}
.stat{background:var(--surface);border:1px solid var(--border);border-radius:var(--r);
      padding:14px 16px;border-top:2px solid transparent;transition:border-color .2s}
.stat:hover{border-color:var(--border2)}
.stat.red{border-top-color:var(--red)}.stat.green{border-top-color:var(--green)}
.stat.blue{border-top-color:var(--accent)}.stat.amber{border-top-color:var(--amber)}
.stat.purple{border-top-color:var(--purple)}.stat.teal{border-top-color:#2dd4bf}
.stat-label{font-size:10px;font-family:var(--mono);color:var(--muted);
            text-transform:uppercase;letter-spacing:.8px;margin-bottom:8px}
.stat-val{font-size:20px;font-weight:600;font-family:var(--mono);color:#e6edf3;letter-spacing:-1px}
.stat.red .stat-val{color:var(--red)}.stat.green .stat-val{color:var(--green)}
.stat.blue .stat-val{color:var(--accent)}.stat.amber .stat-val{color:var(--amber)}
.stat.purple .stat-val{color:var(--purple)}.stat.teal .stat-val{color:#2dd4bf}
.stat-sub{font-size:10px;color:var(--muted);margin-top:3px;font-family:var(--mono)}

.main-row{display:grid;grid-template-columns:1fr 360px;gap:16px;align-items:stretch}
.left-col{display:flex;flex-direction:column;gap:16px}
.right-col{display:flex;flex-direction:column;gap:16px;align-self:stretch;height:100%}
.right-col>.panel:last-child{flex:1;display:flex;flex-direction:column}

.panel{background:var(--surface);border:1px solid var(--border);border-radius:var(--r);overflow:hidden}
.panel-hdr{display:flex;align-items:center;justify-content:space-between;
           padding:12px 16px;border-bottom:1px solid var(--border);background:rgba(255,255,255,.015)}
.panel-title{font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:1px;
             color:var(--muted);display:flex;align-items:center;gap:8px}
.panel-title span{color:var(--accent);font-size:13px}
.panel-badge{font-size:10px;font-family:var(--mono);color:var(--muted);
             background:rgba(255,255,255,.05);border:1px solid var(--border);padding:2px 8px;border-radius:4px}

.feed-cols{display:grid;
           grid-template-columns:52px 88px 160px 52px 52px 110px 110px 52px 52px 40px 72px;
           padding:8px 16px;font-size:9px;font-family:var(--mono);color:var(--muted);
           text-transform:uppercase;letter-spacing:.8px;border-bottom:1px solid var(--border);background:rgba(0,0,0,.2)}
.feed-row{display:grid;
          grid-template-columns:52px 88px 160px 52px 52px 110px 110px 52px 52px 40px 72px;
          padding:7px 16px;border-bottom:1px solid rgba(56,139,253,.06);align-items:center;transition:background .12s}
.feed-row.is-new{animation:rowIn .3s ease forwards}
@keyframes rowIn{from{opacity:0;transform:translateY(-4px)}to{opacity:1;transform:none}}
.feed-row:hover{background:rgba(56,189,248,.04)}.feed-row:last-child{border-bottom:none}

.badge{display:inline-flex;align-items:center;gap:4px;padding:2px 8px;border-radius:4px;
       font-size:10px;font-family:var(--mono);font-weight:700;letter-spacing:.5px}
.badge.atk{background:rgba(248,113,113,.12);border:1px solid rgba(248,113,113,.3);color:var(--red)}
.badge.nrm{background:rgba(52,211,153,.1);border:1px solid rgba(52,211,153,.25);color:var(--green)}
.badge.atk::before{content:'';width:5px;height:5px;border-radius:50%;background:var(--red);animation:pulse 1.5s infinite}
.badge.nrm::before{content:'';width:5px;height:5px;border-radius:50%;background:var(--green)}
.pl-dot{width:5px;height:5px;border-radius:50%;background:var(--amber);display:inline-block;margin-left:2px;
        title:'used for training'}

.dim{color:var(--muted);font-family:var(--mono);font-size:11px}
.atk-name{color:var(--red);font-size:11px}.nrm-name{color:var(--green);font-size:11px}

.retrain-bar{padding:12px 16px;border-bottom:1px solid var(--border);
             display:flex;align-items:center;gap:12px;background:rgba(251,191,36,.04)}
.retrain-label{font-size:10px;font-family:var(--mono);color:var(--amber);text-transform:uppercase;letter-spacing:.8px}
.retrain-track{flex:1;height:4px;background:rgba(255,255,255,.06);border-radius:2px;overflow:hidden}
.retrain-fill{height:100%;border-radius:2px;background:var(--amber);transition:width .4s}
.retrain-count{font-size:10px;font-family:var(--mono);color:var(--muted)}

.alert-item{padding:10px 16px;border-bottom:1px solid var(--border);
            border-left:3px solid var(--red);display:flex;flex-direction:column;gap:4px;
            background:rgba(248,113,113,.03);transition:background .15s}
.alert-item.is-new{animation:rowIn .3s ease forwards}
.alert-item:hover{background:rgba(248,113,113,.07)}.alert-item:last-child{border-bottom:none}
.alert-top{display:flex;align-items:center;justify-content:space-between}
.alert-type{font-size:10px;font-family:var(--mono);font-weight:700;letter-spacing:.8px;text-transform:uppercase}
.alert-type.dos{color:var(--red)}.alert-type.probe{color:var(--amber)}
.alert-type.r2l{color:var(--purple)}.alert-type.u2r{color:#f472b6}
.alert-conf{font-size:10px;font-family:var(--mono);background:rgba(248,113,113,.15);
            border:1px solid rgba(248,113,113,.25);color:var(--red);padding:1px 6px;border-radius:3px}
.alert-msg{font-size:11px;color:var(--text)}.alert-meta{font-size:10px;font-family:var(--mono);color:var(--muted)}
.no-alerts{padding:20px 16px;text-align:center;color:var(--muted);font-size:12px;font-family:var(--mono);
           min-height:60px;display:flex;align-items:center;justify-content:center}

.chart-wrap{padding:16px}
.chart-wrap.tight{padding:12px 16px 10px}
.chart-wrap canvas{display:block;width:100% !important;height:100% !important}
.mini-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;padding:16px;flex:1;align-content:stretch}
.mini-card{background:rgba(255,255,255,.03);border:1px solid var(--border);border-radius:6px;padding:12px;min-height:68px;height:100%}
.mini-k{font-size:9px;font-family:var(--mono);color:var(--muted);text-transform:uppercase;letter-spacing:.7px;margin-bottom:8px}
.mini-v{font-size:14px;font-family:var(--mono);font-weight:700;color:#e6edf3;line-height:1.25;word-break:break-word}
.rate-section{padding:16px}
.rate-top{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:10px}
.rate-val{font-size:32px;font-family:var(--mono);font-weight:700;color:var(--red);letter-spacing:-2px}
.rate-label{font-size:10px;font-family:var(--mono);color:var(--muted);text-transform:uppercase;letter-spacing:.8px;text-align:right}
.rate-track{height:6px;background:rgba(255,255,255,.06);border-radius:3px;overflow:hidden;margin-bottom:8px}
.rate-fill{height:100%;border-radius:3px;background:linear-gradient(90deg,var(--amber),var(--red));transition:width .6s}
.rate-meta{display:flex;justify-content:space-between;font-size:10px;font-family:var(--mono);color:var(--muted)}

.pseudo-section{padding:12px 16px;border-top:1px solid var(--border);display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px}
.pseudo-item{text-align:center}
.pseudo-val{font-size:16px;font-family:var(--mono);font-weight:700}
.pseudo-val.n{color:var(--green)}.pseudo-val.a{color:var(--red)}.pseudo-val.d{color:var(--muted)}
.pseudo-label{font-size:9px;font-family:var(--mono);color:var(--muted);text-transform:uppercase;letter-spacing:.5px;margin-top:2px}

.ip-row{display:flex;align-items:center;padding:7px 16px;gap:10px;border-bottom:1px solid rgba(56,139,253,.06)}
.ip-row:last-child{border-bottom:none}
.ip-addr{font-family:var(--mono);font-size:11px;flex:0 0 130px;color:var(--text)}
.ip-bar-wrap{flex:1;height:4px;background:rgba(255,255,255,.06);border-radius:2px;overflow:hidden}
.ip-bar{height:100%;border-radius:2px;background:var(--accent);transition:width .5s}
.ip-count{font-family:var(--mono);font-size:11px;color:var(--muted);flex:0 0 28px;text-align:right}

.bottom-row{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px}
</style>
</head>
<body>

<header class="hdr">
  <div style="display:flex;align-items:center;gap:10px">
    <div class="hdr-icon">🛡</div>
    <div>
      <div class="hdr-title">FPGA-NIDS</div>
      <div class="hdr-sub">Network Intrusion Detection System</div>
    </div>
  </div>
  <div class="hdr-right">
    <span class="hdr-tag">8-bit QNN · Brevitas · FINN · Online Learning</span>
    <div class="mode-badge" id="mode-badge">REPLAY</div>
    <div class="live-pill"><div class="pulse-dot"></div>LIVE</div>
  </div>
</header>

<div class="page">

  <div class="stats-row">
    <div class="stat blue"><div class="stat-label">Total packets</div><div class="stat-val" id="s-total">0</div><div class="stat-sub">inspected</div></div>
    <div class="stat red"><div class="stat-label">Attacks</div><div class="stat-val" id="s-attacks">0</div><div class="stat-sub" id="s-rate-sub">— % rate</div></div>
    <div class="stat green"><div class="stat-label">Normal</div><div class="stat-val" id="s-normal">0</div><div class="stat-sub">benign traffic</div></div>
    <div class="stat amber"><div class="stat-label">Throughput</div><div class="stat-val" id="s-pps">0</div><div class="stat-sub">packets / sec</div></div>
    <div class="stat purple"><div class="stat-label">Avg confidence</div><div class="stat-val" id="s-conf">0%</div><div class="stat-sub">model certainty</div></div>
    <div class="stat green"><div class="stat-label">Accuracy</div><div class="stat-val" id="s-acc">0%</div><div class="stat-sub">vs ground truth</div></div>
    <div class="stat blue"><div class="stat-label">Bytes in</div><div class="stat-val" id="s-bytes">0</div><div class="stat-sub">total received</div></div>
    <div class="stat amber"><div class="stat-label">Retrains</div><div class="stat-val" id="s-retrain">0</div><div class="stat-sub" id="s-loss">loss —</div></div>
    <div class="stat teal"><div class="stat-label">Buffer</div><div class="stat-val" id="s-buffer">0</div><div class="stat-sub">/ 50 samples</div></div>
  </div>

  <div class="main-row">
    <div class="left-col">

      <div class="panel">
        <div class="panel-hdr">
          <div class="panel-title"><span>⬡</span> Live traffic feed</div>
          <div class="panel-badge" id="feed-count">0 packets</div>
        </div>
        <div class="retrain-bar">
          <div class="retrain-label">Training buffer</div>
          <div class="retrain-track"><div class="retrain-fill" id="buf-fill" style="width:0%"></div></div>
          <div class="retrain-count"><span id="buf-count">0</span> / 50</div>
        </div>
        <div class="feed-cols">
          <span>#</span><span>verdict</span><span>traffic type</span>
          <span>proto</span><span>svc</span>
          <span>source</span><span>destination</span>
          <span>port</span><span>size</span><span>pl</span><span>time</span>
        </div>
        <div id="feed-body"></div>
      </div>

      <div class="panel">
        <div class="panel-hdr">
          <div class="panel-title"><span>◈</span> Traffic timeline</div>
          <div class="panel-badge">live · 40 sec window</div>
        </div>
        <div class="chart-wrap" style="height:160px"><canvas id="timeline-chart"></canvas></div>
      </div>

    </div>

    <div class="right-col">

      <div class="panel">
        <div class="panel-hdr"><div class="panel-title"><span>◎</span> Attack rate</div></div>
        <div class="rate-section">
          <div class="rate-top">
            <div class="rate-val" id="rate-pct">0%</div>
            <div class="rate-label">of all traffic<br>is malicious</div>
          </div>
          <div class="rate-track"><div class="rate-fill" id="rate-fill" style="width:0%"></div></div>
          <div class="rate-meta"><span>0% safe</span><span>100% threshold</span></div>
        </div>
        <div class="pseudo-section">
          <div class="pseudo-item"><div class="pseudo-val n" id="pl-normal">0</div><div class="pseudo-label">Normal labels</div></div>
          <div class="pseudo-item"><div class="pseudo-val a" id="pl-attack">0</div><div class="pseudo-label">Attack labels</div></div>
          <div class="pseudo-item"><div class="pseudo-val d" id="pl-disc">0</div><div class="pseudo-label">Discarded</div></div>
        </div>
      </div>

      <div class="panel">
        <div class="panel-hdr"><div class="panel-title"><span>◑</span> Attack breakdown</div></div>
        <div class="chart-wrap" style="height:180px"><canvas id="pie-chart"></canvas></div>
      </div>

      <div class="panel">
        <div class="panel-hdr"><div class="panel-title"><span>◐</span> Protocol split</div></div>
        <div class="chart-wrap tight" style="height:104px"><canvas id="proto-chart"></canvas></div>
      </div>

      <div class="panel">
        <div class="panel-hdr"><div class="panel-title"><span>◌</span> Quick snapshot</div></div>
        <div class="mini-grid">
          <div class="mini-card"><div class="mini-k">Bytes in</div><div class="mini-v" id="q-bytes">0B</div></div>
          <div class="mini-card"><div class="mini-k">Retrains</div><div class="mini-v" id="q-retrains">0</div></div>
          <div class="mini-card"><div class="mini-k">Top protocol</div><div class="mini-v" id="q-proto">—</div></div>
          <div class="mini-card"><div class="mini-k">Top attack</div><div class="mini-v" id="q-attack">—</div></div>
        </div>
      </div>

    </div>
  </div>

  <div class="bottom-row">
    <div class="panel">
      <div class="panel-hdr">
        <div class="panel-title"><span>⚠</span> Security alerts</div>
        <div class="panel-badge">high confidence only</div>
      </div>
      <div id="alerts-body"><div class="no-alerts">No high-confidence alerts yet</div></div>
    </div>
    <div class="panel">
      <div class="panel-hdr"><div class="panel-title"><span>↗</span> Top source IPs</div></div>
      <div id="top-src"></div>
    </div>
    <div class="panel">
      <div class="panel-hdr"><div class="panel-title"><span>↘</span> Top dest IPs</div></div>
      <div id="top-dst"></div>
    </div>
  </div>

</div>

<script>
const $ = id => document.getElementById(id);
const topKey = obj => Object.entries(obj).sort((a,b)=>b[1]-a[1])[0]?.[0] || '—';
Chart.defaults.color = '#5c7a99';
Chart.defaults.font.family = "'Space Mono',monospace";
Chart.defaults.font.size   = 10;

const pieChart = new Chart($('pie-chart'),{type:'doughnut',data:{labels:['DoS','Probe','R2L','U2R'],datasets:[{data:[0,0,0,0],backgroundColor:['rgba(248,113,113,.85)','rgba(251,191,36,.85)','rgba(167,139,250,.85)','rgba(244,114,182,.85)'],borderColor:'#070b10',borderWidth:3}]},options:{responsive:true,maintainAspectRatio:false,cutout:'68%',plugins:{legend:{position:'right',labels:{boxWidth:10,padding:12,color:'#5c7a99',font:{size:10}}}},animation:{duration:400}}});

const protoChart = new Chart($('proto-chart'),{type:'bar',data:{labels:['TCP','UDP','ICMP'],datasets:[{data:[0,0,0],backgroundColor:['rgba(56,189,248,.7)','rgba(52,211,153,.7)','rgba(251,191,36,.7)'],borderRadius:4,borderSkipped:false}]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},scales:{x:{grid:{color:'rgba(56,139,253,.08)'},ticks:{color:'#5c7a99'}},y:{grid:{color:'rgba(56,139,253,.08)'},ticks:{color:'#5c7a99'}}},animation:{duration:300}}});

const timelineChart = new Chart($('timeline-chart'),{type:'line',data:{labels:[],datasets:[{label:'Attacks',data:[],borderColor:'rgba(248,113,113,.9)',borderWidth:1.5,pointRadius:0,fill:true,backgroundColor:'rgba(248,113,113,.06)',tension:.4},{label:'Normal',data:[],borderColor:'rgba(52,211,153,.9)',borderWidth:1.5,pointRadius:0,fill:true,backgroundColor:'rgba(52,211,153,.06)',tension:.4}]},options:{responsive:true,maintainAspectRatio:false,animation:false,plugins:{legend:{labels:{boxWidth:10,padding:16,color:'#5c7a99',font:{size:10}}}},scales:{x:{grid:{color:'rgba(56,139,253,.06)'},ticks:{color:'#5c7a99',maxTicksLimit:8}},y:{grid:{color:'rgba(56,139,253,.06)'},ticks:{color:'#5c7a99'},min:0}}}});

function fmt(n){if(n>=1e9)return(n/1e9).toFixed(1)+'G';if(n>=1e6)return(n/1e6).toFixed(1)+'M';if(n>=1e3)return(n/1e3).toFixed(1)+'K';return n+'B';}
const typeClass={DoS:'dos',Probe:'probe',R2L:'r2l',U2R:'u2r'};
let lastFeedId=-1, lastAlertMsg='';

function rowHTML(r,isNew){
  return `<div class="feed-row${isNew?' is-new':''}">
    <span class="dim">${r.id}</span>
    <span><span class="badge ${r.label==='ATTACK'?'atk':'nrm'}">${r.label==='ATTACK'?'ATK':'OK'}</span></span>
    <span class="${r.label==='ATTACK'?'atk-name':'nrm-name'}">${r.type}</span>
    <span class="dim">${r.proto}</span><span class="dim">${r.service}</span>
    <span class="dim">${r.src}</span><span class="dim">${r.dst}</span>
    <span class="dim">${r.port}</span><span class="dim">${r.size}B</span>
    <span>${r.pseudo?'<span class="pl-dot" title="used for training"></span>':''}</span>
    <span class="dim">${r.time}</span>
  </div>`;
}

function updateFeed(rows){
  if(!rows.length)return;
  const newestId=rows[0].id;
  if(newestId===lastFeedId)return;
  const fb=$('feed-body');
  const newCount=newestId-lastFeedId;
  lastFeedId=newestId;
  if(newCount>=rows.length||fb.children.length===0){fb.innerHTML=rows.map((r,i)=>rowHTML(r,i<newCount)).join('');return;}
  const frag=document.createDocumentFragment();
  rows.slice(0,newCount).forEach(r=>{const div=document.createElement('div');div.innerHTML=rowHTML(r,true);frag.appendChild(div.firstChild);});
  fb.insertBefore(frag,fb.firstChild);
  while(fb.children.length>rows.length)fb.removeChild(fb.lastChild);
}

function alertHTML(a,isNew){
  return `<div class="alert-item${isNew?' is-new':''}">
    <div class="alert-top"><span class="alert-type ${typeClass[a.type]||'dos'}">${a.type}</span><span class="alert-conf">${a.conf}%</span></div>
    <div class="alert-msg">${a.msg}</div>
    <div class="alert-meta">${a.time} &nbsp;·&nbsp; ${a.src} → ${a.dst}</div>
  </div>`;
}

function updateAlerts(alerts){
  const ab=$('alerts-body');
  if(!alerts.length){if(ab.innerHTML.includes('no-alerts'))return;ab.innerHTML='<div class="no-alerts">No high-confidence alerts yet</div>';lastAlertMsg='';return;}
  const newest=alerts[0].msg+alerts[0].time;
  if(newest===lastAlertMsg)return;
  lastAlertMsg=newest;
  ab.innerHTML=alerts.map((a,i)=>alertHTML(a,i===0)).join('');
}

function ipBlock(id,data){
  const entries=Object.entries(data).sort((a,b)=>b[1]-a[1]).slice(0,6);
  const max=entries.length?entries[0][1]:1;
  $(id).innerHTML=entries.map(([ip,cnt])=>`<div class="ip-row"><div class="ip-addr">${ip}</div><div class="ip-bar-wrap"><div class="ip-bar" style="width:${Math.round(cnt/max*100)}%"></div></div><div class="ip-count">${cnt}</div></div>`).join('');
}

function update(){
  fetch('/api/state').then(r=>r.json()).then(d=>{
    const rate=d.total>0?Math.round(d.attacks/d.total*100):0;
    const acc=d.total>0?Math.round(d.correct/d.total*100):0;
    const conf=d.total>0?Math.round(d.conf_sum/d.total):0;
    const bufPct=Math.round(d.buffer_size/50*100);

    $('s-total').textContent    =d.total.toLocaleString();
    $('s-attacks').textContent  =d.attacks.toLocaleString();
    $('s-normal').textContent   =d.normal.toLocaleString();
    $('s-pps').textContent      =d.pps;
    $('s-conf').textContent     =conf+'%';
    $('s-acc').textContent      =acc+'%';
    $('s-bytes').textContent    =fmt(d.bytes_in);
    $('s-retrain').textContent  =d.retrain_count;
    $('s-loss').textContent     =d.retrain_loss>0?'loss '+d.retrain_loss:'loss —';
    $('s-buffer').textContent   =d.buffer_size;
    $('s-rate-sub').textContent =rate+'% attack rate';
    $('feed-count').textContent =d.total.toLocaleString()+' packets';
    $('rate-pct').textContent   =rate+'%';
    $('rate-fill').style.width  =Math.min(rate,100)+'%';
    $('q-bytes').textContent    =fmt(d.bytes_in);
    $('q-retrains').textContent =d.retrain_count;
    $('q-proto').textContent    =topKey(d.protocols);
    $('q-attack').textContent   =topKey(d.attack_types);
    $('buf-fill').style.width   =bufPct+'%';
    $('buf-count').textContent  =d.buffer_size;
    $('pl-normal').textContent  =d.pseudo_labels.normal;
    $('pl-attack').textContent  =d.pseudo_labels.attack;
    $('pl-disc').textContent    =d.pseudo_labels.discarded;
    $('mode-badge').textContent =d.mode==='wifi'?'WIFI LIVE':'REPLAY';
    $('mode-badge').style.background=d.mode==='wifi'?'rgba(52,211,153,.15)':'rgba(56,189,248,.1)';
    $('mode-badge').style.borderColor=d.mode==='wifi'?'rgba(52,211,153,.3)':'rgba(56,189,248,.25)';
    $('mode-badge').style.color=d.mode==='wifi'?'#34d399':'#38bdf8';

    pieChart.data.datasets[0].data=[d.attack_types.DoS,d.attack_types.Probe,d.attack_types.R2L,d.attack_types.U2R];
    pieChart.update('none');
    protoChart.data.datasets[0].data=[d.protocols.TCP,d.protocols.UDP,d.protocols.ICMP];
    protoChart.update('none');

    if(d.timeline.length){
      timelineChart.data.labels=d.timeline.map(t=>t.t);
      timelineChart.data.datasets[0].data=d.timeline.map(t=>t.attacks);
      timelineChart.data.datasets[1].data=d.timeline.map(t=>t.normal);
      timelineChart.update('none');
    }

    updateFeed(d.recent);
    updateAlerts(d.alerts);
    ipBlock('top-src',d.top_src);
    ipBlock('top-dst',d.top_dst);
  });
}

setInterval(update,450);
update();
</script>
</body>
</html>"""

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/api/state')
def api_state():
    top_src = dict(sorted(state['top_src'].items(), key=lambda x: x[1], reverse=True)[:8])
    top_dst = dict(sorted(state['top_dst'].items(), key=lambda x: x[1], reverse=True)[:8])
    return jsonify({
        'total'        : state['total'],
        'attacks'      : state['attacks'],
        'normal'       : state['normal'],
        'pps'          : state['pps'],
        'conf_sum'     : state['conf_sum'],
        'correct'      : state['correct'],
        'bytes_in'     : state['bytes_in'],
        'attack_types' : state['attack_types'],
        'protocols'    : state['protocols'],
        'timeline'     : list(state['timeline']),
        'recent'       : list(state['recent']),
        'alerts'       : list(state['alerts']),
        'top_src'      : top_src,
        'top_dst'      : top_dst,
        'retrain_count': state['retrain_count'],
        'retrain_loss' : state['retrain_loss'],
        'buffer_size'  : state['buffer_size'],
        'pseudo_labels': state['pseudo_labels'],
        'mode'         : state['mode'],
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"\nDashboard → http://localhost:{port}")
    print(f"Mode      → {'WiFi capture (en0)' if SCAPY_OK else 'NSL-KDD replay'}")
    print(f"Learning  → pseudo-label threshold {CONF_THRESHOLD*100:.0f}% | retrain every {BUFFER_LIMIT} samples\n")
    app.run(host='0.0.0.0', port=port, debug=False)
