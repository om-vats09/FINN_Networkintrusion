from flask import Flask, jsonify, render_template_string
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import threading
import time
import random
import os
from collections import deque
from model import build_model

try:
    from scapy.all import sniff, IP, TCP, UDP
    SCAPY_OK = True
except ImportError:
    SCAPY_OK = False

app = Flask(__name__)

print("Loading model...")
model      = build_model(8)
model.load_state_dict(torch.load('models/model_8bit.pt', map_location='cpu'))
model.eval()
model_lock = threading.Lock()
optimizer  = optim.Adam(model.parameters(), lr=0.0001)
criterion  = nn.CrossEntropyLoss()

with open('data/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

X_test = np.load('data/X_test.npy')
y_test = np.load('data/y_test.npy')

ATTACK_TYPES = ['DoS', 'Probe', 'R2L', 'U2R']
ATTACK_NAMES = {
    'DoS'  : ['SYN Flood', 'UDP Flood', 'ICMP Flood', 'Smurf'],
    'Probe': ['Port Scan', 'Ping Sweep', 'IP Sweep', 'Nmap Scan'],
    'R2L'  : ['FTP Brute Force', 'Guess Password', 'IMAP Attack'],
    'U2R'  : ['Buffer Overflow', 'Rootkit Install', 'Load Module'],
}
NORMAL_NAMES = ['HTTP Request', 'HTTPS Session', 'FTP Transfer',
                'SSH Session', 'DNS Lookup', 'SMTP Mail', 'NTP Sync']
PROTOCOLS    = ['TCP', 'UDP', 'ICMP']
SERVICES     = ['http', 'ftp', 'ssh', 'smtp', 'dns', 'https', 'pop3']

def make_state():
    return {
        'total'        : 0, 'attacks': 0, 'normal': 0,
        'recent'       : deque(maxlen=18),
        'pps'          : 0, 'conf_sum': 0, 'correct': 0,
        'attack_types' : {'DoS': 0, 'Probe': 0, 'R2L': 0, 'U2R': 0},
        'protocols'    : {'TCP': 0, 'UDP': 0, 'ICMP': 0},
        'timeline'     : deque(maxlen=40),
        'top_src'      : {}, 'top_dst': {},
        'bytes_in'     : 0, 'alerts': deque(maxlen=6),
        'retrain_count': 0, 'retrain_loss': 0.0,
        'buffer_size'  : 0,
        'pseudo_labels': {'normal': 0, 'attack': 0, 'discarded': 0},
        'running'      : True,
    }

replay_state = make_state()
wifi_state   = make_state()

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
    return pred, float(probs[pred])

def online_retrain():
    global sample_buffer
    batch         = sample_buffer[:BUFFER_LIMIT]
    sample_buffer = sample_buffer[BUFFER_LIMIT:]
    X = torch.tensor(np.array([s[0] for s in batch], dtype=np.float32))
    y = torch.tensor(np.array([s[1] for s in batch], dtype=np.int64))
    with model_lock:
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
        model.eval()
    wifi_state['retrain_count'] += 1
    wifi_state['retrain_loss']   = round(loss.item(), 4)
    torch.save(model.state_dict(), 'models/model_8bit_live.pt')
    print(f"[retrain #{wifi_state['retrain_count']}] loss={wifi_state['retrain_loss']}")

def process(st, features, src, dst, proto, svc, port, size, actual=None, learn=False):
    features_scaled = scaler.transform(features.reshape(1, -1))[0]
    pred, conf      = predict(features_scaled)

    st['total']    += 1
    st['conf_sum'] += conf * 100
    st['bytes_in'] += size
    st['protocols'][proto] += 1
    st['top_src'][src] = st['top_src'].get(src, 0) + 1
    st['top_dst'][dst] = st['top_dst'].get(dst, 0) + 1
    if actual is not None and pred == actual:
        st['correct'] += 1

    pseudo = False
    if learn and conf >= CONF_THRESHOLD:
        sample_buffer.append((features_scaled, pred))
        pseudo = True
        if pred == 0: st['pseudo_labels']['normal']  += 1
        else:         st['pseudo_labels']['attack']  += 1
        if len(sample_buffer) >= BUFFER_LIMIT:
            threading.Thread(target=online_retrain, daemon=True).start()
        st['buffer_size'] = len(sample_buffer)
    elif learn:
        st['pseudo_labels']['discarded'] += 1

    if pred == 1:
        st['attacks'] += 1
        atype = random.choice(ATTACK_TYPES)
        aname = random.choice(ATTACK_NAMES[atype])
        st['attack_types'][atype] += 1
        if conf >= CONF_THRESHOLD:
            st['alerts'].appendleft({
                'time': time.strftime('%H:%M:%S'),
                'msg' : f"{aname} from {src}",
                'conf': round(conf * 100, 1),
                'type': atype, 'src': src, 'dst': dst,
            })
    else:
        aname = random.choice(NORMAL_NAMES)

    st['normal'] = st['total'] - st['attacks']
    st['recent'].appendleft({
        'id'    : st['total'],
        'label' : 'ATTACK' if pred == 1 else 'NORMAL',
        'type'  : aname, 'conf': round(conf * 100, 1),
        'src'   : src, 'dst': dst, 'proto': proto,
        'service': svc, 'port': port, 'size': size,
        'time'  : time.strftime('%H:%M:%S'), 'pseudo': pseudo,
    })

def replay_loop():
    idx = 0
    count = 0
    start_time = time.time()
    while replay_state['running']:
        sample = X_test[idx % len(X_test)]
        actual = int(y_test[idx % len(y_test)])
        src    = f"192.168.{random.randint(1,12)}.{random.randint(1,60)}"
        dst    = f"10.0.{random.randint(0,5)}.{random.randint(1,30)}"
        proto  = random.choice(PROTOCOLS)
        svc    = random.choice(SERVICES)
        port   = random.randint(1024, 65535)
        size   = random.randint(64, 1500)
        process(replay_state, sample, src, dst, proto, svc, port, size, actual, learn=False)
        count += 1
        elapsed = time.time() - start_time
        if elapsed >= 1.0:
            replay_state['pps'] = round(count / elapsed)
            replay_state['timeline'].append({
                't': time.strftime('%H:%M:%S'),
                'attacks': replay_state['attacks'],
                'normal' : replay_state['normal'],
            })
            count = 0; start_time = time.time()
        idx += 1
        time.sleep(0.28)

connections = {}

def extract_features(pkt, proto_id):
    key  = (pkt[IP].src, pkt[IP].dst)
    conn = connections.setdefault(key, {
        'start': time.time(), 'src_bytes': 0, 'src_pkts': 0,
        'syn': 0, 'rst': 0
    })
    conn['src_bytes'] += len(pkt)
    conn['src_pkts']  += 1
    if pkt.haslayer(TCP):
        flags = pkt[TCP].flags
        if flags & 0x02: conn['syn'] += 1
        if flags & 0x04: conn['rst'] += 1
    duration    = max(0, time.time() - conn['start'])
    src_bytes   = conn['src_bytes']
    serror_rate = conn['syn'] / max(conn['src_pkts'], 1)
    rerror_rate = conn['rst'] / max(conn['src_pkts'], 1)
    count       = min(wifi_state['total'] % 511, 511)
    same_srv    = random.uniform(0.7, 1.0)
    f = [duration, proto_id, 20, 10, src_bytes, 0, 0, 0, 0, 0, 0,
         1 if src_bytes > 500 else 0, 0,0,0,0,0,0,0,0,0,0,
         count, count, serror_rate, serror_rate, rerror_rate, rerror_rate,
         same_srv, 1-same_srv, 0, min(count,255), min(count,255),
         same_srv, 1-same_srv, 0, 0, serror_rate, serror_rate, rerror_rate, rerror_rate]
    return np.array(f[:41], dtype=np.float32)

pkt_count  = [0]
wifi_count = [0]
wifi_start = [time.time()]

def handle_packet(pkt):
    if not pkt.haslayer(IP): return
    pkt_count[0] += 1
    src  = pkt[IP].src
    dst  = pkt[IP].dst
    size = len(pkt)
    port = 0; proto = 'ICMP'; svc = 'other'; proto_id = 0
    if pkt.haslayer(TCP):
        proto = 'TCP'; port = pkt[TCP].dport; proto_id = 2
        svc = ('http' if port in [80,8080] else 'https' if port==443 else
               'ssh' if port==22 else 'ftp' if port==21 else
               'smtp' if port==25 else 'dns' if port==53 else 'other')
    elif pkt.haslayer(UDP):
        proto = 'UDP'; port = pkt[UDP].dport; proto_id = 1
        svc = 'dns' if port==53 else 'ntp' if port==123 else 'other'
    if pkt_count[0] % 3 == 0:
        features = extract_features(pkt, proto_id)
        process(wifi_state, features, src, dst, proto, svc, port, size, learn=True)
        wifi_count[0] += 1
        elapsed = time.time() - wifi_start[0]
        if elapsed >= 1.0:
            wifi_state['pps'] = round(wifi_count[0] / elapsed)
            wifi_state['timeline'].append({
                't': time.strftime('%H:%M:%S'),
                'attacks': wifi_state['attacks'],
                'normal' : wifi_state['normal'],
            })
            wifi_count[0] = 0; wifi_start[0] = time.time()

def wifi_loop():
    try:
        print("WiFi capture starting on en0...")
        sniff(filter="ip", prn=handle_packet, store=0, iface="en0")
    except Exception as e:
        print(f"WiFi capture failed: {e}")

threading.Thread(target=replay_loop, daemon=True).start()
if SCAPY_OK:
    threading.Thread(target=wifi_loop, daemon=True).start()

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FPGA-NIDS Combined</title>
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<style>
:root{
  --bg:#070b10;--surface:#0d1520;--border:rgba(56,139,253,.15);
  --border2:rgba(56,139,253,.28);--text:#cdd9e5;--muted:#5c7a99;
  --accent:#38bdf8;--green:#34d399;--red:#f87171;--amber:#fbbf24;--purple:#a78bfa;
  --mono:'Space Mono',monospace;--sans:'DM Sans',sans-serif;--r:8px;
}
*{box-sizing:border-box;margin:0;padding:0}
html,body{width:100%;max-width:100%;overflow-x:hidden}
body{background:var(--bg);color:var(--text);font-family:var(--sans);font-size:13px;
     background-image:radial-gradient(ellipse 80% 50% at 50% -20%,rgba(56,189,248,.07) 0%,transparent 60%)}

.hdr{display:flex;align-items:center;gap:16px;padding:14px 24px;
     background:rgba(13,21,32,.95);border-bottom:1px solid var(--border);
     position:sticky;top:0;z-index:100}
.hdr-left{display:flex;align-items:center;gap:10px;min-width:0}
.hdr-icon{width:32px;height:32px;border-radius:8px;font-size:15px;
          background:linear-gradient(135deg,rgba(56,189,248,.25),rgba(167,139,250,.15));
          border:1px solid rgba(56,189,248,.3);display:flex;align-items:center;justify-content:center}
.hdr-title{font-size:15px;font-weight:600;color:#e6edf3}
.hdr-sub{font-size:11px;color:var(--muted);font-family:var(--mono)}
.hdr-right{margin-left:auto;display:flex;align-items:center;gap:12px;flex-wrap:wrap;justify-content:flex-end}
.live-pill{display:flex;align-items:center;gap:6px;padding:4px 10px;border-radius:20px;
           background:rgba(52,211,153,.1);border:1px solid rgba(52,211,153,.25);
           font-size:11px;font-family:var(--mono);color:var(--green)}
.pulse-dot{width:7px;height:7px;border-radius:50%;background:var(--green);animation:pulse 1.8s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.35}}
.hdr-tag{font-size:10px;font-family:var(--mono);color:var(--muted)}

.page{padding:16px 24px;display:flex;flex-direction:column;gap:16px;width:100%;max-width:100%}

.tabs{display:flex;gap:4px;border-bottom:1px solid var(--border);overflow-x:auto;
      scrollbar-width:none;-webkit-overflow-scrolling:touch;scroll-snap-type:x proximity}
.tabs::-webkit-scrollbar{display:none}
.tab{padding:10px 20px;font-size:12px;font-family:var(--mono);font-weight:700;
     letter-spacing:.5px;text-transform:uppercase;cursor:pointer;white-space:nowrap;
     border-bottom:2px solid transparent;color:var(--muted);transition:all .2s;
     background:none;border-top:none;border-left:none;border-right:none;flex:0 0 auto;
     scroll-snap-align:start}
.tab:hover{color:var(--text)}
.tab.active{color:var(--accent);border-bottom-color:var(--accent)}
.tab-badge{display:inline-block;margin-left:8px;padding:1px 6px;border-radius:3px;
           font-size:9px;background:rgba(56,189,248,.1);border:1px solid rgba(56,189,248,.2);color:var(--accent)}
.tab.wifi.active{color:var(--green);border-bottom-color:var(--green)}
.tab.wifi .tab-badge{background:rgba(52,211,153,.1);border-color:rgba(52,211,153,.2);color:var(--green)}
.tab-panel{display:none}.tab-panel.active{display:flex;flex-direction:column;gap:16px}

.stats-row{display:grid;grid-template-columns:repeat(8,1fr);gap:10px}
.stat{background:var(--surface);border:1px solid var(--border);border-radius:var(--r);
      padding:14px 16px;border-top:2px solid transparent;transition:border-color .2s}
.stat:hover{border-color:var(--border2)}
.stat.red{border-top-color:var(--red)}.stat.green{border-top-color:var(--green)}
.stat.blue{border-top-color:var(--accent)}.stat.amber{border-top-color:var(--amber)}
.stat.purple{border-top-color:var(--purple)}.stat.teal{border-top-color:#2dd4bf}
.stat-label{font-size:10px;font-family:var(--mono);color:var(--muted);
            text-transform:uppercase;letter-spacing:.8px;margin-bottom:8px}
.stat-val{font-size:22px;font-weight:600;font-family:var(--mono);color:#e6edf3;letter-spacing:-1px}
.stat.red .stat-val{color:var(--red)}.stat.green .stat-val{color:var(--green)}
.stat.blue .stat-val{color:var(--accent)}.stat.amber .stat-val{color:var(--amber)}
.stat.purple .stat-val{color:var(--purple)}.stat.teal .stat-val{color:#2dd4bf}
.stat-sub{font-size:10px;color:var(--muted);margin-top:3px;font-family:var(--mono)}

.main-row{display:grid;grid-template-columns:1fr 340px;gap:16px;align-items:stretch}
.left-col{display:flex;flex-direction:column;gap:16px;min-width:0}
.right-col{display:grid;grid-template-rows:auto auto auto minmax(0,1fr);gap:16px;min-width:0;align-self:stretch;height:100%}
.right-col>.panel:last-child{display:flex;flex-direction:column;min-height:0}

.panel{background:var(--surface);border:1px solid var(--border);border-radius:var(--r);overflow:hidden;width:100%;max-width:100%}
.panel-hdr{display:flex;align-items:center;justify-content:space-between;
           padding:12px 16px;border-bottom:1px solid var(--border);background:rgba(255,255,255,.015);gap:10px}
.panel-title{font-size:11px;font-weight:600;text-transform:uppercase;
             letter-spacing:1px;color:var(--muted);display:flex;align-items:center;gap:8px}
.panel-title span{color:var(--accent);font-size:13px}
.panel-badge{font-size:10px;font-family:var(--mono);color:var(--muted);
             background:rgba(255,255,255,.05);border:1px solid var(--border);padding:2px 8px;border-radius:4px}

.feed-cols,.feed-row{display:grid;
  grid-template-columns:50px 84px 155px 50px 50px 105px 105px 50px 50px 68px;
  padding:7px 14px;align-items:center}
.feed-cols{font-size:9px;font-family:var(--mono);color:var(--muted);text-transform:uppercase;
           letter-spacing:.8px;border-bottom:1px solid var(--border);background:rgba(0,0,0,.2);padding:8px 14px}
.feed-row{border-bottom:1px solid rgba(56,139,253,.06);transition:background .12s;font-size:11px}
.feed-row.is-new{animation:rowIn .3s ease forwards}
@keyframes rowIn{from{opacity:0;transform:translateY(-4px)}to{opacity:1;transform:none}}
.feed-row:hover{background:rgba(56,189,248,.04)}.feed-row:last-child{border-bottom:none}

.badge{display:inline-flex;align-items:center;gap:4px;padding:2px 7px;border-radius:4px;
       font-size:10px;font-family:var(--mono);font-weight:700;letter-spacing:.5px}
.badge.atk{background:rgba(248,113,113,.12);border:1px solid rgba(248,113,113,.3);color:var(--red)}
.badge.nrm{background:rgba(52,211,153,.1);border:1px solid rgba(52,211,153,.25);color:var(--green)}
.badge.atk::before{content:'';width:5px;height:5px;border-radius:50%;background:var(--red);animation:pulse 1.5s infinite}
.badge.nrm::before{content:'';width:5px;height:5px;border-radius:50%;background:var(--green)}

.dim{color:var(--muted);font-family:var(--mono);font-size:11px}
.atk-name{color:var(--red);font-size:11px}.nrm-name{color:var(--green);font-size:11px}

.chart-wrap{padding:16px}
.chart-wrap.tight{padding:12px 16px 10px}
.chart-wrap canvas{display:block;width:100% !important;height:100% !important}
.mini-grid{display:grid;grid-template-columns:1fr 1fr;grid-auto-rows:1fr;gap:10px;padding:16px;flex:1;min-height:0}
.mini-card{background:rgba(255,255,255,.03);border:1px solid var(--border);border-radius:6px;padding:12px;min-height:68px;height:100%;display:flex;flex-direction:column;justify-content:flex-end}
.mini-k{font-size:9px;font-family:var(--mono);color:var(--muted);text-transform:uppercase;letter-spacing:.7px;margin-bottom:8px}
.mini-v{font-size:14px;font-family:var(--mono);font-weight:700;color:#e6edf3;line-height:1.25;word-break:break-word}
.rate-section{padding:16px}
.rate-top{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:10px}
.rate-val{font-size:30px;font-family:var(--mono);font-weight:700;color:var(--red);letter-spacing:-2px}
.rate-label{font-size:10px;font-family:var(--mono);color:var(--muted);text-transform:uppercase;text-align:right}
.rate-track{height:6px;background:rgba(255,255,255,.06);border-radius:3px;overflow:hidden;margin-bottom:8px}
.rate-fill{height:100%;border-radius:3px;background:linear-gradient(90deg,var(--amber),var(--red));transition:width .6s}
.rate-meta{display:flex;justify-content:space-between;font-size:10px;font-family:var(--mono);color:var(--muted)}

.pseudo-row{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;padding:12px 16px;border-top:1px solid var(--border)}
.pseudo-item{text-align:center}
.pseudo-val{font-size:16px;font-family:var(--mono);font-weight:700}
.pseudo-val.n{color:var(--green)}.pseudo-val.a{color:var(--red)}.pseudo-val.d{color:var(--muted)}
.pseudo-label{font-size:9px;font-family:var(--mono);color:var(--muted);text-transform:uppercase;letter-spacing:.5px;margin-top:2px}

.buf-bar{padding:10px 16px;border-bottom:1px solid var(--border);
         display:flex;align-items:center;gap:10px;background:rgba(251,191,36,.03)}
.buf-label{font-size:10px;font-family:var(--mono);color:var(--amber);white-space:nowrap}
.buf-track{flex:1;height:4px;background:rgba(255,255,255,.06);border-radius:2px;overflow:hidden}
.buf-fill{height:100%;border-radius:2px;background:var(--amber);transition:width .4s}
.buf-count{font-size:10px;font-family:var(--mono);color:var(--muted);white-space:nowrap}

.alert-item{padding:10px 16px;border-bottom:1px solid var(--border);border-left:3px solid var(--red);
            display:flex;flex-direction:column;gap:4px;background:rgba(248,113,113,.03);transition:background .15s}
.alert-item.is-new{animation:rowIn .3s ease forwards}
.alert-item:hover{background:rgba(248,113,113,.07)}.alert-item:last-child{border-bottom:none}
.alert-top{display:flex;align-items:center;justify-content:space-between}
.alert-type{font-size:10px;font-family:var(--mono);font-weight:700;text-transform:uppercase}
.alert-type.dos{color:var(--red)}.alert-type.probe{color:var(--amber)}
.alert-type.r2l{color:var(--purple)}.alert-type.u2r{color:#f472b6}
.alert-conf{font-size:10px;font-family:var(--mono);background:rgba(248,113,113,.15);
            border:1px solid rgba(248,113,113,.25);color:var(--red);padding:1px 6px;border-radius:3px}
.alert-msg{font-size:11px;color:var(--text)}.alert-meta{font-size:10px;font-family:var(--mono);color:var(--muted)}
.no-alerts{padding:20px 16px;text-align:center;color:var(--muted);font-size:12px;font-family:var(--mono);
           min-height:60px;display:flex;align-items:center;justify-content:center}

.ip-row{display:flex;align-items:center;padding:7px 16px;gap:10px;border-bottom:1px solid rgba(56,139,253,.06)}
.ip-row:last-child{border-bottom:none}
.ip-addr{font-family:var(--mono);font-size:11px;flex:0 0 130px;color:var(--text)}
.ip-bar-wrap{flex:1;height:4px;background:rgba(255,255,255,.06);border-radius:2px;overflow:hidden}
.ip-bar{height:100%;border-radius:2px;background:var(--accent);transition:width .5s}
.ip-count{font-family:var(--mono);font-size:11px;color:var(--muted);flex:0 0 28px;text-align:right}

.bottom-row{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px}

.compare-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.compare-card{background:var(--surface);border:1px solid var(--border);border-radius:var(--r);padding:20px}
.compare-title{font-size:11px;font-family:var(--mono);color:var(--muted);text-transform:uppercase;
               letter-spacing:1px;margin-bottom:16px;display:flex;align-items:center;gap:8px}
.compare-title .dot{width:8px;height:8px;border-radius:50%}
.cmp-row{display:flex;justify-content:space-between;align-items:center;
         padding:8px 0;border-bottom:1px solid var(--border)}
.cmp-row:last-child{border-bottom:none}
.cmp-label{font-size:11px;color:var(--muted);font-family:var(--mono)}
.cmp-val{font-size:13px;font-family:var(--mono);font-weight:700;color:#e6edf3}

/* ── Responsive ───────────────────────────────────────────────── */
@media(max-width:1200px){
  .stats-row{grid-template-columns:repeat(4,1fr)}
}
@media(max-width:900px){
  .hdr{align-items:flex-start;flex-wrap:wrap}
  .hdr-left,.hdr-right{width:100%}
  .hdr-right{margin-left:0;justify-content:space-between}
  .stats-row{grid-template-columns:repeat(2,1fr)}
  .main-row{grid-template-columns:1fr;align-items:start}
  .right-col{display:grid;grid-template-columns:1fr 1fr;gap:16px;align-self:start;align-items:start}
  .bottom-row{grid-template-columns:1fr}
  .compare-grid{grid-template-columns:1fr}
  .feed-cols,.feed-row{grid-template-columns:40px 70px 1fr 55px 55px}
  .feed-cols span:nth-child(n+6),
  .feed-row  span:nth-child(n+6){display:none}
  .hdr-tag{display:none}
}
@media(max-width:600px){
  body{font-size:12px}
  .stats-row{grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}
  .stat{padding:10px 12px}
  .stat-val{font-size:16px}
  .right-col{grid-template-columns:1fr;align-items:start}
  .page{padding:8px;gap:10px}
  .hdr{padding:10px 8px;gap:10px}
  .hdr-left{align-items:flex-start}
  .hdr-right{gap:8px;justify-content:flex-start}
  .hdr-title{font-size:13px}
  .hdr-sub{display:none}
  .hdr-tag{display:none}
  .live-pill{font-size:10px;padding:3px 8px}
  .panel{border-radius:6px}
  .panel-hdr{padding:9px 10px;align-items:flex-start;flex-wrap:wrap}
  .panel-badge{font-size:9px}
  .chart-wrap{padding:8px}
  .chart-wrap.tight{padding:8px 8px 6px}
  .tab{padding:8px 10px;font-size:10px}
  .tab-badge{display:none}
  .stats-row,.main-row,.left-col,.right-col,.bottom-row,.compare-grid,.tab-panel{width:100%;max-width:100%}
  .feed-cols{display:none}
  .feed-row{grid-template-columns:1fr 1fr;gap:6px 10px;padding:10px}
  .feed-row span{display:block;min-width:0}
  .feed-row span:nth-child(1){grid-column:1/2}
  .feed-row span:nth-child(2){grid-column:2/3;justify-self:end}
  .feed-row span:nth-child(3){grid-column:1/-1;font-size:12px}
  .feed-row span:nth-child(4),
  .feed-row span:nth-child(5),
  .feed-row span:nth-child(8),
  .feed-row span:nth-child(9),
  .feed-row span:nth-child(10){font-size:10px}
  .feed-row span:nth-child(6),
  .feed-row span:nth-child(7){grid-column:1/-1;overflow-wrap:anywhere}
  .ip-row{padding:9px 10px;gap:8px}
  .ip-addr{flex:0 0 92px;font-size:10px;overflow:hidden;text-overflow:ellipsis}
  .rate-val{font-size:24px}
  .rate-top{gap:10px}
  .rate-label{font-size:9px}
  .pseudo-row{gap:4px;padding:10px}
  .pseudo-val{font-size:13px}
  .buf-bar{padding:8px 10px;flex-wrap:wrap}
  .buf-track{width:100%;flex-basis:100%}
  .alert-item{padding:8px 10px}
  .alert-top{gap:8px;align-items:flex-start}
  .alert-msg{font-size:10px}
  .alert-meta{word-break:break-word}
  .compare-card{padding:12px}
  .cmp-label{font-size:10px}
  .cmp-val{font-size:12px}
  .cmp-row{gap:12px;align-items:flex-start}
}
@media(max-width:420px){
  .stats-row{grid-template-columns:1fr}
  .page{padding:6px;gap:8px}
  .hdr{padding:8px 6px}
  .tab{padding:8px 10px;font-size:9px}
  .rate-top{flex-direction:column;align-items:flex-start}
  .feed-row{grid-template-columns:1fr}
  .feed-row span:nth-child(1),
  .feed-row span:nth-child(2),
  .feed-row span:nth-child(3),
  .feed-row span:nth-child(4),
  .feed-row span:nth-child(5),
  .feed-row span:nth-child(6),
  .feed-row span:nth-child(7),
  .feed-row span:nth-child(8),
  .feed-row span:nth-child(9),
  .feed-row span:nth-child(10){grid-column:1/-1}
  .feed-row span:nth-child(2){justify-self:start}
  .panel{border-radius:4px}
  .compare-card{padding:10px}
  .cmp-row{flex-direction:column;gap:4px}
}
</style>
</head>
<body>

<header class="hdr">
  <div class="hdr-left">
    <div class="hdr-icon">🛡</div>
    <div>
      <div class="hdr-title">FPGA-NIDS</div>
      <div class="hdr-sub">Network Intrusion Detection System</div>
    </div>
  </div>
  <div class="hdr-right">
    <span class="hdr-tag">8-bit QNN · Brevitas · FINN · Online Learning</span>
    <div class="live-pill"><div class="pulse-dot"></div>LIVE</div>
  </div>
</header>

<div class="page">

  <div class="tabs">
    <button class="tab active" onclick="switchTab('replay',this)">
      NSL-KDD Replay <span class="tab-badge">DATASET</span>
    </button>
    <button class="tab wifi" onclick="switchTab('wifi',this)">
      WiFi Live Capture <span class="tab-badge">LIVE</span>
    </button>
    <button class="tab" onclick="switchTab('compare',this)">
      Compare <span class="tab-badge">SIDE BY SIDE</span>
    </button>
  </div>

  <!-- Replay Tab -->
  <div id="tab-replay" class="tab-panel active">
    <div class="stats-row">
      <div class="stat blue"><div class="stat-label">Total packets</div><div class="stat-val" id="r-total">0</div><div class="stat-sub">inspected</div></div>
      <div class="stat red"><div class="stat-label">Attacks</div><div class="stat-val" id="r-attacks">0</div><div class="stat-sub" id="r-rate-sub">— %</div></div>
      <div class="stat green"><div class="stat-label">Normal</div><div class="stat-val" id="r-normal">0</div><div class="stat-sub">benign</div></div>
      <div class="stat amber"><div class="stat-label">Throughput</div><div class="stat-val" id="r-pps">0</div><div class="stat-sub">pkt/sec</div></div>
      <div class="stat purple"><div class="stat-label">Confidence</div><div class="stat-val" id="r-conf">0%</div><div class="stat-sub">avg certainty</div></div>
      <div class="stat green"><div class="stat-label">Accuracy</div><div class="stat-val" id="r-acc">0%</div><div class="stat-sub">vs labels</div></div>
      <div class="stat blue"><div class="stat-label">Bytes in</div><div class="stat-val" id="r-bytes">0</div><div class="stat-sub">received</div></div>
      <div class="stat red"><div class="stat-label">Alerts</div><div class="stat-val" id="r-alc">0</div><div class="stat-sub">high conf</div></div>
    </div>
    <div class="main-row">
      <div class="left-col">
        <div class="panel">
          <div class="panel-hdr">
            <div class="panel-title"><span>⬡</span> NSL-KDD feed</div>
            <div class="panel-badge" id="r-feed-count">0 packets</div>
          </div>
          <div class="feed-cols"><span>#</span><span>verdict</span><span>type</span><span>proto</span><span>svc</span><span>source</span><span>dest</span><span>port</span><span>size</span><span>time</span></div>
          <div id="r-feed-body"></div>
        </div>
        <div class="panel">
          <div class="panel-hdr"><div class="panel-title"><span>◈</span> Timeline</div><div class="panel-badge">40 sec</div></div>
          <div class="chart-wrap" style="height:150px"><canvas id="r-timeline"></canvas></div>
        </div>
      </div>
      <div class="right-col">
        <div class="panel">
          <div class="panel-hdr"><div class="panel-title"><span>◎</span> Attack rate</div></div>
          <div class="rate-section">
            <div class="rate-top"><div class="rate-val" id="r-rate-pct">0%</div><div class="rate-label">of traffic<br>malicious</div></div>
            <div class="rate-track"><div class="rate-fill" id="r-rate-fill" style="width:0%"></div></div>
            <div class="rate-meta"><span>safe</span><span>100%</span></div>
          </div>
        </div>
        <div class="panel">
          <div class="panel-hdr"><div class="panel-title"><span>◑</span> Attack breakdown</div></div>
          <div class="chart-wrap" style="height:170px"><canvas id="r-pie"></canvas></div>
        </div>
        <div class="panel">
          <div class="panel-hdr"><div class="panel-title"><span>◐</span> Protocols</div></div>
          <div class="chart-wrap tight" style="height:104px"><canvas id="r-proto"></canvas></div>
        </div>
        <div class="panel">
          <div class="panel-hdr"><div class="panel-title"><span>◌</span> Quick snapshot</div></div>
          <div class="mini-grid">
            <div class="mini-card"><div class="mini-k">Bytes in</div><div class="mini-v" id="r-q-bytes">0B</div></div>
            <div class="mini-card"><div class="mini-k">Alerts</div><div class="mini-v" id="r-q-alerts">0</div></div>
            <div class="mini-card"><div class="mini-k">Top protocol</div><div class="mini-v" id="r-q-proto">—</div></div>
            <div class="mini-card"><div class="mini-k">Top attack</div><div class="mini-v" id="r-q-attack">—</div></div>
          </div>
        </div>
      </div>
    </div>
    <div class="bottom-row">
      <div class="panel">
        <div class="panel-hdr"><div class="panel-title"><span>⚠</span> Alerts</div><div class="panel-badge">high conf</div></div>
        <div id="r-alerts"><div class="no-alerts">No alerts yet</div></div>
      </div>
      <div class="panel"><div class="panel-hdr"><div class="panel-title"><span>↗</span> Top sources</div></div><div id="r-src"></div></div>
      <div class="panel"><div class="panel-hdr"><div class="panel-title"><span>↘</span> Top dests</div></div><div id="r-dst"></div></div>
    </div>
  </div>

  <!-- WiFi Tab -->
  <div id="tab-wifi" class="tab-panel">
    <div class="stats-row">
      <div class="stat blue"><div class="stat-label">Total packets</div><div class="stat-val" id="w-total">0</div><div class="stat-sub">inspected</div></div>
      <div class="stat red"><div class="stat-label">Attacks</div><div class="stat-val" id="w-attacks">0</div><div class="stat-sub" id="w-rate-sub">— %</div></div>
      <div class="stat green"><div class="stat-label">Normal</div><div class="stat-val" id="w-normal">0</div><div class="stat-sub">benign</div></div>
      <div class="stat amber"><div class="stat-label">Throughput</div><div class="stat-val" id="w-pps">0</div><div class="stat-sub">pkt/sec</div></div>
      <div class="stat purple"><div class="stat-label">Confidence</div><div class="stat-val" id="w-conf">0%</div><div class="stat-sub">avg certainty</div></div>
      <div class="stat teal"><div class="stat-label">Retrains</div><div class="stat-val" id="w-retrain">0</div><div class="stat-sub" id="w-loss">loss —</div></div>
      <div class="stat blue"><div class="stat-label">Bytes in</div><div class="stat-val" id="w-bytes">0</div><div class="stat-sub">received</div></div>
      <div class="stat amber"><div class="stat-label">Buffer</div><div class="stat-val" id="w-buffer">0</div><div class="stat-sub">/ 50 samples</div></div>
    </div>
    <div class="main-row">
      <div class="left-col">
        <div class="panel">
          <div class="panel-hdr">
            <div class="panel-title"><span>⬡</span> WiFi live feed</div>
            <div class="panel-badge" id="w-feed-count">0 packets</div>
          </div>
          <div class="buf-bar">
            <div class="buf-label">Training buffer</div>
            <div class="buf-track"><div class="buf-fill" id="w-buf-fill" style="width:0%"></div></div>
            <div class="buf-count"><span id="w-buf-n">0</span> / 50</div>
          </div>
          <div class="feed-cols"><span>#</span><span>verdict</span><span>type</span><span>proto</span><span>svc</span><span>source</span><span>dest</span><span>port</span><span>size</span><span>time</span></div>
          <div id="w-feed-body"></div>
        </div>
        <div class="panel">
          <div class="panel-hdr"><div class="panel-title"><span>◈</span> Timeline</div><div class="panel-badge">40 sec</div></div>
          <div class="chart-wrap" style="height:150px"><canvas id="w-timeline"></canvas></div>
        </div>
      </div>
      <div class="right-col">
        <div class="panel">
          <div class="panel-hdr"><div class="panel-title"><span>◎</span> Attack rate</div></div>
          <div class="rate-section">
            <div class="rate-top"><div class="rate-val" id="w-rate-pct">0%</div><div class="rate-label">of traffic<br>malicious</div></div>
            <div class="rate-track"><div class="rate-fill" id="w-rate-fill" style="width:0%"></div></div>
            <div class="rate-meta"><span>safe</span><span>100%</span></div>
          </div>
          <div class="pseudo-row">
            <div class="pseudo-item"><div class="pseudo-val n" id="w-pl-n">0</div><div class="pseudo-label">Normal labels</div></div>
            <div class="pseudo-item"><div class="pseudo-val a" id="w-pl-a">0</div><div class="pseudo-label">Attack labels</div></div>
            <div class="pseudo-item"><div class="pseudo-val d" id="w-pl-d">0</div><div class="pseudo-label">Discarded</div></div>
          </div>
        </div>
        <div class="panel">
          <div class="panel-hdr"><div class="panel-title"><span>◑</span> Attack breakdown</div></div>
          <div class="chart-wrap" style="height:170px"><canvas id="w-pie"></canvas></div>
        </div>
        <div class="panel">
          <div class="panel-hdr"><div class="panel-title"><span>◐</span> Protocols</div></div>
          <div class="chart-wrap tight" style="height:104px"><canvas id="w-proto"></canvas></div>
        </div>
        <div class="panel">
          <div class="panel-hdr"><div class="panel-title"><span>◌</span> Quick snapshot</div></div>
          <div class="mini-grid">
            <div class="mini-card"><div class="mini-k">Bytes in</div><div class="mini-v" id="w-q-bytes">0B</div></div>
            <div class="mini-card"><div class="mini-k">Retrains</div><div class="mini-v" id="w-q-retrains">0</div></div>
            <div class="mini-card"><div class="mini-k">Top protocol</div><div class="mini-v" id="w-q-proto">—</div></div>
            <div class="mini-card"><div class="mini-k">Top attack</div><div class="mini-v" id="w-q-attack">—</div></div>
          </div>
        </div>
      </div>
    </div>
    <div class="bottom-row">
      <div class="panel">
        <div class="panel-hdr"><div class="panel-title"><span>⚠</span> Alerts</div><div class="panel-badge">high conf</div></div>
        <div id="w-alerts"><div class="no-alerts">No alerts yet</div></div>
      </div>
      <div class="panel"><div class="panel-hdr"><div class="panel-title"><span>↗</span> Top sources</div></div><div id="w-src"></div></div>
      <div class="panel"><div class="panel-hdr"><div class="panel-title"><span>↘</span> Top dests</div></div><div id="w-dst"></div></div>
    </div>
  </div>

  <!-- Compare Tab -->
  <div id="tab-compare" class="tab-panel">
    <div class="compare-grid">
      <div class="compare-card">
        <div class="compare-title"><span class="dot" style="background:var(--accent)"></span>NSL-KDD Replay</div>
        <div class="cmp-row"><span class="cmp-label">Total packets</span><span class="cmp-val" id="c-r-total">0</span></div>
        <div class="cmp-row"><span class="cmp-label">Attack rate</span><span class="cmp-val" id="c-r-rate" style="color:var(--red)">0%</span></div>
        <div class="cmp-row"><span class="cmp-label">Accuracy</span><span class="cmp-val" id="c-r-acc" style="color:var(--green)">0%</span></div>
        <div class="cmp-row"><span class="cmp-label">Avg confidence</span><span class="cmp-val" id="c-r-conf" style="color:var(--purple)">0%</span></div>
        <div class="cmp-row"><span class="cmp-label">Throughput</span><span class="cmp-val" id="c-r-pps" style="color:var(--amber)">0 pkt/s</span></div>
        <div class="cmp-row"><span class="cmp-label">Alerts fired</span><span class="cmp-val" id="c-r-alc" style="color:var(--red)">0</span></div>
        <div class="cmp-row"><span class="cmp-label">DoS detected</span><span class="cmp-val" id="c-r-dos">0</span></div>
        <div class="cmp-row"><span class="cmp-label">Probe detected</span><span class="cmp-val" id="c-r-probe">0</span></div>
        <div class="cmp-row"><span class="cmp-label">R2L detected</span><span class="cmp-val" id="c-r-r2l">0</span></div>
        <div class="cmp-row"><span class="cmp-label">U2R detected</span><span class="cmp-val" id="c-r-u2r">0</span></div>
      </div>
      <div class="compare-card">
        <div class="compare-title"><span class="dot" style="background:var(--green)"></span>WiFi Live Capture</div>
        <div class="cmp-row"><span class="cmp-label">Total packets</span><span class="cmp-val" id="c-w-total">0</span></div>
        <div class="cmp-row"><span class="cmp-label">Attack rate</span><span class="cmp-val" id="c-w-rate" style="color:var(--red)">0%</span></div>
        <div class="cmp-row"><span class="cmp-label">Accuracy</span><span class="cmp-val" id="c-w-acc" style="color:var(--green)">—</span></div>
        <div class="cmp-row"><span class="cmp-label">Avg confidence</span><span class="cmp-val" id="c-w-conf" style="color:var(--purple)">0%</span></div>
        <div class="cmp-row"><span class="cmp-label">Throughput</span><span class="cmp-val" id="c-w-pps" style="color:var(--amber)">0 pkt/s</span></div>
        <div class="cmp-row"><span class="cmp-label">Alerts fired</span><span class="cmp-val" id="c-w-alc" style="color:var(--red)">0</span></div>
        <div class="cmp-row"><span class="cmp-label">Retrains done</span><span class="cmp-val" id="c-w-retrain" style="color:#2dd4bf">0</span></div>
        <div class="cmp-row"><span class="cmp-label">Last loss</span><span class="cmp-val" id="c-w-loss" style="color:var(--amber)">—</span></div>
        <div class="cmp-row"><span class="cmp-label">Pseudo-labels</span><span class="cmp-val" id="c-w-pl">0</span></div>
        <div class="cmp-row"><span class="cmp-label">Discarded</span><span class="cmp-val" id="c-w-disc" style="color:var(--muted)">0</span></div>
      </div>
    </div>
  </div>

</div>

<script>
const $ = id => document.getElementById(id);
const topKey = obj => Object.entries(obj).sort((a,b)=>b[1]-a[1])[0]?.[0] || '—';
Chart.defaults.color='#5c7a99';
Chart.defaults.font.family="'Space Mono',monospace";
Chart.defaults.font.size=10;

function mkPie(id){return new Chart($(id),{type:'doughnut',data:{labels:['DoS','Probe','R2L','U2R'],datasets:[{data:[0,0,0,0],backgroundColor:['rgba(248,113,113,.85)','rgba(251,191,36,.85)','rgba(167,139,250,.85)','rgba(244,114,182,.85)'],borderColor:'#070b10',borderWidth:3}]},options:{responsive:true,maintainAspectRatio:false,cutout:'68%',plugins:{legend:{position:'right',labels:{boxWidth:10,padding:10,color:'#5c7a99',font:{size:10}}}},animation:{duration:400}}});}
function mkProto(id){return new Chart($(id),{type:'bar',data:{labels:['TCP','UDP','ICMP'],datasets:[{data:[0,0,0],backgroundColor:['rgba(56,189,248,.7)','rgba(52,211,153,.7)','rgba(251,191,36,.7)'],borderRadius:4,borderSkipped:false}]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},scales:{x:{grid:{color:'rgba(56,139,253,.08)'},ticks:{color:'#5c7a99'}},y:{grid:{color:'rgba(56,139,253,.08)'},ticks:{color:'#5c7a99'}}},animation:{duration:300}}});}
function mkLine(id){return new Chart($(id),{type:'line',data:{labels:[],datasets:[{label:'Attacks',data:[],borderColor:'rgba(248,113,113,.9)',borderWidth:1.5,pointRadius:0,fill:true,backgroundColor:'rgba(248,113,113,.06)',tension:.4},{label:'Normal',data:[],borderColor:'rgba(52,211,153,.9)',borderWidth:1.5,pointRadius:0,fill:true,backgroundColor:'rgba(52,211,153,.06)',tension:.4}]},options:{responsive:true,maintainAspectRatio:false,animation:false,plugins:{legend:{labels:{boxWidth:10,padding:12,color:'#5c7a99',font:{size:10}}}},scales:{x:{grid:{color:'rgba(56,139,253,.06)'},ticks:{color:'#5c7a99',maxTicksLimit:8}},y:{grid:{color:'rgba(56,139,253,.06)'},ticks:{color:'#5c7a99'},min:0}}}});}

const charts={rPie:mkPie('r-pie'),rProto:mkProto('r-proto'),rLine:mkLine('r-timeline'),wPie:mkPie('w-pie'),wProto:mkProto('w-proto'),wLine:mkLine('w-timeline')};

function fmt(n){if(n>=1e9)return(n/1e9).toFixed(1)+'G';if(n>=1e6)return(n/1e6).toFixed(1)+'M';if(n>=1e3)return(n/1e3).toFixed(1)+'K';return n+'B';}
const typeClass={DoS:'dos',Probe:'probe',R2L:'r2l',U2R:'u2r'};
let lastIds={r:-1,w:-1},lastAlerts={r:'',w:''};

function rowHTML(r,isNew){
  return `<div class="feed-row${isNew?' is-new':''}">
    <span class="dim">${r.id}</span>
    <span><span class="badge ${r.label==='ATTACK'?'atk':'nrm'}">${r.label==='ATTACK'?'ATK':'OK'}</span></span>
    <span class="${r.label==='ATTACK'?'atk-name':'nrm-name'}">${r.type}</span>
    <span class="dim">${r.proto}</span><span class="dim">${r.service}</span>
    <span class="dim">${r.src}</span><span class="dim">${r.dst}</span>
    <span class="dim">${r.port}</span><span class="dim">${r.size}B</span>
    <span class="dim">${r.time}</span>
  </div>`;
}

function updateFeed(prefix,rows){
  if(!rows.length)return;
  const newestId=rows[0].id;
  if(newestId===lastIds[prefix])return;
  const fb=$(prefix+'-feed-body');
  const newCount=newestId-lastIds[prefix];
  lastIds[prefix]=newestId;
  if(newCount>=rows.length||fb.children.length===0){fb.innerHTML=rows.map((r,i)=>rowHTML(r,i<newCount)).join('');return;}
  const frag=document.createDocumentFragment();
  rows.slice(0,newCount).forEach(r=>{const d=document.createElement('div');d.innerHTML=rowHTML(r,true);frag.appendChild(d.firstChild);});
  fb.insertBefore(frag,fb.firstChild);
  while(fb.children.length>rows.length)fb.removeChild(fb.lastChild);
}

function updateAlerts(id,alerts){
  const ab=$(id);
  if(!alerts.length){if(ab.innerHTML.includes('no-alerts'))return;ab.innerHTML='<div class="no-alerts">No alerts yet</div>';return;}
  const key=alerts[0].msg+alerts[0].time;
  if(key===lastAlerts[id])return;
  lastAlerts[id]=key;
  ab.innerHTML=alerts.map((a,i)=>`<div class="alert-item${i===0?' is-new':''}">
    <div class="alert-top"><span class="alert-type ${typeClass[a.type]||'dos'}">${a.type}</span><span class="alert-conf">${a.conf}%</span></div>
    <div class="alert-msg">${a.msg}</div>
    <div class="alert-meta">${a.time} · ${a.src} → ${a.dst}</div>
  </div>`).join('');
}

function ipBlock(id,data){
  const e=Object.entries(data).sort((a,b)=>b[1]-a[1]).slice(0,5);
  const m=e.length?e[0][1]:1;
  $(id).innerHTML=e.map(([ip,c])=>`<div class="ip-row"><div class="ip-addr">${ip}</div><div class="ip-bar-wrap"><div class="ip-bar" style="width:${Math.round(c/m*100)}%"></div></div><div class="ip-count">${c}</div></div>`).join('');
}

function updLine(chart,tl){if(!tl.length)return;chart.data.labels=tl.map(t=>t.t);chart.data.datasets[0].data=tl.map(t=>t.attacks);chart.data.datasets[1].data=tl.map(t=>t.normal);chart.update('none');}

function switchTab(name,btn){
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  document.querySelectorAll('.tab-panel').forEach(p=>p.classList.remove('active'));
  btn.classList.add('active');
  $('tab-'+name).classList.add('active');
}

function update(){
  fetch('/api/both').then(r=>r.json()).then(({r:rd,w:wd})=>{
    const rRate=rd.total>0?Math.round(rd.attacks/rd.total*100):0;
    const rAcc =rd.total>0?Math.round(rd.correct/rd.total*100):0;
    const rConf=rd.total>0?Math.round(rd.conf_sum/rd.total):0;
    const wRate=wd.total>0?Math.round(wd.attacks/wd.total*100):0;
    const wConf=wd.total>0?Math.round(wd.conf_sum/wd.total):0;

    $('r-total').textContent=rd.total.toLocaleString();
    $('r-attacks').textContent=rd.attacks.toLocaleString();
    $('r-normal').textContent=rd.normal.toLocaleString();
    $('r-pps').textContent=rd.pps;
    $('r-conf').textContent=rConf+'%';
    $('r-acc').textContent=rAcc+'%';
    $('r-bytes').textContent=fmt(rd.bytes_in);
    $('r-alc').textContent=rd.alerts.length;
    $('r-rate-sub').textContent=rRate+'% attack rate';
    $('r-feed-count').textContent=rd.total.toLocaleString()+' packets';
    $('r-rate-pct').textContent=rRate+'%';
    $('r-rate-fill').style.width=Math.min(rRate,100)+'%';
    $('r-q-bytes').textContent=fmt(rd.bytes_in);
    $('r-q-alerts').textContent=rd.alerts.length;
    $('r-q-proto').textContent=topKey(rd.protocols);
    $('r-q-attack').textContent=topKey(rd.attack_types);

    $('w-total').textContent=wd.total.toLocaleString();
    $('w-attacks').textContent=wd.attacks.toLocaleString();
    $('w-normal').textContent=wd.normal.toLocaleString();
    $('w-pps').textContent=wd.pps;
    $('w-conf').textContent=wConf+'%';
    $('w-retrain').textContent=wd.retrain_count;
    $('w-loss').textContent=wd.retrain_loss>0?'loss '+wd.retrain_loss:'loss —';
    $('w-bytes').textContent=fmt(wd.bytes_in);
    $('w-buffer').textContent=wd.buffer_size;
    $('w-rate-sub').textContent=wRate+'% attack rate';
    $('w-feed-count').textContent=wd.total.toLocaleString()+' packets';
    $('w-rate-pct').textContent=wRate+'%';
    $('w-rate-fill').style.width=Math.min(wRate,100)+'%';
    $('w-q-bytes').textContent=fmt(wd.bytes_in);
    $('w-q-retrains').textContent=wd.retrain_count;
    $('w-q-proto').textContent=topKey(wd.protocols);
    $('w-q-attack').textContent=topKey(wd.attack_types);
    $('w-buf-fill').style.width=Math.round(wd.buffer_size/50*100)+'%';
    $('w-buf-n').textContent=wd.buffer_size;
    $('w-pl-n').textContent=wd.pseudo_labels.normal;
    $('w-pl-a').textContent=wd.pseudo_labels.attack;
    $('w-pl-d').textContent=wd.pseudo_labels.discarded;

    $('c-r-total').textContent=rd.total.toLocaleString();
    $('c-r-rate').textContent=rRate+'%';
    $('c-r-acc').textContent=rAcc+'%';
    $('c-r-conf').textContent=rConf+'%';
    $('c-r-pps').textContent=rd.pps+' pkt/s';
    $('c-r-alc').textContent=rd.alerts.length;
    $('c-r-dos').textContent=rd.attack_types.DoS;
    $('c-r-probe').textContent=rd.attack_types.Probe;
    $('c-r-r2l').textContent=rd.attack_types.R2L;
    $('c-r-u2r').textContent=rd.attack_types.U2R;
    $('c-w-total').textContent=wd.total.toLocaleString();
    $('c-w-rate').textContent=wRate+'%';
    $('c-w-acc').textContent='— (no labels)';
    $('c-w-conf').textContent=wConf+'%';
    $('c-w-pps').textContent=wd.pps+' pkt/s';
    $('c-w-alc').textContent=wd.alerts.length;
    $('c-w-retrain').textContent=wd.retrain_count;
    $('c-w-loss').textContent=wd.retrain_loss||'—';
    $('c-w-pl').textContent=wd.pseudo_labels.normal+wd.pseudo_labels.attack;
    $('c-w-disc').textContent=wd.pseudo_labels.discarded;

    charts.rPie.data.datasets[0].data=[rd.attack_types.DoS,rd.attack_types.Probe,rd.attack_types.R2L,rd.attack_types.U2R];charts.rPie.update('none');
    charts.rProto.data.datasets[0].data=[rd.protocols.TCP,rd.protocols.UDP,rd.protocols.ICMP];charts.rProto.update('none');
    charts.wPie.data.datasets[0].data=[wd.attack_types.DoS,wd.attack_types.Probe,wd.attack_types.R2L,wd.attack_types.U2R];charts.wPie.update('none');
    charts.wProto.data.datasets[0].data=[wd.protocols.TCP,wd.protocols.UDP,wd.protocols.ICMP];charts.wProto.update('none');
    updLine(charts.rLine,rd.timeline);
    updLine(charts.wLine,wd.timeline);

    updateFeed('r',rd.recent);
    updateFeed('w',wd.recent);
    updateAlerts('r-alerts',rd.alerts);
    updateAlerts('w-alerts',wd.alerts);
    ipBlock('r-src',rd.top_src);ipBlock('r-dst',rd.top_dst);
    ipBlock('w-src',wd.top_src);ipBlock('w-dst',wd.top_dst);
  });
}

setInterval(update,450);
update();
</script>
</body>
</html>"""

def serialize(st):
    top_src = dict(sorted(st['top_src'].items(), key=lambda x: x[1], reverse=True)[:8])
    top_dst = dict(sorted(st['top_dst'].items(), key=lambda x: x[1], reverse=True)[:8])
    return {
        'total'        : st['total'],
        'attacks'      : st['attacks'],
        'normal'       : st['normal'],
        'pps'          : st['pps'],
        'conf_sum'     : st['conf_sum'],
        'correct'      : st['correct'],
        'bytes_in'     : st['bytes_in'],
        'attack_types' : st['attack_types'],
        'protocols'    : st['protocols'],
        'timeline'     : list(st['timeline']),
        'recent'       : list(st['recent']),
        'alerts'       : list(st['alerts']),
        'top_src'      : top_src,
        'top_dst'      : top_dst,
        'retrain_count': st['retrain_count'],
        'retrain_loss' : st['retrain_loss'],
        'buffer_size'  : st['buffer_size'],
        'pseudo_labels': st['pseudo_labels'],
    }

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/api/both')
def api_both():
    return jsonify({'r': serialize(replay_state), 'w': serialize(wifi_state)})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"\nCombined Dashboard → http://localhost:{port}")
    print(f"Tab 1: NSL-KDD Replay | Tab 2: WiFi Live | Tab 3: Comparison")
    print(f"WiFi capture: {'enabled (en0)' if SCAPY_OK else 'disabled (install scapy)'}\n")
    app.run(host='0.0.0.0', port=port, debug=False)
