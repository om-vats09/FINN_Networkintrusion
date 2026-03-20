from flask import Flask, jsonify, render_template_string
import torch
import numpy as np
import pickle
import threading
import time
import random
from model import build_model
from collections import deque

app = Flask(__name__)

import os
if not os.path.exists('models/model_8bit.pt'):
    print("Models not found — running setup...")
    os.system('python3 preprocess.py')
    os.system('python3 train.py')
    print("Models ready.")

def load_model():
    model = build_model(8)
    model.load_state_dict(torch.load('models/model_8bit.pt', map_location='cpu'))
    model.eval()
    return model

def load_scaler():
    with open('data/scaler.pkl', 'rb') as f:
        return pickle.load(f)

model  = load_model()
scaler = load_scaler()
X_test = np.load('data/X_test.npy')
y_test = np.load('data/y_test.npy')

state = {
    'total'       : 0,
    'attacks'     : 0,
    'normal'      : 0,
    'recent'      : deque(maxlen=12),
    'pps'         : 0,
    'attack_types': {'DoS': 0, 'Probe': 0, 'R2L': 0, 'U2R': 0, 'Unknown': 0},
    'running'     : True
}

ATTACK_NAMES = [
    'SYN Flood', 'UDP Flood', 'Port Scan', 'Ping Sweep',
    'FTP Brute Force', 'Root Exploit', 'ICMP Flood', 'Web Attack'
]
ATTACK_TYPES = ['DoS', 'DoS', 'Probe', 'Probe', 'R2L', 'U2R', 'DoS', 'R2L']

def predict_sample(features):
    inp   = scaler.transform(features.reshape(1, -1))
    t     = torch.tensor(inp, dtype=torch.float32)
    with torch.no_grad():
        out   = model(t)
        pred  = out.argmax(dim=1).item()
        probs = torch.softmax(out, dim=1).numpy()[0]
    return pred, float(probs[pred]) * 100

def feed_loop():
    idx        = 0
    count      = 0
    start_time = time.time()

    while state['running']:
        sample  = X_test[idx % len(X_test)]
        actual  = int(y_test[idx % len(y_test)])
        pred, conf = predict_sample(sample)

        state['total'] += 1
        count          += 1

        elapsed = time.time() - start_time
        if elapsed >= 1.0:
            state['pps'] = round(count / elapsed)
            count        = 0
            start_time   = time.time()

        if pred == 1:
            state['attacks'] += 1
            atype = random.choice(ATTACK_TYPES)
            aname = ATTACK_NAMES[ATTACK_TYPES.index(atype)]
            state['attack_types'][atype] += 1
        else:
            state['normal'] += 1
            aname = 'Normal traffic'
            atype = 'Normal'

        correct = (pred == actual)
        state['recent'].appendleft({
            'id'      : state['total'],
            'label'   : 'ATTACK' if pred == 1 else 'NORMAL',
            'type'    : aname,
            'conf'    : round(conf, 1),
            'correct' : correct,
            'src'     : f"192.168.{random.randint(1,254)}.{random.randint(1,254)}",
            'dst'     : f"10.0.{random.randint(0,10)}.{random.randint(1,50)}",
        })

        idx  += 1
        time.sleep(0.3)

thread = threading.Thread(target=feed_loop, daemon=True)
thread.start()

HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>IDS Live Dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #0d1117; color: #e6edf3; font-family: 'Segoe UI', Arial, sans-serif; }
.header { background: #161b22; border-bottom: 1px solid #30363d;
          padding: 14px 24px; display: flex; align-items: center; gap: 16px; }
.header h1 { font-size: 18px; font-weight: 500; color: #58a6ff; }
.dot { width: 10px; height: 10px; border-radius: 50%; background: #3fb950;
       animation: pulse 1.5s ease-in-out infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }
.grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 12px;
        padding: 16px 24px; }
.card { background: #161b22; border: 1px solid #30363d;
        border-radius: 10px; padding: 16px; }
.card h3 { font-size: 11px; color: #8b949e; text-transform: uppercase;
           letter-spacing: 1px; margin-bottom: 8px; }
.card .val { font-size: 28px; font-weight: 600; }
.val.green { color: #3fb950; }
.val.red   { color: #f85149; }
.val.blue  { color: #58a6ff; }
.val.amber { color: #d29922; }
.body { display: grid; grid-template-columns: 1fr 340px;
        gap: 12px; padding: 0 24px 24px; }
.feed { background: #161b22; border: 1px solid #30363d;
        border-radius: 10px; overflow: hidden; }
.feed-header { padding: 12px 16px; border-bottom: 1px solid #30363d;
               font-size: 13px; color: #8b949e; }
.feed-row { display: grid;
            grid-template-columns: 60px 90px 160px 140px 140px 80px;
            padding: 9px 16px; border-bottom: 1px solid #21262d;
            font-size: 12px; align-items: center; transition: background .2s; }
.feed-row:hover { background: #1c2128; }
.feed-labels { display: grid;
               grid-template-columns: 60px 90px 160px 140px 140px 80px;
               padding: 8px 16px; border-bottom: 1px solid #30363d;
               font-size: 11px; color: #8b949e; text-transform: uppercase; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 12px;
         font-size: 11px; font-weight: 500; }
.badge.attack { background: rgba(248,81,73,.15); color: #f85149; }
.badge.normal { background: rgba(63,185,80,.15); color: #3fb950; }
.right-col { display: flex; flex-direction: column; gap: 12px; }
.chart-card { background: #161b22; border: 1px solid #30363d;
              border-radius: 10px; padding: 16px; }
.chart-card h3 { font-size: 11px; color: #8b949e; text-transform: uppercase;
                 letter-spacing: 1px; margin-bottom: 12px; }
.rate-bar { height: 8px; background: #21262d; border-radius: 4px;
            margin-top: 8px; overflow: hidden; }
.rate-fill { height: 100%; border-radius: 4px; background: #f85149;
             transition: width .5s; }
.rate-label { font-size: 12px; color: #8b949e; margin-top: 6px; }
</style>
</head>
<body>
<div class="header">
  <div class="dot"></div>
  <h1>FPGA-NIDS Live Dashboard</h1>
  <span style="font-size:12px;color:#8b949e;margin-left:auto">
    NSL-KDD · 8-bit Quantized MLP · Brevitas
  </span>
</div>

<div class="grid">
  <div class="card">
    <h3>Total packets</h3>
    <div class="val blue" id="total">0</div>
  </div>
  <div class="card">
    <h3>Attacks detected</h3>
    <div class="val red" id="attacks">0</div>
  </div>
  <div class="card">
    <h3>Normal traffic</h3>
    <div class="val green" id="normal">0</div>
  </div>
  <div class="card">
    <h3>Predictions / sec</h3>
    <div class="val amber" id="pps">0</div>
  </div>
</div>

<div class="body">
  <div class="feed">
    <div class="feed-header">Live traffic feed</div>
    <div class="feed-labels">
      <span>#</span><span>Result</span><span>Type</span>
      <span>Source IP</span><span>Dest IP</span><span>Confidence</span>
    </div>
    <div id="feed-body"></div>
  </div>

  <div class="right-col">
    <div class="chart-card">
      <h3>Attack type breakdown</h3>
      <canvas id="pie" height="200"></canvas>
    </div>
    <div class="chart-card">
      <h3>Attack rate</h3>
      <div class="val red" id="rate-pct">0%</div>
      <div class="rate-bar"><div class="rate-fill" id="rate-fill" style="width:0%"></div></div>
      <div class="rate-label" id="rate-label">of all traffic is malicious</div>
    </div>
  </div>
</div>

<script>
const pie = new Chart(document.getElementById('pie'), {
  type: 'doughnut',
  data: {
    labels: ['DoS', 'Probe', 'R2L', 'U2R', 'Unknown'],
    datasets: [{ data: [0,0,0,0,0],
      backgroundColor: ['#f85149','#d29922','#a371f7','#58a6ff','#8b949e'],
      borderWidth: 0 }]
  },
  options: {
    responsive: true,
    plugins: { legend: { position: 'bottom',
      labels: { color: '#8b949e', font: { size: 11 }, boxWidth: 10 } } }
  }
});

function update() {
  fetch('/api/state').then(r => r.json()).then(d => {
    document.getElementById('total').textContent   = d.total.toLocaleString();
    document.getElementById('attacks').textContent = d.attacks.toLocaleString();
    document.getElementById('normal').textContent  = d.normal.toLocaleString();
    document.getElementById('pps').textContent     = d.pps;

    const rate = d.total > 0 ? Math.round(d.attacks / d.total * 100) : 0;
    document.getElementById('rate-pct').textContent  = rate + '%';
    document.getElementById('rate-fill').style.width = rate + '%';

    pie.data.datasets[0].data = [
      d.attack_types.DoS, d.attack_types.Probe,
      d.attack_types.R2L, d.attack_types.U2R, d.attack_types.Unknown
    ];
    pie.update('none');

    const fb = document.getElementById('feed-body');
    fb.innerHTML = d.recent.map(r => `
      <div class="feed-row">
        <span style="color:#8b949e">${r.id}</span>
        <span><span class="badge ${r.label === 'ATTACK' ? 'attack' : 'normal'}">${r.label}</span></span>
        <span style="color:${r.label==='ATTACK'?'#f85149':'#3fb950'}">${r.type}</span>
        <span style="color:#8b949e">${r.src}</span>
        <span style="color:#8b949e">${r.dst}</span>
        <span style="color:#e6edf3">${r.conf}%</span>
      </div>`).join('');
  });
}

setInterval(update, 500);
update();
</script>
</body>
</html>"""

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/api/state')
def api_state():
    return jsonify({
        'total'        : state['total'],
        'attacks'      : state['attacks'],
        'normal'       : state['normal'],
        'pps'          : state['pps'],
        'attack_types' : state['attack_types'],
        'recent'       : list(state['recent'])
    })

if __name__ == '__main__':
    print("Dashboard running at http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)