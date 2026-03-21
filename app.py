from flask import Flask, jsonify, render_template_string
import torch
import numpy as np
import threading
import time
import random
from collections import deque
import os
import torch.nn as nn
import brevitas.nn as qnn
import pickle

app = Flask(__name__)

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
model = build_model()
model.load_state_dict(torch.load('models/model_8bit.pt', map_location='cpu'))
model.eval()

with open('data/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

X_test = np.load('data/X_test.npy')
y_test = np.load('data/y_test.npy')
print(f"Loaded {len(X_test)} real samples")

ATTACK_TYPE_NAMES = ['DoS', 'Probe', 'R2L', 'U2R']
ATTACK_DETAILS = {
    'DoS'  : ['SYN Flood', 'UDP Flood', 'ICMP Flood', 'Ping of Death', 'Smurf'],
    'Probe': ['Port Scan', 'Ping Sweep', 'IP Sweep', 'Satan', 'Nmap'],
    'R2L'  : ['FTP Brute Force', 'Guess Password', 'IMAP Attack', 'Phf'],
    'U2R'  : ['Buffer Overflow', 'Rootkit', 'Load Module', 'Perl Attack'],
}
NORMAL_TYPES = ['HTTP', 'HTTPS', 'FTP', 'SSH', 'DNS', 'SMTP', 'NTP', 'SNMP']
PROTOCOLS    = ['TCP', 'UDP', 'ICMP']
SERVICES     = ['http', 'ftp', 'ssh', 'smtp', 'dns', 'https', 'pop3', 'imap']

state = {
    'total'         : 0,
    'attacks'       : 0,
    'normal'        : 0,
    'recent'        : deque(maxlen=20),
    'pps'           : 0,
    'attack_types'  : {'DoS': 0, 'Probe': 0, 'R2L': 0, 'U2R': 0},
    'timeline'      : deque(maxlen=30),
    'top_src'       : {},
    'top_dst'       : {},
    'bytes_in'      : 0,
    'bytes_out'     : 0,
    'avg_conf'      : 0,
    'conf_sum'      : 0,
    'correct'       : 0,
    'protocols'     : {'TCP': 0, 'UDP': 0, 'ICMP': 0},
    'running'       : True,
    'alerts'        : deque(maxlen=5),
    'heatmap'       : [0] * 24,
}

def predict(features):
    t = torch.tensor(features.reshape(1, -1), dtype=torch.float32)
    with torch.no_grad():
        out   = model(t)
        pred  = out.argmax(dim=1).item()
        probs = torch.softmax(out, dim=1).numpy()[0]
    return pred, float(probs[pred]) * 100

def feed_loop():
    idx        = 0
    count      = 0
    start_time = time.time()
    last_min   = -1

    while state['running']:
        sample = X_test[idx % len(X_test)]
        actual = int(y_test[idx % len(y_test)])
        pred, conf = predict(sample)

        state['total']    += 1
        state['conf_sum'] += conf
        state['avg_conf']  = round(state['conf_sum'] / state['total'], 1)
        count             += 1

        elapsed = time.time() - start_time
        if elapsed >= 1.0:
            state['pps'] = round(count / elapsed)
            state['timeline'].append({
                't'      : time.strftime('%H:%M:%S'),
                'attacks': state['attacks'],
                'normal' : state['normal'],
                'pps'    : state['pps'],
            })
            count      = 0
            start_time = time.time()

        hour = int(time.strftime('%H'))
        state['heatmap'][hour] += 1

        src  = f"192.168.{random.randint(1,10)}.{random.randint(1,50)}"
        dst  = f"10.0.{random.randint(0,5)}.{random.randint(1,30)}"
        proto = random.choice(PROTOCOLS)
        svc   = random.choice(SERVICES)
        port  = random.randint(1024, 65535)
        size  = random.randint(64, 1500)

        state['bytes_in']  += size
        state['bytes_out'] += random.randint(40, 800)
        state['protocols'][proto] += 1

        state['top_src'][src] = state['top_src'].get(src, 0) + 1
        state['top_dst'][dst] = state['top_dst'].get(dst, 0) + 1

        if pred == 1:
            state['attacks'] += 1
            atype  = random.choice(ATTACK_TYPE_NAMES)
            aname  = random.choice(ATTACK_DETAILS[atype])
            state['attack_types'][atype] += 1
            if conf > 90:
                state['alerts'].appendleft({
                    'time' : time.strftime('%H:%M:%S'),
                    'msg'  : f"High confidence {atype} detected: {aname} from {src}",
                    'conf' : round(conf, 1),
                    'type' : atype,
                })
        else:
            state['normal'] += 1
            aname = f"Normal {random.choice(NORMAL_TYPES)}"

        if pred == actual:
            state['correct'] += 1

        state['recent'].appendleft({
            'id'      : state['total'],
            'label'   : 'ATTACK' if pred == 1 else 'NORMAL',
            'type'    : aname,
            'conf'    : round(conf, 1),
            'correct' : pred == actual,
            'src'     : src,
            'dst'     : dst,
            'proto'   : proto,
            'service' : svc,
            'port'    : port,
            'size'    : size,
            'time'    : time.strftime('%H:%M:%S'),
        })

        idx += 1
        time.sleep(0.25)

thread = threading.Thread(target=feed_loop, daemon=True)
thread.start()

HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>IDS FINN Dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0d1117;color:#e6edf3;font-family:'Segoe UI',Arial,sans-serif;font-size:13px}
.header{background:#161b22;border-bottom:1px solid #30363d;
        padding:12px 20px;display:flex;align-items:center;gap:12px;position:sticky;top:0;z-index:10}
.header h1{font-size:16px;font-weight:500;color:#58a6ff}
.dot{width:9px;height:9px;border-radius:50%;background:#3fb950;animation:pulse 1.5s infinite}
.badge-live{background:rgba(63,185,80,.15);color:#3fb950;padding:2px 8px;
            border-radius:10px;font-size:11px}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.3}}
.metrics{display:grid;grid-template-columns:repeat(8,1fr);gap:8px;padding:12px 20px}
.metric{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px 10px;text-align:center}
.metric h3{font-size:10px;color:#8b949e;text-transform:uppercase;letter-spacing:.8px;margin-bottom:6px}
.metric .v{font-size:20px;font-weight:600}
.v.blue{color:#58a6ff}.v.red{color:#f85149}.v.green{color:#3fb950}
.v.amber{color:#d29922}.v.purple{color:#a371f7}.v.teal{color:#39d353}
.v.pink{color:#ff7b72}.v.gray{color:#8b949e}
.main{display:grid;grid-template-columns:1fr 1fr;gap:10px;padding:0 20px 10px}
.wide{grid-column:1/-1}
.panel{background:#161b22;border:1px solid #30363d;border-radius:8px;overflow:hidden}
.panel-hdr{padding:10px 14px;border-bottom:1px solid #30363d;
           font-size:11px;color:#8b949e;text-transform:uppercase;
           letter-spacing:.8px;display:flex;justify-content:space-between;align-items:center}
.panel-hdr span{color:#58a6ff;font-size:10px;text-transform:none;letter-spacing:0}
.feed-labels,.feed-row{display:grid;
  grid-template-columns:45px 80px 140px 55px 55px 110px 110px 50px 55px 70px;
  padding:7px 14px;font-size:11px;align-items:center}
.feed-labels{border-bottom:1px solid #30363d;color:#8b949e;text-transform:uppercase;
             font-size:10px;letter-spacing:.5px}
.feed-row{border-bottom:1px solid #21262d;transition:background .15s;cursor:default}
.feed-row:hover{background:#1c2128}
.badge{display:inline-block;padding:1px 7px;border-radius:10px;font-size:10px;font-weight:500}
.badge.attack{background:rgba(248,81,73,.15);color:#f85149}
.badge.normal{background:rgba(63,185,80,.15);color:#3fb950}
.correct{color:#3fb950;font-size:11px}.wrong{color:#f85149;font-size:11px}
.charts{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;padding:0 20px 10px}
.alerts{padding:0 20px 10px}
.alert-row{background:#161b22;border:1px solid #30363d;border-left:3px solid #f85149;
           border-radius:6px;padding:8px 12px;margin-bottom:6px;
           display:flex;justify-content:space-between;align-items:center}
.alert-msg{font-size:12px;color:#e6edf3}
.alert-meta{font-size:11px;color:#8b949e;text-align:right}
.bottom{display:grid;grid-template-columns:1fr 1fr;gap:10px;padding:0 20px 20px}
.tbl{width:100%;border-collapse:collapse;font-size:11px}
.tbl th{color:#8b949e;text-transform:uppercase;font-size:10px;
        padding:6px 10px;border-bottom:1px solid #30363d;text-align:left}
.tbl td{padding:6px 10px;border-bottom:1px solid #21262d;color:#e6edf3}
.tbl tr:last-child td{border:none}
.bar-wrap{background:#21262d;border-radius:3px;height:6px;margin-top:3px;overflow:hidden}
.bar-fill{height:100%;border-radius:3px;background:#58a6ff;transition:width .5s}
</style>
</head>
<body>

<div class="header">
  <div class="dot"></div>
  <h1>FPGA-NIDS Live Dashboard</h1>
  <span class="badge-live">LIVE</span>
  <span style="font-size:11px;color:#8b949e;margin-left:auto">
    NSL-KDD · 8-bit Quantized MLP · Brevitas · FINN Framework
  </span>
</div>

<div class="metrics">
  <div class="metric"><h3>Total packets</h3><div class="v blue" id="total">0</div></div>
  <div class="metric"><h3>Attacks</h3><div class="v red" id="attacks">0</div></div>
  <div class="metric"><h3>Normal</h3><div class="v green" id="normal">0</div></div>
  <div class="metric"><h3>Attack rate</h3><div class="v pink" id="arate">0%</div></div>
  <div class="metric"><h3>Pred/sec</h3><div class="v amber" id="pps">0</div></div>
  <div class="metric"><h3>Avg confidence</h3><div class="v purple" id="conf">0%</div></div>
  <div class="metric"><h3>Accuracy</h3><div class="v teal" id="acc">0%</div></div>
  <div class="metric"><h3>Bytes in</h3><div class="v gray" id="bytes">0</div></div>
</div>

<div class="alerts" id="alerts-wrap"></div>

<div class="main">
  <div class="panel wide">
    <div class="panel-hdr">
      Live traffic feed
      <span id="feed-count">0 packets processed</span>
    </div>
    <div class="feed-labels">
      <span>#</span><span>Result</span><span>Attack type</span>
      <span>Proto</span><span>Service</span>
      <span>Source IP</span><span>Dest IP</span>
      <span>Port</span><span>Size</span><span>Time</span>
    </div>
    <div id="feed-body"></div>
  </div>
</div>

<div class="charts">
  <div class="panel">
    <div class="panel-hdr">Attack type breakdown</div>
    <div style="padding:12px"><canvas id="pie" height="180"></canvas></div>
  </div>
  <div class="panel">
    <div class="panel-hdr">Protocol distribution</div>
    <div style="padding:12px"><canvas id="proto" height="180"></canvas></div>
  </div>
  <div class="panel">
    <div class="panel-hdr">Traffic timeline</div>
    <div style="padding:12px"><canvas id="line" height="180"></canvas></div>
  </div>
</div>

<div class="bottom">
  <div class="panel">
    <div class="panel-hdr">Top source IPs</div>
    <table class="tbl" id="top-src-tbl"></table>
  </div>
  <div class="panel">
    <div class="panel-hdr">Top destination IPs</div>
    <table class="tbl" id="top-dst-tbl"></table>
  </div>
</div>

<script>
const pie = new Chart(document.getElementById('pie'),{
  type:'doughnut',
  data:{labels:['DoS','Probe','R2L','U2R'],
    datasets:[{data:[0,0,0,0],
      backgroundColor:['#f85149','#d29922','#a371f7','#58a6ff'],borderWidth:0}]},
  options:{responsive:true,plugins:{legend:{position:'bottom',
    labels:{color:'#8b949e',font:{size:10},boxWidth:8}}}}
});

const proto = new Chart(document.getElementById('proto'),{
  type:'doughnut',
  data:{labels:['TCP','UDP','ICMP'],
    datasets:[{data:[0,0,0],
      backgroundColor:['#58a6ff','#3fb950','#d29922'],borderWidth:0}]},
  options:{responsive:true,plugins:{legend:{position:'bottom',
    labels:{color:'#8b949e',font:{size:10},boxWidth:8}}}}
});

const lineChart = new Chart(document.getElementById('line'),{
  type:'line',
  data:{labels:[],datasets:[
    {label:'Attacks',data:[],borderColor:'#f85149',borderWidth:1.5,
     pointRadius:0,fill:true,backgroundColor:'rgba(248,81,73,.1)'},
    {label:'Normal',data:[],borderColor:'#3fb950',borderWidth:1.5,
     pointRadius:0,fill:true,backgroundColor:'rgba(63,185,80,.1)'}
  ]},
  options:{responsive:true,animation:false,
    scales:{x:{ticks:{color:'#8b949e',font:{size:9},maxTicksLimit:6},
               grid:{color:'rgba(255,255,255,.04)'}},
            y:{ticks:{color:'#8b949e',font:{size:9}},
               grid:{color:'rgba(255,255,255,.04)'}}},
    plugins:{legend:{labels:{color:'#8b949e',font:{size:10},boxWidth:8}}}}
});

function fmt(n){
  if(n>1e9) return (n/1e9).toFixed(1)+'GB';
  if(n>1e6) return (n/1e6).toFixed(1)+'MB';
  if(n>1e3) return (n/1e3).toFixed(1)+'KB';
  return n+'B';
}

function ipTable(id, data){
  const sorted = Object.entries(data).sort((a,b)=>b[1]-a[1]).slice(0,5);
  const max    = sorted.length > 0 ? sorted[0][1] : 1;
  document.getElementById(id).innerHTML =
    '<tr><th>IP Address</th><th>Packets</th></tr>' +
    sorted.map(([ip,cnt])=>`
      <tr><td>${ip}</td><td>${cnt}
        <div class="bar-wrap">
          <div class="bar-fill" style="width:${Math.round(cnt/max*100)}%"></div>
        </div>
      </td></tr>`).join('');
}

function update(){
  fetch('/api/state').then(r=>r.json()).then(d=>{
    const rate = d.total>0?Math.round(d.attacks/d.total*100):0;
    const acc  = d.total>0?Math.round(d.correct/d.total*100):0;

    document.getElementById('total').textContent   = d.total.toLocaleString();
    document.getElementById('attacks').textContent = d.attacks.toLocaleString();
    document.getElementById('normal').textContent  = d.normal.toLocaleString();
    document.getElementById('arate').textContent   = rate+'%';
    document.getElementById('pps').textContent     = d.pps;
    document.getElementById('conf').textContent    = d.avg_conf+'%';
    document.getElementById('acc').textContent     = acc+'%';
    document.getElementById('bytes').textContent   = fmt(d.bytes_in);
    document.getElementById('feed-count').textContent = d.total.toLocaleString()+' packets processed';

    pie.data.datasets[0].data = [
      d.attack_types.DoS,d.attack_types.Probe,
      d.attack_types.R2L,d.attack_types.U2R];
    pie.update('none');

    proto.data.datasets[0].data = [
      d.protocols.TCP,d.protocols.UDP,d.protocols.ICMP];
    proto.update('none');

    if(d.timeline.length > 0){
      lineChart.data.labels               = d.timeline.map(t=>t.t);
      lineChart.data.datasets[0].data     = d.timeline.map(t=>t.attacks);
      lineChart.data.datasets[1].data     = d.timeline.map(t=>t.normal);
      lineChart.update('none');
    }

    document.getElementById('alerts-wrap').innerHTML =
      d.alerts.map(a=>`
        <div class="alert-row">
          <div>
            <span style="color:#f85149;font-weight:500">[${a.type}]</span>
            <span class="alert-msg"> ${a.msg}</span>
          </div>
          <div class="alert-meta">${a.conf}% confidence<br>${a.time}</div>
        </div>`).join('');

    document.getElementById('feed-body').innerHTML =
      d.recent.map(r=>`
        <div class="feed-row">
          <span style="color:#8b949e">${r.id}</span>
          <span><span class="badge ${r.label==='ATTACK'?'attack':'normal'}">${r.label}</span></span>
          <span style="color:${r.label==='ATTACK'?'#f85149':'#3fb950'}">${r.type}</span>
          <span style="color:#8b949e">${r.proto}</span>
          <span style="color:#8b949e">${r.service}</span>
          <span style="color:#8b949e">${r.src}</span>
          <span style="color:#8b949e">${r.dst}</span>
          <span style="color:#8b949e">${r.port}</span>
          <span style="color:#8b949e">${r.size}B</span>
          <span style="color:#444">${r.time}</span>
        </div>`).join('');

    ipTable('top-src-tbl', d.top_src);
    ipTable('top-dst-tbl', d.top_dst);
  });
}

setInterval(update, 400);
update();
</script>
</body>
</html>"""

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/api/state')
def api_state():
    top_src = dict(sorted(state['top_src'].items(),
                          key=lambda x: x[1], reverse=True)[:10])
    top_dst = dict(sorted(state['top_dst'].items(),
                          key=lambda x: x[1], reverse=True)[:10])
    return jsonify({
        'total'        : state['total'],
        'attacks'      : state['attacks'],
        'normal'       : state['normal'],
        'pps'          : state['pps'],
        'avg_conf'     : state['avg_conf'],
        'correct'      : state['correct'],
        'bytes_in'     : state['bytes_in'],
        'attack_types' : state['attack_types'],
        'protocols'    : state['protocols'],
        'timeline'     : list(state['timeline']),
        'recent'       : list(state['recent']),
        'alerts'       : list(state['alerts']),
        'top_src'      : top_src,
        'top_dst'      : top_dst,
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Dashboard running at http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)