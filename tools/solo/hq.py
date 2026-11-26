#!/usr/bin/env python3
"""Field page: a top-down view of the field in the browser; click a point and the robot drives there.
    source tools/solo/env.sh; python3 tools/solo/hq.py <team> [--port 8700] [--bind 127.0.0.1] [--speed 0.6]
Then open http://localhost:8700 . Needs the vision connector running (/vision/poses/...).
    click            drive to that point, keeping the current heading
    click and drag   drive to the point and turn to face along the drag
    STOP / space / Esc   stop at once
The page is the dead man's switch: when it stops asking for the state for 1.5 s (tab closed, laptop asleep), the
target is dropped and the robot stops. Without a fresh pose from vision (0.2 s) the robot is not driven.
It refuses to drive while another program of ours that commands the robot is running (the strategy node, a test):
checked at start, on each click and about once a second while a target is active; it looks only at this computer's processes.
The target is dropped when our robot has not been seen for 2 s (it does not resume when vision sees it again).
POST requests must carry Content-Type: application/json, else 415 and nothing changes (no plain cross-site form posts).
A request whose Host header is not 127.0.0.1:<port>, localhost:<port> or the --bind address with the port gets 403.
Malformed JSON, and x, y, th, v that are not finite numbers, get 400 and nothing changes.
Speed: the slider (at most --max-speed, default 1.5 m/s); the commanded velocity is ramped at 1.5 m/s2.
"""
import json, math, os, subprocess, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Pose2D, Twist

args = sys.argv[1:]
def opt(name, default):
    if name in args:
        i = args.index(name); v = args[i + 1]; del args[i:i + 2]; return v
    return default
PORT, BIND = int(opt("--port", "8700")), opt("--bind", "127.0.0.1")
VMAX = float(opt("--max-speed", "1.5"))
SPEED0 = max(0.1, min(VMAX, float(opt("--speed", "0.6"))))
if not args or args[0] not in ("blue", "yellow"):
    sys.exit(__doc__)
TEAM = args[0]; IDX = int(os.environ.get("SOLO_ROBOT_INDEX", "0"))
LX, LY = 2.25, 1.5                                  # half field (m)
BOX = (-LX - 0.10, LX + 0.10, -LY - 0.10, LY + 0.10)   # targets are kept inside this
ACC = 1.5                                           # m/s2 ramp of the commanded velocity
wrap = lambda a: (a + math.pi) % (2 * math.pi) - math.pi

S = dict(us=None, us_t=0.0, ball=None, ball_t=0.0, others={}, target=None, speed=SPEED0, poll=0.0,
         note="", cmd=(0.0, 0.0, 0.0))
LOCK = threading.Lock()
now = time.monotonic

def others_running():
    """our other programs that command the robot (by process list; each publishes on the same topic)"""
    out = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout.splitlines()
    return [n for n in ("solo_node", "characterize.py", "series.py", "ram_series.py") if any(n in l and "hq.py" not in l for l in out)]

def num(v):
    """a finite JSON number (not a bool, not a string) as float, else None"""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    v = float(v)
    return v if math.isfinite(v) else None

def check_goto(b):
    """/goto body -> (ok, x, y, th or None); th None: keep the current heading"""
    if not isinstance(b, dict):
        return False, None, None, None
    x, y = num(b.get("x")), num(b.get("y"))
    th = num(b["th"]) if "th" in b else None
    if x is None or y is None or ("th" in b and th is None):
        return False, None, None, None
    return True, x, y, th

def check_speed(b):
    """/speed body -> (ok, v)"""
    v = num(b.get("v")) if isinstance(b, dict) else None
    return v is not None, v

def host_ok(host, port, bind):
    ok = {f"127.0.0.1:{port}", f"localhost:{port}"}
    if bind != "127.0.0.1":
        ok.add(f"{bind}:{port}")
    return (host or "").strip().lower() in ok

rclpy.init(); node = Node("solo_hq")
pub = node.create_publisher(Twist, os.environ.get("SOLO_CMD_TOPIC", "/cmd_vel"), qos_profile_sensor_data)
def on_us(m):
    with LOCK: S["us"], S["us_t"] = (m.x, m.y, m.theta), now()
def on_ball(m):
    with LOCK: S["ball"], S["ball_t"] = (m.x, m.y), now()
def on_other(key):
    def f(m):
        with LOCK: S["others"][key] = (m.x, m.y, m.theta, now())
    return f
node.create_subscription(Pose2D, f"/vision/poses/{TEAM}/robot{IDX}", on_us, 20)
node.create_subscription(Pose2D, "/vision/poses/ball", on_ball, 20)
for tm in ("blue", "yellow"):
    for i in range(6):
        if (tm, i) != (TEAM, IDX):
            node.create_subscription(Pose2D, f"/vision/poses/{tm}/robot{i}", on_other(f"{tm}{i}"), 20)

def send(vx, vy, w):
    m = Twist(); m.linear.x, m.linear.y, m.angular.z = float(vx), float(vy), float(w); pub.publish(m)
    S["cmd"] = (round(vx, 3), round(vy, 3), round(w, 3))

def control():
    """50 Hz: field-frame velocity toward the target, ramped, turned into the body frame"""
    vf = [0.0, 0.0]; quiet = 0; dt = 0.02; checked = now()
    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=dt)
        with LOCK:
            tgt, us, age, sp, poll = S["target"], S["us"], now() - S["us_t"], S["speed"], now() - S["poll"]
        if tgt is not None and now() - checked > 1.0:                # another commanding program started meanwhile
            checked = now(); busy = others_running()
            if busy:
                with LOCK: BUSY[0] = busy; S["target"] = tgt = None; S["note"] = "target dropped: " + ", ".join(busy) + " is running"
        if tgt is not None and poll > 1.5:
            with LOCK: S["target"] = tgt = None; S["note"] = "page stopped asking: target dropped"
        if tgt is not None and (us is None or age > 2.0):
            with LOCK: S["target"] = tgt = None; S["note"] = "target dropped: our robot not seen"
        if tgt is None or us is None or age > 0.2:
            vf = [0.0, 0.0]
            if tgt is not None or quiet < 25: send(0, 0, 0)          # zeros while blind, and for 0.5 s after a stop
            quiet = 0 if tgt is not None else quiet + 1
            continue
        quiet = 0
        x, y, th = us; tx, ty, tth = tgt
        d = math.hypot(tx - x, ty - y); eth = wrap(tth - th)
        if d < 0.03 and abs(eth) < 0.05:
            with LOCK: S["target"] = None; S["note"] = "arrived"
            continue
        v = min(sp, 1.8 * d + 0.03) * (0.3 if abs(eth) > 0.8 else 1.0)   # turn first when far off the heading
        want = ((tx - x) / d * v, (ty - y) / d * v) if d > 1e-3 else (0.0, 0.0)
        for k in (0, 1):                                              # ramp up, never delay slowing down
            if abs(want[k]) > abs(vf[k]) and want[k] * vf[k] >= 0: vf[k] += max(-ACC * dt, min(ACC * dt, want[k] - vf[k]))
            else: vf[k] = want[k]
        w = max(-1.5, min(1.5, 2.5 * eth))
        thp = th + w * 0.1                                            # the heading when the command acts
        c, s = math.cos(thp), math.sin(thp)
        send(c * vf[0] + s * vf[1], -s * vf[0] + c * vf[1], w)

PAGE = r"""<!doctype html><meta charset="utf-8"><title>field</title>
<style>body{margin:0;background:#111;color:#ddd;font:14px sans-serif}#bar{padding:6px 10px;display:flex;gap:14px;align-items:center}
#stop{background:#c22;color:#fff;border:0;font-size:20px;font-weight:bold;padding:8px 26px;border-radius:6px;cursor:pointer}
canvas{display:block;background:#0b3d1c;cursor:crosshair}#note{color:#fc6}</style>
<div id="bar"><button id="stop">STOP</button><label>speed <input id="sp" type="range" min="0.1" max="VMAXV" step="0.05"> <span id="spv"></span> m/s</label>
<span id="st"></span><span id="note"></span></div><canvas id="c"></canvas>
<script>
const LX=2.25,LY=1.5,M=0.6,c=document.getElementById('c'),g=c.getContext('2d');let st=null,drag=null,k=1;
function fit(){c.width=innerWidth;c.height=innerHeight-50;k=Math.min(c.width/(2*(LX+M)),c.height/(2*(LY+M)))}
const X=x=>c.width/2+x*k,Y=y=>c.height/2-y*k,FX=p=>(p-c.width/2)/k,FY=p=>(c.height/2-p)/k;
function post(u,b){fetch(u,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b||{})})}
function robot(x,y,th,col,lab){g.fillStyle=col;g.beginPath();g.arc(X(x),Y(y),0.09*k,0,7);g.fill();g.strokeStyle='#fff';g.lineWidth=2;g.beginPath();
 g.moveTo(X(x),Y(y));g.lineTo(X(x+0.16*Math.cos(th)),Y(y+0.16*Math.sin(th)));g.stroke();g.fillStyle='#fff';g.fillText(lab,X(x)+8,Y(y)-10)}
function draw(){fit();g.clearRect(0,0,c.width,c.height);g.strokeStyle='#fff';g.lineWidth=2;g.strokeRect(X(-LX),Y(LY),2*LX*k,2*LY*k);
 g.beginPath();g.moveTo(X(0),Y(LY));g.lineTo(X(0),Y(-LY));g.stroke();g.beginPath();g.arc(X(0),Y(0),0.5*k,0,7);g.stroke();
 for(const s of[-1,1]){g.strokeRect(X(s>0?LX-0.5:-LX),Y(0.675),0.5*k,1.35*k);g.strokeStyle='#c6f';g.strokeRect(X(s>0?LX:-LX-0.18),Y(0.4),0.18*k,0.8*k);g.strokeStyle='#fff'}
 g.fillStyle='#9c9';g.fillText('-x',X(-LX)+4,Y(-LY)-6);g.fillText('+x',X(LX)-20,Y(-LY)-6);g.fillText('+y',X(-LX)+4,Y(LY)+14);
 if(!st)return;for(const [n,o] of Object.entries(st.others))robot(o[0],o[1],o[2],n.startsWith('blue')?'#36c':'#cc3',n);
 if(st.ball){g.fillStyle='#f80';g.beginPath();g.arc(X(st.ball[0]),Y(st.ball[1]),0.035*k,0,7);g.fill()}
 if(st.target){const t=st.target;g.strokeStyle='#f44';g.lineWidth=2;g.beginPath();g.arc(X(t[0]),Y(t[1]),0.1*k,0,7);g.moveTo(X(t[0]),Y(t[1]));
  g.lineTo(X(t[0]+0.2*Math.cos(t[2])),Y(t[1]+0.2*Math.sin(t[2])));g.stroke()}
 if(st.us)robot(st.us[0],st.us[1],st.us[2],st.us_age<0.3?(st.team=='blue'?'#4af':'#ff4'):'#888','us');
 if(drag){g.strokeStyle='#f44';g.beginPath();g.moveTo(drag.x0,drag.y0);g.lineTo(drag.x1,drag.y1);g.stroke()}
 document.getElementById('st').textContent=(st.us?`us (${st.us[0].toFixed(2)}, ${st.us[1].toFixed(2)}) ${(st.us[2]*57.3).toFixed(0)} deg, seen ${st.us_age.toFixed(1)} s ago`:'our robot not seen')+
  (st.target?`  ->  (${st.target[0].toFixed(2)}, ${st.target[1].toFixed(2)})`:'')+`  cmd ${st.cmd}`+(st.busy.length?'  BLOCKED by '+st.busy:'');
 document.getElementById('note').textContent=st.note}
c.onmousedown=e=>{drag={x0:e.offsetX,y0:e.offsetY,x1:e.offsetX,y1:e.offsetY}};c.onmousemove=e=>{if(drag){drag.x1=e.offsetX;drag.y1=e.offsetY}};
c.onmouseup=e=>{if(!drag)return;const dx=drag.x1-drag.x0,dy=drag.y1-drag.y0,b={x:FX(drag.x0),y:FY(drag.y0)};if(Math.hypot(dx,dy)>15)b.th=Math.atan2(-dy,dx);post('/goto',b);drag=null};
document.getElementById('stop').onclick=()=>post('/stop');addEventListener('keydown',e=>{if(e.key==' '||e.key=='Escape'){e.preventDefault();post('/stop')}});
const sp=document.getElementById('sp');sp.oninput=()=>{post('/speed',{v:+sp.value});document.getElementById('spv').textContent=sp.value};
async function poll(){try{st=await(await fetch('/state')).json();if(document.activeElement!=sp){sp.value=st.speed;document.getElementById('spv').textContent=st.speed}}catch(e){st=null}draw()}
setInterval(poll,100);</script>""".replace("VMAXV", str(VMAX))

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def out(self, body, ctype="application/json", code=200):
        b = body.encode(); self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(b)
    def refuse(self, code, why):
        self.out(json.dumps(dict(error=why)), code=code)
    def do_GET(self):
        if not host_ok(self.headers.get("Host"), PORT, BIND):
            return self.refuse(403, "bad Host header")
        if self.path == "/state":
            with LOCK:
                S["poll"] = now(); t = now()
                d = dict(team=TEAM, us=S["us"], us_age=round(t - S["us_t"], 2) if S["us"] else None,
                         ball=S["ball"] if S["ball"] and t - S["ball_t"] < 1.0 else None,
                         others={k: v[:3] for k, v in S["others"].items() if t - v[3] < 1.0},
                         target=S["target"], speed=S["speed"], note=S["note"], cmd=S["cmd"], busy=BUSY[0])
            self.out(json.dumps(d))
        else:
            self.out(PAGE, "text/html; charset=utf-8")
    def do_POST(self):
        if not host_ok(self.headers.get("Host"), PORT, BIND):
            return self.refuse(403, "bad Host header")
        if self.headers.get("Content-Type", "").split(";")[0].strip().lower() != "application/json":
            return self.refuse(415, "Content-Type must be application/json")
        if self.path not in ("/stop", "/speed", "/goto"):
            return self.refuse(404, "no such request")
        try:
            n = int(self.headers.get("Content-Length", 0))
            if not 0 <= n <= 4096: raise ValueError("body size")
            b = json.loads(self.rfile.read(n) or b"{}")
        except (ValueError, UnicodeDecodeError):
            return self.refuse(400, "malformed JSON")
        if self.path == "/speed":
            ok, v = check_speed(b)
            if not ok: return self.refuse(400, "v must be a finite number")
            with LOCK: S["speed"] = max(0.1, min(VMAX, v))
        elif self.path == "/goto":
            ok, gx, gy, gth = check_goto(b)
            if not ok: return self.refuse(400, "x, y (and th if given) must be finite numbers")
            busy = others_running()
            with LOCK:
                BUSY[0] = busy
                if busy: S["note"] = "not driving: " + ", ".join(busy) + " is running"
                elif S["us"] is None or now() - S["us_t"] > 0.5: S["note"] = "not driving: our robot is not seen"
                else:
                    x = max(BOX[0], min(BOX[1], gx)); y = max(BOX[2], min(BOX[3], gy))
                    S["target"] = (x, y, gth if gth is not None else S["us"][2]); S["note"] = ""
        else:
            with LOCK: S["target"] = None; S["note"] = "stopped"
        self.out("{}")

BUSY = [others_running()]
if BUSY[0]: print("note: running now:", ", ".join(BUSY[0]), "(driving is refused while they run)", flush=True)
srv = ThreadingHTTPServer((BIND, PORT), H)
threading.Thread(target=srv.serve_forever, daemon=True).start()
print(f"field page on http://{BIND}:{PORT}  team {TEAM} robot {IDX}", flush=True)
try:
    control()
except KeyboardInterrupt:
    pass
finally:
    for _ in range(10): send(0, 0, 0); time.sleep(0.02)
