"""Which of our programs command the robot, and the strategy node's command line: one place for hq.py,
auto_session.py and place_ball.py (standard library only, Linux /proc).
    COMMANDERS            every tool here that publishes velocity commands on the command topic, and the strategy node.
                          hq.py publishes too but is not listed: it is the field page the others report to (its own
                          driving is in its /state), and it refuses to drive while any of these runs.
    is_instance(argv, n)  argv is a running instance of n: the executable itself, or a python whose script is n
                          (interpreter options such as -u, -B, -X dev are skipped; python -c / -m runs no script)
    is_launcher(argv)     `ros2 run solo_strategy solo_node` (the launcher around the node; it does not publish)
    process_table()       [(pid, argv)] of this computer's processes
    ancestors(pid)        that process and all its ancestors
    running(...)          the commanding programs that run now, outside a set of excluded pids
    check_node_cmd(argv)  "" when argv is exactly the strategy node's own command line, else why not
"""
import os, re

COMMANDERS = ("solo_node", "characterize.py", "series.py", "ram_series.py", "place_ball.py", "goto.py", "nudge.py",
              "frame_check.py")

# the parameters solo_strategy/solo_node.py declares (tests/test_procs.py compares this with the node's source)
NODE_PARAMS = frozenset((
    "team", "robot_index", "cmd_topic", "rate_hz", "pose_timeout", "latency", "predict", "blind_slow", "blind_limit",
    "blind_floor", "pred_after", "pred_delay", "pred_accel", "pred_vel_mix", "opp_hold", "ball_coast", "pred_policy_vel",
    "speed_cap", "angular_cap", "ramp_fwd", "ramp_side", "side", "force_state", "server_port", "cmd_reliable",
    "vision_prefix", "foe_all", "recover", "recover_speed", "recover_accel", "recover_time", "recover_dist",
    "recover_frames", "recover_stop_speed", "recover_stale", "recover_gyro", "recover_map", "recover_map_module",
    "recover_min_rate", "recover_min_evidence", "odom_latency", "odom_wheel_lag", "wheels_topic", "imu_topic",
    "recover_states", "recover_source", "odom_topic", "odom_robot_lag"))

PYTHON = re.compile(r"python3(\.\d+)?")
PARAM = re.compile(r"([A-Za-z_][A-Za-z0-9_]*):=(.*)", re.S)


def is_python(path):
    return os.path.basename(path).startswith("python")


def script_of(argv):
    """the script a python interpreter runs (its path), or None (no script: -c, -m, stdin, no argument)"""
    rest = list(argv[1:])
    while rest:
        a = rest.pop(0)
        if a == "--": return rest[0] if rest else None
        if a == "-" or not a.startswith("-"): return None if a == "-" else a
        if a.startswith("--"):
            if a == "--check-hash-based-pycs" and rest: rest.pop(0)
            continue
        for i, ch in enumerate(a[1:], 1):            # a cluster of one-letter options, e.g. -u, -uB, -Xdev, -X dev
            if ch in "cm": return None
            if ch in "XW":
                if i == len(a) - 1 and rest: rest.pop(0)
                break
    return None


def is_instance(argv, name):
    """argv is a running instance of the program `name`: the executable itself, or a python whose script is it
    (a command line that merely mentions the name, an editor or a pager on it, python -c ... is not)"""
    if not argv: return False
    if os.path.basename(argv[0]) == name: return True
    if not is_python(argv[0]): return False
    s = script_of(argv)
    return s is not None and os.path.basename(s) == name


def is_launcher(argv):
    """`ros2 run solo_strategy solo_node` (the launcher around the node; it does not publish itself)"""
    return (len(argv) > 4 and is_python(argv[0]) and os.path.basename(argv[1]) == "ros2"
            and argv[2:5] == ["run", "solo_strategy", "solo_node"])


def process_table():
    """[(pid, argv)] from /proc"""
    out = []
    for d in os.listdir("/proc"):
        if not d.isdigit(): continue
        try:
            with open(f"/proc/{d}/cmdline", "rb") as f: argv = [a.decode(errors="replace") for a in f.read().split(b"\0")[:-1]]
        except OSError: continue
        if argv: out.append((int(d), argv))
    return out


def parent(pid):
    with open(f"/proc/{pid}/stat") as f: return int(f.read().rsplit(")", 1)[1].split()[1])


def ancestors(pid=None):
    """this process (or pid) and all its ancestors: wrapper scripts, shells, a terminal"""
    mine = set(); pid = os.getpid() if pid is None else pid
    while pid > 1 and pid not in mine:
        mine.add(pid)
        try: pid = parent(pid)
        except (OSError, ValueError, IndexError): break
    return mine


def running(names=COMMANDERS, table=None, exclude=None):
    """the names in `names` that run now as a process outside `exclude` (default: this process and its ancestors)"""
    table = process_table() if table is None else table
    exclude = ancestors() if exclude is None else exclude
    return sorted({n for pid, a in table if pid not in exclude for n in names if is_instance(a, n)})


def check_node_cmd(argv):
    """"" when argv is exactly [<python>, <node>, "--ros-args", ("-p", "<name>:=<value>")*]: <python> an absolute path
    of an existing executable named python3 or python3.N, <node> an absolute path of an existing regular file named
    solo_node, every name a parameter the node declares, no element beginning with -c or holding a newline or a NUL.
    Else the reason it is refused."""
    if not isinstance(argv, list): return "not a list"
    if len(argv) < 3: return "too short"
    if not all(isinstance(a, str) for a in argv): return "not all strings"
    for a in argv:
        if a.startswith("-c"): return f"an element begins with -c: {a[:40]!r}"
        if "\n" in a or "\r" in a or "\0" in a: return "an element holds a newline or a NUL"
    py, node = argv[0], argv[1]
    if not os.path.isabs(py): return f"interpreter not an absolute path: {py[:80]!r}"
    if not PYTHON.fullmatch(os.path.basename(py)): return f"interpreter not python3: {py[:80]!r}"
    if not (os.path.isfile(py) and os.access(py, os.X_OK)): return f"interpreter not an executable file: {py[:80]!r}"
    if not os.path.isabs(node): return f"node not an absolute path: {node[:80]!r}"
    if os.path.basename(node) != "solo_node": return f"script not named solo_node: {node[:80]!r}"
    if not os.path.isfile(node): return f"node not a regular file: {node[:80]!r}"
    if argv[2] != "--ros-args": return "the third element is not --ros-args"
    rest = argv[3:]
    if len(rest) % 2: return "parameters not in -p name:=value pairs"
    for i in range(0, len(rest), 2):
        if rest[i] != "-p": return f"not a -p parameter: {rest[i][:40]!r}"
        m = PARAM.fullmatch(rest[i + 1])
        if not m: return f"not name:=value: {rest[i + 1][:40]!r}"
        if m.group(1) not in NODE_PARAMS: return f"unknown parameter: {m.group(1)[:40]!r}"
    return ""
