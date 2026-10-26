"""
========================================
-PSEUDOCODE (plain language)
========================================

Think of this program like this:

1) Check the machine:
   - Which Python version?
   - Is an NVIDIA GPU (CUDA) available?
   - Which operating system?

2) Retrieve a list from the PyTorch site:
   - different CUDA versions (cu121, cu118, etc.)
   - CPU option

3) Go through each option:
   - retrieve torch, torchaudio, torchcodec, and torchvision wheels
   - filter compatible wheels (Python + platform)

4) Try to match:
   - wheels from the same build date (devYYYYMMDD)
   - so that all required libraries are available

5) Select the best option:
   - prefer the user's CUDA target
   - otherwise use a fallback

6) Print:
   ✅ what was found
   ✅ the best matching set
   ✅ a ready-to-run pip install command

7) UI:
   - "Run" button
   - "Stop" button
   - checkboxes
   - target override dropdown
   - log window

========================================
"""

import tkinter as tk
from tkinter import ttk
import sys, requests, re, platform, subprocess, threading
from urllib.parse import unquote
from collections import defaultdict
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL = "https://download.pytorch.org/whl/nightly/"
stop_flag = False  # 🔴 used to cancel processing from the UI


# =============================
# DETECT (environment detection)
# =============================
def detect_cuda():
    """
    :
    Try to determine whether the machine has a GPU and CUDA available.
    1) Ask torch
    2) If that fails, use nvidia-smi
    """

    try:
        import torch
        if torch.cuda.is_available():
            return torch.version.cuda
    except:
        pass

    try:
        # fallback: nvidia-smi
        r = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        m = re.search(r"CUDA Version:\s+(\d+\.\d+)", r.stdout)
        if m:
            return m.group(1)
    except:
        pass

    return "No CUDA"


def cuda_to_target(cuda):
    """
    :
    Convert the CUDA version to this format:
    12.1 → cu121
    """
    if "No CUDA" in cuda:
        return "cpu"

    try:
        major, minor = cuda.split(".")
        return f"cu{major}{minor}"
    except:
        return "cpu"


def detect_platform():
    """
    :
    Detect the operating system for wheel filtering
    """
    if platform.system() == "Windows":
        return "win_amd64"
    elif platform.system() == "Linux":
        return "manylinux"
    return "macosx"


# =============================
# ENV (environment information)
# =============================
def get_env():
    """
    Collect all important information in one place
    """

    def safe(p):
        # Check whether the library is installed
        try:
            return __import__(p).__version__
        except:
            return "Not installed"

    cuda = detect_cuda()

    return {
        "Python": sys.version.split()[0],
        "CUDA": cuda,
        "Target": cuda_to_target(cuda),
        "Platform": detect_platform(),
        "torch": safe("torch"),
        "torchaudio": safe("torchaudio"),
        "torchcodec": safe("torchcodec"),
        "torchvision": safe("torchvision")
    }

# =============================
# TARGETS
# =============================
def get_targets():
    """
    :
    Retrieve all available CUDA builds from the internet
    """

    try:
        r = requests.get(BASE_URL, timeout=10)
        dirs = re.findall(r'href="(cu\d+/)"', r.text)
        dirs = [d.strip("/") for d in dirs]
    except:
        dirs = []

    targets = ["cpu"] + dirs

    # Sort so that CPU is last

    return sorted(
        targets,
        key=lambda x: (x == "cpu", int(x.replace("cu","")) if x != "cpu" else -1),
        reverse=True
    )

# =============================
# FETCH (data retrieval)
# =============================

def fetch_wheels(url):
    try:
        r = requests.get(url, timeout=10)
        links = re.findall(r'href="([^"]+\.whl[^"]*)"', r.text)

        fixed = []
        for l in links:
            l = unquote(l)

            # ✅ Fix: if the path is relative, add the domain
            if l.startswith("/"):
                l = "https://download-r2.pytorch.org" + l

            fixed.append(l)

        return fixed

    except:
        return []

def is_regular_python_wheel(wheel_url, py_tag):
    """Accept only standard CPython builds, not free-threaded cpXXXt wheels."""
    filename = unquote(wheel_url).rsplit("/", 1)[-1]

    # For example, cp314t means a Python 3.14 free-threaded build.
    if re.search(r"(?:^|[-_.])cp\d+t(?:[-_.]|$)", filename, re.IGNORECASE):
        return False

    # The Python tag must match a standard interpreter, e.g. cp314, not cp314t.
    return re.search(
        rf"(?:^|[-_.]){re.escape(py_tag)}(?:[-_.]|$)",
        filename,
        re.IGNORECASE,
    ) is not None


def fetch_target(target, libs, show_all, plat, py_tag):
    global stop_flag

    result = {}
    logs = []   # 🔥 collect logs here

    if stop_flag:
        return target, {}, logs

    logs.append(f"→ {target}")

    for lib in libs:

        if stop_flag:
            return target, {}, logs

        url = f"{BASE_URL}{target}/{lib}/"
        wheels = fetch_wheels(url)

        wheels = [w for w in wheels if plat in w and is_regular_python_wheel(w, py_tag)]

        logs.append(f"   {lib}: {len(wheels)}")

        limit = None if show_all else 5

        for w in wheels[:limit]:
            logs.append(f"      {w}")

        result[lib] = wheels

    return target, result, logs


def fetch_all(progress, show_all, plat, py_tag, progress_update=None):
    global stop_flag

    libs = ["torch", "torchaudio", "torchcodec", "torchvision"]
    targets = get_targets()
    result = {}
    max_workers = os.cpu_count() or 4

    if progress_update:
        progress_update(20, "Scanning available PyTorch targets...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(fetch_target, t, libs, show_all, plat, py_tag): t
            for t in targets
        }
        completed = {}
        total = max(len(futures), 1)

        for completed_count, future in enumerate(as_completed(futures), start=1):
            if stop_flag:
                for pending in futures:
                    pending.cancel()
                break

            target, data, logs = future.result()
            completed[target] = (data, logs)

            if progress_update:
                value = 20 + (completed_count / total) * 55
                progress_update(value, f"Scanning target {completed_count}/{total}: {target}")

        # Print results in the original target order.
        for t in targets:
            if t in completed:
                data, logs = completed[t]
                result[t] = data
                for line in logs:
                    progress(line)

    return result

# =============================
# MATCH (matching)
# =============================
def get_date(w):
    """
    :
    Extract the build date (devYYYYMMDD) from the wheel name
    """
    m = re.search(r'dev(\d{8})', w)
    return m.group(1) if m else None


def match(all_data):
    """
    :
    Try to find matching builds for all libraries
    """

    results = []

    for target, libs in all_data.items():
        builds = defaultdict(dict)

        for lib, wheels in libs.items():
            for w in wheels:
                d = get_date(w)
                if d:
                    builds[d][lib] = w

        for d, lm in builds.items():
            results.append((len(lm), d, target, lm))

    # Best match first (score descending)
    return sorted(results, reverse=True)


def pick_best(matches, preferred, only_full):
    """
    :
    Select the best match:
    - preferably the same CUDA target
    - preferably all four libraries
    """

    if only_full:
        matches = [m for m in matches if m[0] == 4]

    for m in matches:
        if m[2] == preferred:
            return m

    return matches[0] if matches else None


# =============================
# MAIN (core logic)
# =============================
def run(show_all, only_full, override_target):
    """Run the resolver and report each major stage in the progress bar."""
    global stop_flag
    stop_flag = False

    def log(msg):
        root.after(0, append_log, msg)

    def stage(value, message):
        root.after(0, set_progress, value, message)

    root.after(0, clear_output)
    stage(5, "Detecting the local environment...")

    env = get_env()
    py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"

    log("=== Installed versions ===")
    for k, v in env.items():
        log(f"{k}: {v}")

    if stop_flag:
        stage(0, "Stopped by user")
        root.after(0, finish_run)
        return

    stage(15, "Retrieving available builds...")
    log("\n[Retrieving data...]")

    data = fetch_all(
        log,
        show_all,
        env["Platform"],
        py_tag,
        stage,
    )

    if stop_flag:
        log("\nSearch stopped by user.")
        stage(0, "Stopped by user")
        root.after(0, finish_run)
        return

    stage(80, "Creating the compatibility summary...")
    log("\n=== SUMMARY ===")
    for t, libs in data.items():
        ok = sum(1 for v in libs.values() if v)
        log(f"{t}: {ok}/4")

    stage(88, "Matching builds by date...")
    matches = match(data)

    stage(94, "Selecting the best compatible set...")
    selected_target = override_target or env["Target"]
    best = pick_best(matches, selected_target, only_full)

    if not best:
        log("\n❌ No compatible set found")
        stage(100, "Finished: no compatible set found")
        root.after(0, finish_run)
        return

    score, date, target, lm = best
    log("\n=== BEST MATCH ===")
    log(f"Target: {target}")
    log(f"Build: {date}")
    log(f"Score: {score}/4\n")

    for lib in ["torch", "torchaudio", "torchcodec", "torchvision"]:
        log(f"{lib}: {'✅' if lib in lm else '❌'}")

    stage(98, "Creating the installation command...")
    log("\n=== INSTALL ===")
    order = ["torch", "torchvision", "torchaudio", "torchcodec"]
    wheels_ordered = [lm[l] for l in order if l in lm]
    log("pip install " + " ".join(f'"{w}"' for w in wheels_ordered))

    stage(100, "Complete")
    root.after(0, finish_run)

# =============================
# THREAD (keep the UI responsive)
# =============================
def clear_output():
    output.delete("1.0", tk.END)


def append_log(msg):
    output.insert(tk.END, msg + "\n")
    output.see(tk.END)


def set_progress(value, message):
    progress_var.set(max(0, min(100, float(value))))
    progress_text_var.set(f"{message}  {int(progress_var.get())}%")


def finish_run():
    run_button.config(state="normal")
    stop_button.config(state="disabled")


def start():
    """Start the resolver in a background thread and keep the UI responsive."""
    global stop_flag
    stop_flag = False
    run_button.config(state="disabled")
    stop_button.config(state="normal")
    set_progress(0, "Starting...")

    # Read Tkinter control values on the UI thread before starting the worker.
    args = (show_all_var.get(), only_full_var.get(), target_override.get())
    threading.Thread(target=run, args=args, daemon=True).start()


def stop():
    """Request cancellation of the current search."""
    global stop_flag
    stop_flag = True
    progress_text_var.set("Stopping after the current request...")

# =============================
# UI (user interface)
# =============================

def refresh_targets():
    try:
        targets = get_targets()
        target_override_combo["values"] = [""] + targets
    except:
        target_override_combo["values"] = ["", "cpu"]


root = tk.Tk()
root.title("PyTorch Resolver")
root.geometry("950x750")

frame = tk.Frame(root)
frame.pack(pady=(10, 4))

run_button = tk.Button(frame, text="Run", command=start)
run_button.pack(side="left", padx=4)
stop_button = tk.Button(frame, text="Stop", command=stop, state="disabled")
stop_button.pack(side="left", padx=4)

# Blue determinate progress bar for the resolver stages.
progress_style = ttk.Style(root)
progress_style.configure(
    "Blue.Horizontal.TProgressbar",
    troughcolor="#d9e7f7",
    background="#1976d2",
    lightcolor="#1976d2",
    darkcolor="#1565c0",
    bordercolor="#b7cde5",
)
progress_var = tk.DoubleVar(value=0)
progress_text_var = tk.StringVar(value="Ready  0%")
progress_frame = ttk.Frame(root)
progress_frame.pack(fill="x", padx=18, pady=(4, 10))
ttk.Label(progress_frame, textvariable=progress_text_var).pack(anchor="w", pady=(0, 3))
progress_bar = ttk.Progressbar(
    progress_frame,
    variable=progress_var,
    maximum=100,
    mode="determinate",
    style="Blue.Horizontal.TProgressbar",
)
progress_bar.pack(fill="x")

show_all_var = tk.BooleanVar()
only_full_var = tk.BooleanVar()

tk.Checkbutton(root, text="Show all wheels", variable=show_all_var).pack()
tk.Checkbutton(root, text="Only fully compatible sets", variable=only_full_var).pack()

# ✅ target override dropdown
target_override = tk.StringVar()
ttk.Label(root, text="Target override:").pack()

target_override_combo = ttk.Combobox(root, textvariable=target_override)
target_override_combo.pack()

frame_output = tk.Frame(root)
frame_output.pack(fill="both", expand=True)

scrollbar = tk.Scrollbar(frame_output)

output = tk.Text(frame_output, yscrollcommand=scrollbar.set)

scrollbar.config(command=output.yview)

scrollbar.pack(side="right", fill="y")
output.pack(side="left", fill="both", expand=True)

refresh_targets()

root.mainloop()
