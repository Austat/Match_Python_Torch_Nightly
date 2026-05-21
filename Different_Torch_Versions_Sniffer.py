"""
========================================
-PSEUDOKOODI (selkokielellä)
========================================

Ajattele tätä ohjelmaa näin:

1) Tarkistetaan kone:
   - Mikä Python versio?
   - Onko NVIDIA GPU (CUDA)?
   - Mikä käyttöjärjestelmä?

2) Haetaan PyTorchin sivulta lista:
   - eri CUDA-versiot (cu121, cu118 jne.)
   - CPU vaihtoehto

3) Käydään läpi jokainen vaihtoehto:
   - haetaan torch, torchaudio, torchcodec wheelit
   - suodatetaan sopivat (Python + platform)

4) Yritetään yhdistää:
   - samalta build-päivältä (devYYYYMMDD)
   - niin että kaikki 3 kirjastoa löytyy

5) Valitaan paras:
   - käyttäjän CUDA mieluummin
   - tai fallback

6) Tulostetaan:
   ✅ mitä löytyi
   ✅ mikä on paras setti
   ✅ suoraan pip install -komento

7) UI:
   - nappi "Suorita"
   - nappi "Stop"
   - checkboxit
   - dropdown override
   - logi-ikkuna

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
stop_flag = False  # 🔴 tätä käytetään keskeyttämiseen UI:sta


# =============================
# DETECT (ympäristön tunnistus)
# =============================
def detect_cuda():
    """
    :
    Yritetään selvittää onko koneessa GPU ja CUDA käytössä.
    1) kysytään torchilta
    2) jos ei toimi → käytetään nvidia-smi
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

    return "Ei CUDA"


def cuda_to_target(cuda):
    """
    :
    Muutetaan CUDA versio muotoon:
    12.1 → cu121
    """
    if "Ei CUDA" in cuda:
        return "cpu"

    try:
        major, minor = cuda.split(".")
        return f"cu{major}{minor}"
    except:
        return "cpu"


def detect_platform():
    """
    :
    Selvitetään käyttöjärjestelmä → wheel filtteröintiä varten
    """
    if platform.system() == "Windows":
        return "win_amd64"
    elif platform.system() == "Linux":
        return "manylinux"
    return "macosx"


# =============================
# ENV (ympäristötiedot)
# =============================
def get_env():
    """
    Kerätään kaikki tärkeät tiedot yhteen paikkaan
    """

    def safe(p):
        # : kokeillaan onko kirjasto asennettu
        try:
            return __import__(p).__version__
        except:
            return "Ei asennettu"

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
    Haetaan kaikki saatavilla olevat CUDA buildit netistä
    """

    try:
        r = requests.get(BASE_URL, timeout=10)
        dirs = re.findall(r'href="(cu\d+/)"', r.text)
        dirs = [d.strip("/") for d in dirs]
    except:
        dirs = []

    targets = ["cpu"] + dirs

    # järjestetään niin että cpu viimeisenä

    return sorted(
        targets,
        key=lambda x: (x == "cpu", int(x.replace("cu","")) if x != "cpu" else -1),
        reverse=True
    )

# =============================
# FETCH (data haku)
# =============================

def fetch_wheels(url):
    try:
        r = requests.get(url, timeout=10)
        links = re.findall(r'href="([^"]+\.whl[^"]*)"', r.text)

        fixed = []
        for l in links:
            l = unquote(l)

            # ✅ korjaus: jos suhteellinen polku → lisää domain
            if l.startswith("/"):
                l = "https://download-r2.pytorch.org" + l

            fixed.append(l)

        return fixed

    except:
        return []

def fetch_target(target, libs, show_all, plat, py_tag):
    global stop_flag

    result = {}
    logs = []   # 🔥 kerätään logit tänne

    if stop_flag:
        return target, {}, logs

    logs.append(f"→ {target}")

    for lib in libs:

        if stop_flag:
            return target, {}, logs

        url = f"{BASE_URL}{target}/{lib}/"
        wheels = fetch_wheels(url)

        wheels = [w for w in wheels if plat in w and py_tag in w]

        logs.append(f"   {lib}: {len(wheels)}")

        limit = None if show_all else 5

        for w in wheels[:limit]:
            logs.append(f"      {w}")

        result[lib] = wheels

    return target, result, logs


def fetch_all(progress, show_all, plat, py_tag):

    global stop_flag

    libs = ["torch", "torchaudio", "torchcodec", "torchvision"]
    targets = get_targets()

    result = {}

    max_workers = os.cpu_count() or 4

    with ThreadPoolExecutor(max_workers=max_workers) as executor:

        futures = {
            executor.submit(fetch_target, t, libs, show_all, plat, py_tag): t
            for t in targets
        }

        completed = {}

        # 🔥 kerätään kaikki ensin
        for future in as_completed(futures):

            if stop_flag:
                break

            target, data, logs = future.result()
            completed[target] = (data, logs)

        # 🔥 TÄRKEÄ: tulostetaan alkuperäisessä järjestyksessä
        for t in targets:
            if t in completed:
                data, logs = completed[t]

                result[t] = data

                for line in logs:
                    progress(line)

    return result

# =============================
# MATCH (yhdistely)
# =============================
def get_date(w):
    """
    :
    Kaivetaan wheel nimestä build päivä (devYYYYMMDD)
    """
    m = re.search(r'dev(\d{8})', w)
    return m.group(1) if m else None


def match(all_data):
    """
    :
    Yritetään löytää samat buildit kaikille kirjastoille
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

    # paras ensin (score desc)
    return sorted(results, reverse=True)


def pick_best(matches, preferred, only_full):
    """
    :
    Valitaan paras:
    - mieluiten sama CUDA
    - mieluiten kaikki 4 kirjastoa
    """

    if only_full:
        matches = [m for m in matches if m[0] == 4]

    for m in matches:
        if m[2] == preferred:
            return m

    return matches[0] if matches else None


# =============================
# MAIN (päälogiikka)
# =============================
def run():
    """
    :
    Tämä on koko ohjelman "suunnitelma"
    """

    global stop_flag
    stop_flag = False

    output.delete("1.0", tk.END)

    def log(msg):
        output.insert(tk.END, msg + "\n")
        output.see(tk.END)
        output.update()

    env = get_env()
    py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"

    log("=== Asennetut versiot ===")
    for k, v in env.items():
        log(f"{k}: {v}")

    log("\n[Haetaan dataa...]")

    data = fetch_all(
        log,
        show_all_var.get(),
        env["Platform"],
        py_tag
    )

    log("\n=== YHTEENVETO ===")
    for t, libs in data.items():
        ok = sum(1 for v in libs.values() if v)
        log(f"{t}: {ok}/4")

    matches = match(data)

    selected_target = target_override.get() or env["Target"]

    best = pick_best(matches, selected_target, only_full_var.get())

    if not best:
        log("\n❌ Ei yhteensopivaa settiä")
        return

    score, date, target, lm = best

    log("\n=== PARAS ===")
    log(f"Target: {target}")
    log(f"Build: {date}")
    log(f"Score: {score}/4\n")

    for lib in ["torch", "torchaudio", "torchcodec", "torchvision"]:
        log(f"{lib}: {'✅' if lib in lm else '❌'}")

    log("\n=== INSTALL ===")
    
    order = ["torch", "torchvision", "torchaudio", "torchcodec"]
    wheels_ordered = [lm[l] for l in order if l in lm]

    log("pip install " + " ".join(f'"{w}"' for w in wheels_ordered))

# =============================
# THREAD (UI ei jäädy)
# =============================
def start():
    """
    :
    Ajetaan run taustasäikeessä → UI pysyy responsiivisena
    """
    threading.Thread(target=run).start()


def stop():
    """
    :
    Käyttäjä voi keskeyttää haun
    """
    global stop_flag
    stop_flag = True


# =============================
# UI (käyttöliittymä)
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
frame.pack()

tk.Button(frame, text="Suorita", command=start).pack(side="left")
tk.Button(frame, text="Stop", command=stop).pack(side="left")

show_all_var = tk.BooleanVar()
only_full_var = tk.BooleanVar()

tk.Checkbutton(root, text="Näytä kaikki wheelit", variable=show_all_var).pack()
tk.Checkbutton(root, text="Vain täysin yhteensopivat", variable=only_full_var).pack()

# ✅ override dropdown
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
