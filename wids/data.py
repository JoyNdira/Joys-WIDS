"""Getting wireless frames into one simple table.

Every source (synthetic demo, AWID3 CSV, PCAP file, live capture) is converted to
the same schema, so the rest of the system never cares where the frames came from.

  time    seconds
  src     transmitter MAC
  dst     receiver MAC
  ftype   0 = management, 1 = control, 2 = data
  subtype 802.11 subtype (management: 8 = beacon, 10 = disassoc, 12 = deauth)
  rssi    signal strength in dBm
  length  frame size in bytes
  label   0 = normal, 1 = malicious, -1 = unknown (unlabelled capture)
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

BROADCAST = "ff:ff:ff:ff:ff:ff"
COLS = ["time", "src", "dst", "ftype", "subtype", "rssi", "length", "label"]
DEAUTH, DISASSOC = 12, 10

ATTACKER_MAC = "de:ad:be:ef:00:01"
AP_MACS = [f"aa:bb:cc:00:00:{i:02x}" for i in range(3)]


# --------------------------------------------------------------------------- #
# 1. Synthetic demo traffic (AWID3-style) - used when the real dataset is absent
# --------------------------------------------------------------------------- #
def _frames(t, src, dst, ftype, subtype, rssi_mu, rssi_sd, len_mu, label, rng):
    n = len(t)
    return pd.DataFrame({
        "time": t,
        "src": src,
        "dst": dst,
        "ftype": ftype,
        "subtype": subtype,
        "rssi": np.round(rng.normal(rssi_mu, rssi_sd, n)),
        "length": np.clip(rng.normal(len_mu, len_mu * 0.2, n), 24, 2346).round(),
        "label": label,
    })


def generate_synthetic(duration: int = 600, seed: int = 1, n_clients: int = 12) -> pd.DataFrame:
    """Normal Wi-Fi traffic plus deauthentication/disassociation flood episodes."""
    rng = np.random.default_rng(seed)
    parts = []

    for i, ap in enumerate(AP_MACS):                                   # beacons
        n = rng.poisson(10 * duration)
        parts.append(_frames(rng.uniform(0, duration, n), ap, BROADCAST, 0, 8, -45 - 6 * i, 2.5, 260, 0, rng))
        n = rng.poisson(duration / 45)                                 # rare legitimate deauths
        t = rng.uniform(0, duration, n)
        dst = rng.choice([f"00:11:22:33:44:{k:02x}" for k in range(n_clients)], n)
        parts.append(_frames(t, ap, dst, 0, rng.choice([DEAUTH, DISASSOC], n), -45 - 6 * i, 2.5, 30, 0, rng))

    for i in range(n_clients):                                         # clients
        c = f"00:11:22:33:44:{i:02x}"
        ap_i = i % len(AP_MACS)
        ap = AP_MACS[ap_i]
        rate = rng.choice([3, 8, 15, 30, 60])
        rssi = rng.uniform(-75, -45)
        n = rng.poisson(rate * duration)
        parts.append(_frames(rng.uniform(0, duration, n), c, ap, 2, 40, rssi, 3, 600, 0, rng))
        n = rng.poisson(rate * 0.8 * duration)
        parts.append(_frames(rng.uniform(0, duration, n), ap, c, 2, 40, -45 - 6 * ap_i, 2.5, 700, 0, rng))
        n = rng.poisson(0.3 * duration)
        parts.append(_frames(rng.uniform(0, duration, n), c, BROADCAST, 0, 4, rssi, 3, 120, 0, rng))

    n_ep = max(3, duration // 90)                                      # attacks
    seg = duration / n_ep
    for k in range(n_ep):
        start = k * seg + rng.uniform(0.15, 0.6) * seg
        length = rng.uniform(10, 25)
        rate = float(np.exp(rng.uniform(np.log(15), np.log(300))))
        spoof = (k % 3 == 2)                                           # every 3rd attack spoofs an AP MAC
        src = AP_MACS[int(rng.integers(len(AP_MACS)))] if spoof else ATTACKER_MAC
        n = rng.poisson(rate * length)
        t = rng.uniform(start, start + length, n)
        dst = BROADCAST if k % 2 == 0 else f"00:11:22:33:44:{int(rng.integers(n_clients)):02x}"
        sub = rng.choice([DEAUTH, DISASSOC], n, p=[0.85, 0.15])
        parts.append(_frames(t, src, dst, 0, sub, rng.uniform(-65, -35), 3, 30, 1, rng))

    df = pd.concat(parts, ignore_index=True).sort_values("time").reset_index(drop=True)
    return df[COLS]


# --------------------------------------------------------------------------- #
# 2. AWID3 CSV
# --------------------------------------------------------------------------- #
_ALIASES = {
    "time": ["frame.time_epoch", "frame.time_relative", "time"],
    "src": ["wlan.sa", "wlan.ta"],
    "dst": ["wlan.da", "wlan.ra"],
    "subtype": ["wlan.fc.type_subtype", "wlan.fc.subtype"],
    "ftype": ["wlan.fc.type"],
    "rssi": ["radiotap.dbm_antsignal", "wlan_radio.signal_dbm"],
    "length": ["frame.len", "frame.cap_len"],
    "label": ["Label", "label", "class", "Class", "attack", "Attack"],
}
_NORMAL = {"normal", "benign", "0", "0.0", "none", "legitimate"}


def _to_num(s: pd.Series) -> pd.Series:
    s = s.astype("string").str.split(",").str[0].str.strip()
    hexm = s.str.lower().str.startswith("0x").fillna(False).astype(bool)
    out = pd.to_numeric(s.where(~hexm), errors="coerce").astype(float)
    if hexm.any():
        out.loc[hexm] = s[hexm].map(lambda v: int(v, 16)).astype(float)
    return out


def _to_mac(s: pd.Series) -> pd.Series:
    s = s.astype("string").str.split(",").str[0].str.strip().str.lower()
    return s.mask(s == "")


def _natural(p):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(p))]


def load_awid3_csv(source, max_rows: int | None = None, skip_rows: int = 0, deauth_only: bool = True):
    """Read an AWID3 CSV file, an uploaded file, or a FOLDER of CSV parts. Returns (frames, info)."""
    if isinstance(source, (str, os.PathLike)) and os.path.isdir(source):
        return _load_folder(Path(source), max_rows, skip_rows, deauth_only)
    return _load_one_csv(source, max_rows, skip_rows, deauth_only)


def _load_folder(folder, max_rows, skip_rows, deauth_only):
    files = sorted((p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() == ".csv"), key=_natural)
    if not files:
        raise ValueError(f"No .csv files found in {folder}")
    frames, used, counts, rows_read, remaining, skip, last = [], [], {}, 0, max_rows, skip_rows, {}
    for f in files:
        if remaining is not None and remaining <= 0:
            break
        if skip > 0:
            with open(f, "rb") as fh:
                n = sum(1 for _ in fh) - 1
            if skip >= n:
                skip -= n
                continue
        try:
            fr, info = _load_one_csv(str(f), remaining, skip, deauth_only)
        except ValueError as e:
            raise ValueError(f"{f.name}: {e}")
        skip = 0
        frames.append(fr); used.append(f.name); rows_read += info["rows_read"]; last = info
        for k, v in info["label_counts"].items():
            counts[k] = counts.get(k, 0) + v
        if remaining is not None:
            remaining -= info["rows_read"]
    if not frames:
        raise ValueError("'Rows to skip first' is larger than the data in the folder.")
    out = pd.concat(frames, ignore_index=True).sort_values("time").reset_index(drop=True)
    return out, {"columns_used": last["columns_used"], "rows_read": rows_read, "rows_kept": len(out),
                 "label_counts": counts, "files_used": used}


def _load_one_csv(source, max_rows, skip_rows, deauth_only):
    cols = list(pd.read_csv(source, nrows=0).columns)
    if hasattr(source, "seek"):
        source.seek(0)
    lower = {c.lower(): c for c in cols}

    found = {k: list(dict.fromkeys(lower[a.lower()] for a in names if a.lower() in lower)) for k, names in _ALIASES.items()}
    missing = [name for name, key in (("frame.time_epoch", "time"), ("wlan.sa or wlan.ta", "src"),
                                      ("wlan.fc.type_subtype or wlan.fc.subtype", "subtype"),
                                      ("Label", "label")) if not found[key]]
    if "wlan.fc.type_subtype" not in lower and "wlan.fc.subtype" in lower and not found["ftype"]:
        missing.append("wlan.fc.type")
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing) + ". All columns in this file: " + ", ".join(cols))

    usecols = sorted({c for v in found.values() for c in v})
    raw = pd.read_csv(source, usecols=usecols, nrows=max_rows, dtype=str, low_memory=False,
                      skiprows=range(1, skip_rows + 1) if skip_rows else None)

    def first(key):  # first non-empty value across alias columns
        sub = raw[found[key]].astype("string")
        return sub.bfill(axis=1).iloc[:, 0]

    sub_full = _to_num(first("subtype"))
    if found["ftype"] and sub_full.max() <= 15:
        ftype, subtype = _to_num(first("ftype")), sub_full
    else:                                   # Wireshark combines them: value = type * 16 + subtype
        ftype, subtype = sub_full // 16, sub_full % 16

    lab_raw = first("label").astype("string").str.strip().str.lower().fillna("normal")
    is_normal = lab_raw.isin(_NORMAL)
    out = pd.DataFrame({
        "time": _to_num(first("time")),
        "src": _to_mac(first("src")),
        "dst": _to_mac(first("dst")).fillna("unknown") if found["dst"] else "unknown",
        "ftype": ftype, "subtype": subtype,
        "rssi": _to_num(first("rssi")) if found["rssi"] else np.nan,
        "length": _to_num(first("length")) if found["length"] else 0.0,
        "label": (~is_normal).astype(int),
    })
    label_counts = lab_raw.value_counts().to_dict()

    if deauth_only:                          # keep Normal + deauth/disassoc attacks only
        keep = is_normal | lab_raw.str.contains("deauth|disas", regex=True)
        out = out[keep.values]
    out = out.dropna(subset=["time", "src", "ftype", "subtype"])
    out["rssi"] = out["rssi"].fillna(out["rssi"].median() if out["rssi"].notna().any() else -80.0)
    out["length"] = out["length"].fillna(0)
    out = out.sort_values("time").reset_index(drop=True)[COLS]

    info = {"columns_used": {k: v for k, v in found.items() if v}, "rows_read": len(raw),
            "rows_kept": len(out), "label_counts": label_counts}
    return out, info


# --------------------------------------------------------------------------- #
# 3. PCAP files and live monitor-mode capture (needs scapy)
# --------------------------------------------------------------------------- #
def packet_to_row(pkt):
    from scapy.layers.dot11 import Dot11, RadioTap
    if not pkt.haslayer(Dot11):
        return None
    d = pkt[Dot11]
    if not d.addr2:
        return None
    rssi = np.nan
    if pkt.haslayer(RadioTap):
        v = getattr(pkt[RadioTap], "dBm_AntSignal", None)
        if v is not None:
            rssi = float(v)
    return (float(pkt.time), d.addr2.lower(), (d.addr1 or "unknown").lower(),
            int(d.type), int(d.subtype), rssi, len(pkt), -1)


def _rows_to_frames(rows) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=COLS)
    if df.empty:
        raise ValueError("No 802.11 frames found. Capture in monitor mode with radiotap headers.")
    df["rssi"] = df["rssi"].fillna(df["rssi"].median() if df["rssi"].notna().any() else -80.0)
    return df.sort_values("time").reset_index(drop=True)


def load_pcap(path: str, max_packets: int | None = None) -> pd.DataFrame:
    from scapy.utils import PcapReader
    rows = []
    with PcapReader(path) as reader:
        for i, pkt in enumerate(reader):
            if max_packets and i >= max_packets:
                break
            r = packet_to_row(pkt)
            if r:
                rows.append(r)
    return _rows_to_frames(rows)


def capture_frames(iface: str, seconds: int) -> pd.DataFrame:
    """Sniff `seconds` of traffic. Linux, root, and an interface already in monitor mode."""
    from scapy.sendrecv import sniff
    rows = []

    def cb(pkt):
        r = packet_to_row(pkt)
        if r:
            rows.append(r)

    sniff(iface=iface, timeout=seconds, store=False, prn=cb)
    return _rows_to_frames(rows)