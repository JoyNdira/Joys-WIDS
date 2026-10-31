"""Joy's WIDS - wireless intrusion detection & prevention prototype (Streamlit GUI)."""
import tempfile
import time

import pandas as pd
import streamlit as st

from wids import data, features, model as mdl, plots
from wids.wips import ALERTS_CSV, BLOCKLIST_CSV, Wips

st.set_page_config(page_title="Joy's WIDS", page_icon="📡", layout="wide")

RED, TEAL = "#FF4D6D", "#19D3B4"
DEFAULT_TRUSTED = ", ".join(data.AP_MACS)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@500&display=swap');
html, body, [class*="css"], .stMarkdown, button, input, textarea { font-family: 'IBM Plex Sans', sans-serif; }
.block-container { padding-top: 1.6rem; max-width: 1280px; }
.hero { display:flex; justify-content:space-between; align-items:center; padding:1.3rem 1.6rem; margin-bottom:1.2rem;
        border:1px solid #223250; border-left:5px solid #19D3B4; border-radius:10px; background:#0E1829; }
.hero h1 { margin:0; font-size:2rem; font-weight:600; letter-spacing:-0.5px; }
.hero p { margin:.2rem 0 0; color:#8FA3BF; font-size:.95rem; }
.pill { font-family:'IBM Plex Mono', monospace; font-size:.8rem; padding:.35rem .8rem; border-radius:99px; border:1px solid; }
.pill.ok { color:#19D3B4; border-color:#19D3B4; background:rgba(25,211,180,.08); }
.pill.off { color:#FFB547; border-color:#FFB547; background:rgba(255,181,71,.08); }
div[data-testid="stMetric"] { background:#0E1829; border:1px solid #223250; border-radius:8px; padding:.7rem 1rem; }
div[data-testid="stMetricValue"] { font-family:'IBM Plex Mono', monospace; font-size:1.55rem; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------ state ---
if "bundle" not in st.session_state:
    try:
        st.session_state.bundle = mdl.load_saved()
    except Exception:
        st.session_state.bundle = None
for k in ("train_info", "det"):
    st.session_state.setdefault(k, None)
bundle = st.session_state.bundle

# ---------------------------------------------------------------- sidebar ---
with st.sidebar:
    st.markdown("### 📡 Joy's WIDS")
    page = st.radio("Navigate", ["Overview", "Train model", "Detect & protect", "Logs"], label_visibility="collapsed")
    st.divider()
    st.markdown("**Response mode**")
    prevention = st.toggle("Prevention (WIPS)", value=True, help="On: block and log the offending MAC. Off: detection and alerts only.")
    trusted = st.text_area("Trusted devices (alert, never block)", DEFAULT_TRUSTED, height=90,
                           help="Comma-separated MACs, e.g. your own access points. Attackers can spoof these, so they are alerted on but not blocked.")
    st.divider()
    if bundle:
        st.caption(f"Model: {bundle['algo']}  \nAccuracy: {bundle['metrics']['accuracy']:.2%}  \nTrained: {bundle['trained_at']}")
    else:
        st.caption("No model trained yet.")
    st.caption("Use only on networks you own or are authorised to test.")

pill = (f'<span class="pill ok">● model ready · {bundle["algo"]}</span>' if bundle
        else '<span class="pill off">● no model - train one first</span>')
st.markdown(f'<div class="hero"><div><h1>Joy\'s WIDS</h1><p>Wireless intrusion detection and prevention using machine learning</p></div>{pill}</div>',
            unsafe_allow_html=True)


# ---------------------------------------------------------------- helpers ---
def csv_inputs(key: str, allow_upload=True):
    """Shared widgets for choosing an AWID3 CSV. Returns (source, kwargs) or (None, None)."""
    c1, c2, c3 = st.columns(3)
    max_rows = c1.number_input("Rows to read", 10_000, 5_000_000, 500_000, 50_000, key=f"{key}_n")
    skip = c2.number_input("Rows to skip first", 0, 50_000_000, 0, 100_000, key=f"{key}_s",
                           help="Lets you test on a later part of the file than you trained on.")
    deauth_only = c3.checkbox("Keep Normal + Deauth only", True, key=f"{key}_d",
                              help="Drops other AWID3 attack classes so the model focuses on deauthentication floods.")
    kwargs = dict(max_rows=int(max_rows), skip_rows=int(skip), deauth_only=deauth_only)
    if allow_upload:
        up = st.file_uploader("AWID3 CSV file", type="csv", key=f"{key}_u")
        return up, kwargs
    path = st.text_input("Path to AWID3 CSV on this computer", key=f"{key}_p", placeholder=r"C:\Users\Joy\Downloads\AWID3\Deauth.csv")
    return (path.strip().strip('"') or None), kwargs


def feature_table():
    return pd.DataFrame({"Feature": list(features.DESCRIPTIONS), "What it measures": list(features.DESCRIPTIONS.values())})


# --------------------------------------------------------------- Overview ---
def page_overview():
    st.subheader("How it works")
    cols = st.columns(4)
    steps = [("1. Capture", "Reads 802.11 frames from an AWID3 CSV, a PCAP file, or a live monitor-mode interface."),
             ("2. Extract features", "Splits traffic into 1-second windows and summarises each device: frame types, signal strength, frames per second."),
             ("3. Classify", "A scikit-learn Random Forest or Decision Tree labels every device-window as normal or malicious."),
             ("4. Respond", "Raises an alert. In prevention mode it also blocks and logs the offending MAC address.")]
    for c, (h, b) in zip(cols, steps):
        with c.container(border=True):
            st.markdown(f"**{h}**")
            st.write(b)

    st.subheader("Get started")
    st.markdown("Open **Train model** to teach the classifier, then **Detect & protect** to run it on a capture.")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Features the model sees")
        st.dataframe(feature_table(), hide_index=True, width="stretch")
    with right:
        st.subheader("Why these features catch a deauth flood")
        st.write("A deauthentication flood sends hundreds of management frames per second. "
                 "A normal access point sends one every minute or so. "
                 "The deauth count, its share of the device's traffic and the frame rate separate the two clearly.")


# ------------------------------------------------------------ Train model ---
def show_training_results(b):
    m = b["metrics"]
    c = st.columns(4)
    c[0].metric("Accuracy", f"{m['accuracy']:.2%}")
    c[1].metric("Precision", f"{m['precision']:.2%}", help="Of everything flagged, how much was a real attack.")
    c[2].metric("Recall", f"{m['recall']:.2%}", help="Of all real attacks, how many were caught.")
    c[3].metric("F1 score", f"{m['f1']:.2%}")
    st.caption(f"{b['algo']} · {b['split']} split · {b['n_train']:,} training / {b['n_test']:,} test windows · {b['window']:g}s windows. "
               "Attacks are rare, so read precision and recall alongside accuracy.")
    l, r = st.columns(2)
    with l:
        fig = plots.confusion(m["cm"], dark=True)
        st.pyplot(fig, width="stretch")
        st.download_button("Download confusion matrix (poster PNG)", plots.png_bytes(plots.confusion(m["cm"], dark=False)),
                           "confusion_matrix.png", "image/png")
    with r:
        st.pyplot(plots.importances(b["importances"], dark=True), width="stretch")
        st.download_button("Download feature importance (poster PNG)", plots.png_bytes(plots.importances(b["importances"], dark=False)),
                           "feature_importance.png", "image/png")
    with st.expander("Per-class report"):
        st.dataframe(m["report"], width="stretch")
    if b.get("tree_rules"):
        with st.expander("Decision rules the tree learned (great for explaining it)"):
            st.code(b["tree_rules"])


def page_train():
    st.subheader("Train the model")
    mode = st.radio("Training data", ["Built-in demo data (synthetic, AWID3-style)", "AWID3 CSV - file path", "AWID3 CSV - upload"], horizontal=True)
    c1, c2, c3 = st.columns(3)
    window = c1.select_slider("Window length (seconds)", [0.5, 1.0, 2.0, 5.0], value=1.0)
    algo = c2.selectbox("Classifier", ["Random Forest", "Decision Tree"])
    split = c3.selectbox("Train/test split", ["Random (stratified)", "Chronological"],
                         help="Chronological trains on the earlier part of the capture and tests on the later part - a stricter test.")
    src, kw, duration = None, {}, 600
    if mode.startswith("Built-in"):
        duration = st.slider("Demo capture length (seconds)", 300, 1800, 600, 60)
        st.info("The demo data imitates AWID3: beacons, client traffic, rare legitimate deauths, and several deauth/disassoc flood attacks. "
                "Use the real AWID3 CSV for the figures on your poster.")
    else:
        src, kw = csv_inputs("train", allow_upload=mode.endswith("upload"))

    if st.button("Train model", type="primary"):
        try:
            if mode.startswith("Built-in"):
                frames, info = data.generate_synthetic(duration, seed=1), {}
            else:
                if src is None:
                    raise ValueError("Choose a CSV file first.")
                with st.spinner("Reading CSV..."):
                    frames, info = data.load_awid3_csv(src, **kw)
            with st.spinner("Extracting features..."):
                feat = features.extract_features(frames, window)
            with st.spinner("Training classifier..."):
                b = mdl.train(feat, algo, window, 0.3, split)
            st.session_state.bundle = b
            st.session_state.train_info = {"source": mode, "frames": len(frames), "windows": len(feat),
                                           "attack_windows": int(feat["label"].sum()), "info": info}
            st.session_state.det = None
            st.rerun()
        except Exception as e:
            st.error(f"Training failed: {e}")

    b, ti = st.session_state.bundle, st.session_state.train_info
    if b:
        st.divider()
        st.subheader("Results")
        if ti:
            st.caption(f"Data: {ti['source']} · {ti['frames']:,} frames → {ti['windows']:,} device-windows ({ti['attack_windows']:,} malicious)")
            if ti["info"]:
                with st.expander("How the CSV was read (columns used, label counts)"):
                    st.json({"columns_used": ti["info"]["columns_used"], "rows_read": ti["info"]["rows_read"],
                             "rows_kept": ti["info"]["rows_kept"], "labels_in_file": ti["info"]["label_counts"]})
        show_training_results(b)


# -------------------------------------------------------- Detect & protect ---
def run_replay(det, speed):
    feat = det["feat"]
    groups = dict(tuple(feat.groupby("win")))
    wins = sorted(groups)
    alert_wins = feat.loc[feat["alert"], "win"]
    start = max(wins[0], int(alert_wins.min()) - 15) if len(alert_wins) else wins[0]
    seq = [w for w in wins if w >= start][:90]

    banner = st.empty()
    mcols = st.columns(4)
    phs = [c.empty() for c in mcols]
    chart = st.empty()
    table = st.empty()
    engine = Wips(det["trusted"], det["prevention"], persist=False)
    hist, acts, n_alerts = [], [], 0

    for w in seq:
        g = groups[w]
        al = g[g["alert"]]
        t = float(g["t"].iloc[0])
        hist.append({"time (s)": t, "deauth + disassoc frames": int(g["deauth_count"].sum() + g["disassoc_count"].sum())})
        if len(al):
            n_alerts += len(al)
            new = engine.process(al)
            if not new.empty:
                acts.append(new)
            srcs = ", ".join(al["src"].unique()[:2])
            banner.error(f"🚨 Deauthentication flood detected at t = {t:.0f}s from {srcs}")
        else:
            banner.success(f"✅ Network normal at t = {t:.0f}s")
        phs[0].metric("Time", f"{t:.0f}s")
        phs[1].metric("Alerts", n_alerts)
        phs[2].metric("Devices handled", len(engine.blocked))
        phs[3].metric("Frames this second", f"{g['fps'].sum():.0f}")
        chart.line_chart(pd.DataFrame(hist).set_index("time (s)"), color=RED, height=230)
        if acts:
            table.dataframe(pd.concat(acts), hide_index=True, width="stretch")
        time.sleep(1.0 / speed)
    st.caption("Replay is a visual walk-through. The logs are written when you press Analyse.")


def page_detect():
    b = st.session_state.bundle
    if not b:
        st.warning("Train a model first (Train model page).")
        return
    st.subheader("Analyse a capture")
    mode = st.radio("Traffic source", ["New synthetic capture", "AWID3 CSV - file path", "AWID3 CSV - upload",
                                        "PCAP file - upload", "Live capture (monitor mode)"], horizontal=True)
    thr = st.slider("Alert sensitivity (alert when attack score ≥)", 0.1, 0.9, 0.5, 0.05,
                    help="Lower catches more attacks but risks false alarms.")
    src, kw, seed, pcap, iface, secs = None, {}, 99, None, "", 30
    if mode.startswith("New synthetic"):
        seed = st.number_input("Random seed", 1, 9999, 99)
    elif mode.startswith("AWID3"):
        src, kw = csv_inputs("det", allow_upload=mode.endswith("upload"))
    elif mode.startswith("PCAP"):
        pcap = st.file_uploader("PCAP/PCAPNG captured in monitor mode", type=["pcap", "pcapng", "cap"])
    else:
        st.warning("Linux only, run as root, interface already in monitor mode, on a network you are authorised to test.")
        iface = st.text_input("Interface", "wlan0mon")
        secs = st.slider("Capture seconds", 10, 120, 30)

    if st.button("Analyse capture", type="primary"):
        try:
            with st.spinner("Reading traffic..."):
                if mode.startswith("New synthetic"):
                    frames = data.generate_synthetic(300, seed=int(seed))
                elif mode.startswith("AWID3"):
                    if src is None:
                        raise ValueError("Choose a CSV file first.")
                    frames, _ = data.load_awid3_csv(src, **kw)
                elif mode.startswith("PCAP"):
                    if pcap is None:
                        raise ValueError("Upload a capture file first.")
                    with tempfile.NamedTemporaryFile(suffix=".pcap", delete=False) as tf:
                        tf.write(pcap.getvalue())
                    frames = data.load_pcap(tf.name)
                else:
                    frames = data.capture_frames(iface, secs)
            feat = features.extract_features(frames, b["window"])
            feat["score"] = mdl.predict_scores(b, feat)
            feat["alert"] = feat["score"] >= thr
            trusted_list = [m for m in trusted.replace("\n", ",").split(",")]
            engine = Wips(trusted_list, prevention, persist=True)
            actions = engine.process(feat[feat["alert"]])
            st.session_state.det = {"feat": feat, "frames": len(frames), "actions": actions,
                                    "trusted": trusted_list, "prevention": prevention,
                                    "labelled": bool((feat["label"] >= 0).all())}
        except Exception as e:
            st.error(f"Analysis failed: {e}")

    det = st.session_state.det
    if not det:
        return
    feat, actions = det["feat"], det["actions"]
    alerts = feat[feat["alert"]]
    st.divider()
    if len(alerts):
        st.error(f"🚨 {len(alerts)} alert(s) raised from {alerts['src'].nunique()} device(s).")
    else:
        st.success("✅ No attacks detected in this capture.")
    c = st.columns(5)
    c[0].metric("Frames analysed", f"{det['frames']:,}")
    c[1].metric("Device-windows", f"{len(feat):,}")
    c[2].metric("Alerts", f"{len(alerts):,}")
    c[3].metric("Devices flagged", alerts["src"].nunique())
    c[4].metric("Blocked", int(actions["action"].str.startswith("BLOCKED").sum()) if len(actions) else 0)

    if det["labelled"]:
        m = mdl.metrics_for(feat["label"].astype(int), feat["alert"].astype(int))
        st.markdown("**Scored against the labels in this capture**")
        l, r = st.columns([1, 2])
        with l:
            st.pyplot(plots.confusion(m["cm"], dark=True, title="This capture"), width="stretch")
        with r:
            k = st.columns(3)
            k[0].metric("Accuracy", f"{m['accuracy']:.2%}")
            k[1].metric("Precision", f"{m['precision']:.2%}")
            k[2].metric("Recall", f"{m['recall']:.2%}")

    st.pyplot(plots.timeline(feat, dark=True), width="stretch")
    st.download_button("Download timeline (poster PNG)", plots.png_bytes(plots.timeline(feat, dark=False)), "timeline.png", "image/png")

    t1, t2, t3 = st.tabs(["WIPS actions", "Incidents by device", "All alerts"])
    with t1:
        if len(actions):
            st.dataframe(actions, hide_index=True, width="stretch")
        else:
            st.write("No actions taken.")
    with t2:
        if len(alerts):
            inc = alerts.groupby("src").agg(first_seen_s=("t", "min"), last_seen_s=("t", "max"), windows=("t", "size"),
                                            deauth_frames=("deauth_count", "sum"), disassoc_frames=("disassoc_count", "sum"),
                                            peak_fps=("fps", "max"), avg_score=("score", "mean")).round(2).reset_index()
            st.dataframe(inc, hide_index=True, width="stretch")
        else:
            st.write("Nothing to show.")
    with t3:
        st.dataframe(alerts[["t", "src", "deauth_count", "disassoc_count", "fps", "rssi_mean", "score"]].round(2),
                     hide_index=True, width="stretch")

    st.divider()
    st.subheader("Live replay")
    st.caption("Plays the capture window by window, as if it were arriving live.")
    speed = st.slider("Speed (windows per second)", 1, 20, 6)
    if st.button("Start replay"):
        run_replay(det, speed)


# ------------------------------------------------------------------- Logs ---
def page_logs():
    st.subheader("Logs")
    t1, t2 = st.tabs(["Alerts", "Blocklist"])
    for tab, path in ((t1, ALERTS_CSV), (t2, BLOCKLIST_CSV)):
        with tab:
            if path.exists():
                df = pd.read_csv(path)
                st.caption(f"{len(df):,} entries · {path.name}")
                st.dataframe(df.iloc[::-1], hide_index=True, width="stretch")
                st.download_button(f"Download {path.name}", path.read_bytes(), path.name, "text/csv", key=path.name)
            else:
                st.info("Nothing logged yet. Run an analysis on the Detect & protect page.")
    if st.button("Clear all logs"):
        for p in (ALERTS_CSV, BLOCKLIST_CSV):
            p.unlink(missing_ok=True)
        st.rerun()


{"Overview": page_overview, "Train model": page_train, "Detect & protect": page_detect, "Logs": page_logs}[page]()
