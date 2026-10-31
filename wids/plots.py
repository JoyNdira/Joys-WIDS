"""Matplotlib figures. dark=True for the GUI, dark=False for print-friendly poster exports."""
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

TEAL, RED, AMBER = "#19D3B4", "#FF4D6D", "#FFB547"


def _theme(dark):
    return dict(bg="#0A1220" if dark else "white", fg="#E4ECF7" if dark else "#1A2433",
                grid="#24324A" if dark else "#D9DFE8", teal=TEAL if dark else "#0E9F8A")


def _style(ax, t):
    ax.set_facecolor(t["bg"])
    for s in ax.spines.values():
        s.set_color(t["grid"])
    ax.tick_params(colors=t["fg"], labelsize=9)
    ax.xaxis.label.set_color(t["fg"]); ax.yaxis.label.set_color(t["fg"]); ax.title.set_color(t["fg"])


def _fig(w, h, t):
    fig, ax = plt.subplots(figsize=(w, h), dpi=130)
    fig.patch.set_facecolor(t["bg"])
    _style(ax, t)
    return fig, ax


def confusion(cm, dark=True, title="Confusion matrix"):
    t = _theme(dark)
    fig, ax = _fig(4.6, 4.0, t)
    cmap = LinearSegmentedColormap.from_list("c", [t["bg"], t["teal"]])
    ax.imshow(cm, cmap=cmap)
    labels = ["Normal", "Malicious"]
    ax.set_xticks([0, 1], labels); ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title, fontsize=11)
    top = cm.max() if cm.max() else 1
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=15, fontweight="bold",
                    color=t["bg"] if cm[i, j] > top * 0.55 else t["fg"])
    ax.grid(False)
    fig.tight_layout()
    return fig


def importances(imp: dict, dark=True):
    t = _theme(dark)
    items = sorted(imp.items(), key=lambda kv: kv[1])
    fig, ax = _fig(5.4, 4.0, t)
    ax.barh([k for k, _ in items], [v for _, v in items], color=t["teal"])
    ax.set_xlabel("Importance"); ax.set_title("What the model relies on", fontsize=11)
    ax.grid(axis="x", color=t["grid"], lw=0.6); ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def timeline(feat, dark=True):
    """Top: deauth+disassoc frames per second. Bottom: total frames per second. Red = alert raised."""
    t = _theme(dark)
    g = feat.groupby("t").agg(mgmt_kill=("deauth_count", "sum"), dis=("disassoc_count", "sum"),
                              frames=("fps", "sum"), alert=("alert", "max")).reset_index()
    g["kill"] = g["mgmt_kill"] + g["dis"]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 4.6), dpi=130, sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    fig.patch.set_facecolor(t["bg"])
    for a in (a1, a2):
        _style(a, t); a.grid(axis="y", color=t["grid"], lw=0.6); a.set_axisbelow(True)
    width = max(g["t"].diff().median() if len(g) > 1 else 1, 1) * 0.9
    a1.bar(g["t"], g["kill"], width=width, color=np.where(g["alert"], RED, t["teal"]))
    a1.set_ylabel("Deauth + disassoc\nframes per window"); a1.set_title("Management-frame activity (red = alert raised)", fontsize=11)
    a2.fill_between(g["t"], g["frames"], color=AMBER, alpha=0.5, step="mid")
    a2.set_ylabel("All frames\nper second"); a2.set_xlabel("Time since capture start (s)")
    fig.tight_layout()
    return fig


def png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()
