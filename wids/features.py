"""Turn raw frames into one row of simple features per (time window, device)."""
import numpy as np
import pandas as pd

from .data import BROADCAST, DEAUTH, DISASSOC

FEATURES = ["fps", "mgmt_ratio", "deauth_count", "disassoc_count", "deauth_ratio",
            "uniq_dst", "bcast_ratio", "rssi_mean", "rssi_std", "len_mean"]

DESCRIPTIONS = {
    "fps": "Frames per second sent by the device",
    "mgmt_ratio": "Share of its frames that are management frames (frame type)",
    "deauth_count": "Deauthentication frames in the window",
    "disassoc_count": "Disassociation frames in the window",
    "deauth_ratio": "Share of its frames that are deauth/disassoc",
    "uniq_dst": "Number of different destinations it talked to",
    "bcast_ratio": "Share of its frames sent to the broadcast address",
    "rssi_mean": "Average signal strength (dBm)",
    "rssi_std": "Signal strength variation (spoofed senders jump around)",
    "len_mean": "Average frame length (bytes)",
}


def extract_features(frames: pd.DataFrame, window: float = 1.0) -> pd.DataFrame:
    df = frames.copy()
    df["win"] = np.floor((df["time"] - df["time"].min()) / window).astype(int)
    mgmt = df["ftype"] == 0
    df["is_mgmt"] = mgmt
    df["is_deauth"] = mgmt & (df["subtype"] == DEAUTH)
    df["is_disas"] = mgmt & (df["subtype"] == DISASSOC)
    df["is_bcast"] = df["dst"] == BROADCAST

    f = df.groupby(["win", "src"]).agg(
        frames=("time", "size"), mgmt_n=("is_mgmt", "sum"),
        deauth_count=("is_deauth", "sum"), disassoc_count=("is_disas", "sum"),
        uniq_dst=("dst", "nunique"), bcast_n=("is_bcast", "sum"),
        rssi_mean=("rssi", "mean"), rssi_std=("rssi", "std"),
        len_mean=("length", "mean"), label=("label", "max"),
    ).reset_index()

    f["fps"] = f["frames"] / window
    f["mgmt_ratio"] = f["mgmt_n"] / f["frames"]
    f["deauth_ratio"] = (f["deauth_count"] + f["disassoc_count"]) / f["frames"]
    f["bcast_ratio"] = f["bcast_n"] / f["frames"]
    f["rssi_std"] = f["rssi_std"].fillna(0)
    f["t"] = f["win"] * window            # seconds from the start of the capture
    return f.drop(columns=["mgmt_n", "bcast_n"])
