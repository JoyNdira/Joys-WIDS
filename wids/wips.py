"""Alert logging (IDS) and automatic response (WIPS): block + log the offending MAC."""
from datetime import datetime
from pathlib import Path

import pandas as pd

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
ALERTS_CSV = LOG_DIR / "alerts.csv"
BLOCKLIST_CSV = LOG_DIR / "blocklist.csv"


def _append(df: pd.DataFrame, path: Path):
    LOG_DIR.mkdir(exist_ok=True)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


class Wips:
    def __init__(self, trusted=(), prevention=True, persist=True):
        self.trusted = {m.strip().lower() for m in trusted if m.strip()}
        self.prevention = prevention
        self.persist = persist
        self.blocked: dict[str, str] = {}

    def process(self, alerts: pd.DataFrame) -> pd.DataFrame:
        """`alerts` = rows the model flagged. Returns one action row per newly handled device."""
        cols = ["time", "mac", "action", "reason"]
        if alerts.empty:
            return pd.DataFrame(columns=cols)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if self.persist:
            _append(alerts.assign(logged_at=now)[["logged_at", "t", "src", "deauth_count", "disassoc_count", "fps", "rssi_mean", "score"]], ALERTS_CSV)

        actions = []
        for _, a in alerts.sort_values("t").iterrows():
            mac = a["src"]
            if mac in self.blocked:
                continue
            reason = f"{int(a['deauth_count'])} deauth + {int(a['disassoc_count'])} disassoc frames, {a['fps']:.0f} fps, score {a['score']:.2f}"
            if not self.prevention:
                action = "ALERT (detection only)"
            elif mac in self.trusted:
                action = "ALERT - trusted device, not blocked"
            else:
                action = "BLOCKED - MAC added to blocklist"
            self.blocked[mac] = action
            actions.append({"time": a["t"], "mac": mac, "action": action, "reason": reason})

        out = pd.DataFrame(actions, columns=cols)
        if self.persist and self.prevention and not out.empty:
            blocked = out[out["action"].str.startswith("BLOCKED")]
            if not blocked.empty:
                _append(blocked.assign(logged_at=now)[["logged_at", "mac", "action", "reason"]], BLOCKLIST_CSV)
        return out
