# Joy's WIDS

A wireless intrusion detection and prevention system (WIDS/WIPS) that uses machine learning to detect
Wi-Fi deauthentication and disassociation flood attacks. It has a Streamlit GUI, a scikit-learn classifier
(Random Forest or Decision Tree), and an automatic response that blocks and logs the offending MAC address.

Use only on networks you own or are authorised to test.

## 1. Quick start (VS Code)

1. Install Python 3.10+ and VS Code (with the Python extension).
2. Open the `joys_wids` folder: File > Open Folder.
3. Open a terminal in VS Code (Terminal > New Terminal) and run:

        python -m venv .venv
        .venv\Scripts\activate          (Windows)   |   source .venv/bin/activate   (Linux/macOS)
        pip install -r requirements.txt
        streamlit run app.py

4. The GUI opens in your browser at http://localhost:8501.
5. In the app: **Train model**, then **Detect & protect**.

## 2. How the system works

    Frames -> Features -> Classifier -> Alert / Block

1. **Capture** (`wids/data.py`): frames come from an AWID3 CSV, a PCAP file, a live monitor-mode interface,
   or built-in synthetic demo data. All are converted to one table: time, source MAC, destination MAC,
   frame type, subtype, signal strength (RSSI), length, label.
2. **Features** (`wids/features.py`): traffic is cut into 1-second windows. For each device in each window,
   10 features are computed (frames per second, share of management frames, deauth count, disassoc count,
   deauth share, unique destinations, broadcast share, RSSI mean and spread, average frame length).
3. **Classifier** (`wids/model.py`): a scikit-learn Random Forest or Decision Tree is trained on labelled
   windows and labels each device-window as normal or malicious. Accuracy, precision, recall, F1 and the
   confusion matrix are reported on a held-out test set.
4. **Response** (`wids/wips.py`): every detection is written to `logs/alerts.csv`. In prevention mode the
   offending MAC is added to `logs/blocklist.csv`. Trusted devices (for example your own access points) are
   alerted on but never blocked, because attackers can spoof an AP's MAC address.

## 3. Files

| File | Purpose |
|---|---|
| `app.py` | The GUI: Overview, Train model, Detect & protect, Logs |
| `wids/data.py` | Loaders for AWID3 CSV, PCAP, live capture, synthetic data |
| `wids/features.py` | Feature extraction per device per time window |
| `wids/model.py` | Training, evaluation, prediction, saving the model |
| `wids/wips.py` | Alert logging and block/log response |
| `wids/plots.py` | Confusion matrix, feature importance, attack timeline |
| `requirements.txt` | Python dependencies |
| `.streamlit/config.toml` | GUI theme and upload limit |
| `logs/` | Alerts and blocklist written at run time |
| `models/` | Trained model saved at run time |

## 4. Using the AWID3 dataset

1. Download the dataset from Kaggle (suumia/awid3-dataset) and unzip it.
2. Train model > "AWID3 CSV - file path" > paste the full path to the deauth CSV. Start with 500,000 rows.
3. Open "How the CSV was read" to confirm the columns and labels were detected.
4. Choose the Chronological split for the strictest test. Download the confusion matrix and feature
   importance PNGs for the poster.
5. To test on unseen traffic: Detect & protect > "AWID3 CSV", with "Rows to skip first" set beyond the
   rows used for training.

## 5. Results

Fill in after training on AWID3:

| Metric | Value |
|---|---|
| Accuracy | |
| Precision | |
| Recall | |
| F1 | |

## 6. Limitations and future work

- Windows are per device; a spoofed AP MAC mixes attack and legitimate frames in one window.
- The block action is a logged blocklist. Real enforcement (for example `ebtables` on a gateway) is future work.
- Only deauth/disassoc floods are targeted. Other AWID3 attacks (evil twin, Kr00k, rogue AP) are future work.
- Live capture needs Linux, root and an adapter in monitor mode, and has not been tested on hardware.

## 7. AI assistance statement 

AI coding assistants (Claude) were used to generate the initial code structure and GUI. I, Joy Ndira, reviewed
the code, tested it on AWID3, and can explain each stage of the pipeline.
