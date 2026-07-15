# Image Captioning (YOLOv8 + ResNet50 + LSTM) — Streamlit App

## ⚠️ Required before deploying
This repo contains the **app code only**. The trained model artifacts are NOT included
(they're produced by `copy_of_caption_genrator_with_yoloandresnet_for_scenary.py` and are
too large / environment-specific to ship in source control). Before deploying, add these
two files to the repo root:

- `caption_model.keras` — the trained LSTM decoder
- `tokenizer.pkl` — the fitted Keras `Tokenizer`

`yolov8m.pt` does NOT need to be added — `ultralytics` downloads it automatically on first run.

If either file is missing, the app will show a clear error on startup instead of crashing.

> If `caption_model.keras` is larger than 100MB, GitHub will reject a normal push — use
> [Git LFS](https://git-lfs.com/) to track it.

## Deploying on Streamlit Community Cloud
1. Push this repo (with the two model files added) to GitHub.
2. On [share.streamlit.io](https://share.streamlit.io), create a new app pointing at `app.py`.
3. Streamlit Cloud will automatically:
   - install the apt packages listed in `packages.txt`
   - install the Python packages listed in `requirements.txt`, using the Python version pinned in `runtime.txt`
4. First load will be slow (~1–2 min) while YOLOv8m and ResNet50 weights download — this is expected.

### Note on resources
This app loads TensorFlow, PyTorch (via `ultralytics`), YOLOv8m, and ResNet50 simultaneously,
which is memory-heavy. If you hit out-of-memory errors on a free-tier host, consider a smaller
YOLO checkpoint (e.g. `yolov8n.pt`) or a paid tier with more RAM.

## Local development
```bash
pip install -r requirements.txt
streamlit run app.py
```
