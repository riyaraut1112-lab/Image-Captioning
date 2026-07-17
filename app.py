import os
import pickle
import numpy as np
import cv2
import tensorflow as tf
from collections import Counter
import streamlit as st
from PIL import Image
import tempfile

from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

from ultralytics import YOLO

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
st.set_page_config(page_title="Image Caption Generator", layout="wide")

MAX_LENGTH = 34
BEAM_WIDTH = 5
ALPHA = 0.8

TOKENIZER_PATH = "tokenizer.pkl"       # <-- from your Colab training
CAPTION_MODEL_PATH = "caption_model.keras"  # <-- from your Colab training
YOLO_WEIGHTS = "yolov8m.pt"            # auto-downloads if not present

# -------------------------------------------------------------------
# 1. LOAD ALL MODELS (cached — only runs once)
# -------------------------------------------------------------------
@st.cache_resource
def load_all_models():
    yolo_model = YOLO(YOLO_WEIGHTS)
    resnet_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")

    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)

    lstm_model = load_model(CAPTION_MODEL_PATH)

    return yolo_model, resnet_model, tokenizer, lstm_model


with st.spinner("Loading models (YOLOv8, ResNet50, LSTM)... this can take a minute on first run"):
    yolo_model, resnet_model, tokenizer, lstm_model = load_all_models()

# -------------------------------------------------------------------
# 2. FEATURE EXTRACTION (identical logic to your notebook)
# -------------------------------------------------------------------
def extract_image_features(img_path):
    """Extracts and combines YOLOv8m and ResNet50 features for a single image."""
    yolo_results = yolo_model(img_path, verbose=False)[0]
    boxes = yolo_results.boxes
    num_objs = len(boxes)
    yolo_feats = []
    detected_classes = []

    names = yolo_model.names

    if num_objs > 0:
        conf_scores = boxes.conf.cpu().numpy()
        sorted_indices = np.argsort(conf_scores)[::-1]

        for i in range(min(15, num_objs)):
            idx = sorted_indices[i]
            b = boxes[idx]
            class_id = float(b.cls[0].cpu().numpy())
            conf = float(b.conf[0].cpu().numpy())

            if conf > 0.40:
                detected_classes.append(names[int(class_id)])

            cx, cy, w, h = b.xywhn[0].cpu().numpy().tolist()
            area = w * h
            yolo_feats.extend([class_id, conf, cx, cy, w, h, area, cx, cy])

    padding_needed = 135 - len(yolo_feats)
    yolo_feats.extend([0.0] * padding_needed)
    yolo_feats.append(float(num_objs))
    yolo_vector = np.array(yolo_feats)

    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    resnet_vector = resnet_model.predict(img_array, verbose=0).flatten()

    combined_vector = np.concatenate((resnet_vector, yolo_vector))

    return np.array([combined_vector]), yolo_results, detected_classes


# -------------------------------------------------------------------
# 3. CAPTION GENERATION — BEAM SEARCH + SEMANTIC INJECTION (identical logic)
# -------------------------------------------------------------------
def int_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


def generate_caption_beam_search(model, tokenizer, image_feature, max_length,
                                  detected_classes, beam_width=5, alpha=0.8):
    start_word = "<start>"
    beam = [([start_word], 0.0)]

    img_tensor = tf.convert_to_tensor(image_feature)

    stop_words = {"a", "the", "and", "is", "in", "on", "of", "with", "at", "to", "by", "an", "are"}
    class_counts = Counter(detected_classes)

    for _ in range(max_length):
        candidates = []
        for seq, score in beam:
            if seq[-1] == "end":
                candidates.append((seq, score))
                continue

            seq_str = " ".join(seq)
            encoded_seq = tokenizer.texts_to_sequences([seq_str])[0]
            padded_seq = pad_sequences([encoded_seq], maxlen=max_length, padding="post")
            seq_tensor = tf.convert_to_tensor(padded_seq)

            yhat = model.predict_on_batch([img_tensor, seq_tensor])[0]

            current_max_prob = np.max(yhat)
            yolo_seen_broad = set()

            for class_name, count in class_counts.items():
                boost_words = [class_name]
                if class_name == "person":
                    if count > 1:
                        boost_words.extend(["women", "men", "people", "group", "girls", "friends"])
                    else:
                        boost_words.extend(["man", "woman", "boy", "girl", "runner"])
                elif class_name == "dog":
                    boost_words.extend(["dogs", "puppies", "pack"] if count > 1 else ["puppy", "hound"])
                elif class_name == "cat":
                    boost_words.extend(["cats", "kittens"] if count > 1 else ["kitten", "feline"])
                elif class_name == "car":
                    boost_words.extend(["cars", "vehicles"] if count > 1 else ["vehicle", "automobile"])
                elif class_name == "handbag":
                    boost_words.extend(["bag", "purse"])

                yolo_seen_broad.update(boost_words)

                if not any(w in seq for w in boost_words):
                    for w in boost_words:
                        if w in tokenizer.word_index:
                            word_idx = tokenizer.word_index[w]
                            yhat[word_idx] += (current_max_prob * 0.35)

            common_biases = {"dog", "dogs", "man", "woman", "boy", "girl", "person", "people", "child", "children"}
            hallucination_risks = common_biases - yolo_seen_broad

            for risk_word in hallucination_risks:
                if risk_word in tokenizer.word_index:
                    word_idx = tokenizer.word_index[risk_word]
                    yhat[word_idx] *= 0.001

            for word in set(seq):
                if word not in stop_words and word in tokenizer.word_index:
                    word_idx = tokenizer.word_index[word]
                    yhat[word_idx] *= 0.001

            if len(seq) > 0:
                last_word = seq[-1]
                if last_word in tokenizer.word_index:
                    yhat[tokenizer.word_index[last_word]] *= 0.001

            yhat = yhat / (np.sum(yhat) + 1e-10)

            top_indices = np.argsort(yhat)[-beam_width:]

            for idx in top_indices:
                word = int_to_word(idx, tokenizer)
                if word is None:
                    continue

                prob = yhat[idx]
                new_score = score - np.log(prob + 1e-10)
                new_seq = seq + [word]
                candidates.append((new_seq, new_score))

        def score_with_length_penalty(item):
            s, current_score = item
            L = len(s) - 1
            penalty = (L ** alpha) if L > 0 else 1.0
            return current_score / penalty

        beam = sorted(candidates, key=score_with_length_penalty)[:beam_width]

        if all(seq[-1] == "end" for seq, _ in beam):
            break

    best_seq = beam[0][0]

    if best_seq[0] == "<start>":
        best_seq = best_seq[1:]
    if best_seq[-1] == "end":
        best_seq = best_seq[:-1]

    return " ".join(best_seq).strip()


# -------------------------------------------------------------------
# 4. STREAMLIT UI
# -------------------------------------------------------------------
st.title("🖼️ Scenery Image Caption Generator")
st.caption("YOLOv8 object detection + ResNet50 features + LSTM caption generation (beam search)")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save to a temp path since YOLO + load_img both expect a file path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    image = Image.open(tmp_path).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Generate Caption"):
        with st.spinner("Detecting objects with YOLO..."):
            img_feature, yolo_results, detected_classes = extract_image_features(tmp_path)

        st.write(f"**Detected objects (conf > 0.40):** {detected_classes if detected_classes else 'None'}")

        with st.spinner("Generating caption (beam search)..."):
            caption = generate_caption_beam_search(
                lstm_model, tokenizer, img_feature, MAX_LENGTH,
                detected_classes, beam_width=BEAM_WIDTH, alpha=ALPHA
            )

        # Show YOLO-annotated image
        plotted_img_bgr = yolo_results.plot()
        plotted_img_rgb = cv2.cvtColor(plotted_img_bgr, cv2.COLOR_BGR2RGB)
        st.image(plotted_img_rgb, caption="YOLO Detections", use_container_width=True)

        st.subheader("Generated Caption")
        st.success(caption.capitalize() + ".")

    os.unlink(tmp_path) if False else None  # keep temp file alive until Streamlit reruns; cleaned by OS temp gc

st.sidebar.markdown("### Models used")
st.sidebar.markdown(
    "- **YOLOv8m** — object detection (`yolov8m.pt`, auto-downloaded)\n"
    "- **ResNet50** — image feature extraction (ImageNet weights, auto-downloaded)\n"
    "- **Custom LSTM** — caption decoder (`caption_model.keras`, your trained model)\n"
    "- **Tokenizer** — `tokenizer.pkl`, your trained tokenizer"
)
    resnet_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")

    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)

    lstm_model = load_model(CAPTION_MODEL_PATH)

    return yolo_model, resnet_model, tokenizer, lstm_model


with st.spinner("Loading models (YOLOv8, ResNet50, LSTM)... this can take a minute on first run"):
    yolo_model, resnet_model, tokenizer, lstm_model = load_all_models()

# -------------------------------------------------------------------
# 2. FEATURE EXTRACTION (identical logic to your notebook)
# -------------------------------------------------------------------
def extract_image_features(img_path):
    """Extracts and combines YOLOv8m and ResNet50 features for a single image."""
    yolo_results = yolo_model(img_path, verbose=False)[0]
    boxes = yolo_results.boxes
    num_objs = len(boxes)
    yolo_feats = []
    detected_classes = []

    names = yolo_model.names

    if num_objs > 0:
        conf_scores = boxes.conf.cpu().numpy()
        sorted_indices = np.argsort(conf_scores)[::-1]

        for i in range(min(15, num_objs)):
            idx = sorted_indices[i]
            b = boxes[idx]
            class_id = float(b.cls[0].cpu().numpy())
            conf = float(b.conf[0].cpu().numpy())

            if conf > 0.40:
                detected_classes.append(names[int(class_id)])

            cx, cy, w, h = b.xywhn[0].cpu().numpy().tolist()
            area = w * h
            yolo_feats.extend([class_id, conf, cx, cy, w, h, area, cx, cy])

    padding_needed = 135 - len(yolo_feats)
    yolo_feats.extend([0.0] * padding_needed)
    yolo_feats.append(float(num_objs))
    yolo_vector = np.array(yolo_feats)

    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    resnet_vector = resnet_model.predict(img_array, verbose=0).flatten()

    combined_vector = np.concatenate((resnet_vector, yolo_vector))

    return np.array([combined_vector]), yolo_results, detected_classes


# -------------------------------------------------------------------
# 3. CAPTION GENERATION — BEAM SEARCH + SEMANTIC INJECTION (identical logic)
# -------------------------------------------------------------------
def int_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


def generate_caption_beam_search(model, tokenizer, image_feature, max_length,
                                  detected_classes, beam_width=5, alpha=0.8):
    start_word = "<start>"
    beam = [([start_word], 0.0)]

    img_tensor = tf.convert_to_tensor(image_feature)

    stop_words = {"a", "the", "and", "is", "in", "on", "of", "with", "at", "to", "by", "an", "are"}
    class_counts = Counter(detected_classes)

    for _ in range(max_length):
        candidates = []
        for seq, score in beam:
            if seq[-1] == "end":
                candidates.append((seq, score))
                continue

            seq_str = " ".join(seq)
            encoded_seq = tokenizer.texts_to_sequences([seq_str])[0]
            padded_seq = pad_sequences([encoded_seq], maxlen=max_length, padding="post")
            seq_tensor = tf.convert_to_tensor(padded_seq)

            yhat = model.predict_on_batch([img_tensor, seq_tensor])[0]

            current_max_prob = np.max(yhat)
            yolo_seen_broad = set()

            for class_name, count in class_counts.items():
                boost_words = [class_name]
                if class_name == "person":
                    if count > 1:
                        boost_words.extend(["women", "men", "people", "group", "girls", "friends"])
                    else:
                        boost_words.extend(["man", "woman", "boy", "girl", "runner"])
                elif class_name == "dog":
                    boost_words.extend(["dogs", "puppies", "pack"] if count > 1 else ["puppy", "hound"])
                elif class_name == "cat":
                    boost_words.extend(["cats", "kittens"] if count > 1 else ["kitten", "feline"])
                elif class_name == "car":
                    boost_words.extend(["cars", "vehicles"] if count > 1 else ["vehicle", "automobile"])
                elif class_name == "handbag":
                    boost_words.extend(["bag", "purse"])

                yolo_seen_broad.update(boost_words)

                if not any(w in seq for w in boost_words):
                    for w in boost_words:
                        if w in tokenizer.word_index:
                            word_idx = tokenizer.word_index[w]
                            yhat[word_idx] += (current_max_prob * 0.35)

            common_biases = {"dog", "dogs", "man", "woman", "boy", "girl", "person", "people", "child", "children"}
            hallucination_risks = common_biases - yolo_seen_broad

            for risk_word in hallucination_risks:
                if risk_word in tokenizer.word_index:
                    word_idx = tokenizer.word_index[risk_word]
                    yhat[word_idx] *= 0.001

            for word in set(seq):
                if word not in stop_words and word in tokenizer.word_index:
                    word_idx = tokenizer.word_index[word]
                    yhat[word_idx] *= 0.001

            if len(seq) > 0:
                last_word = seq[-1]
                if last_word in tokenizer.word_index:
                    yhat[tokenizer.word_index[last_word]] *= 0.001

            yhat = yhat / (np.sum(yhat) + 1e-10)

            top_indices = np.argsort(yhat)[-beam_width:]

            for idx in top_indices:
                word = int_to_word(idx, tokenizer)
                if word is None:
                    continue

                prob = yhat[idx]
                new_score = score - np.log(prob + 1e-10)
                new_seq = seq + [word]
                candidates.append((new_seq, new_score))

        def score_with_length_penalty(item):
            s, current_score = item
            L = len(s) - 1
            penalty = (L ** alpha) if L > 0 else 1.0
            return current_score / penalty

        beam = sorted(candidates, key=score_with_length_penalty)[:beam_width]

        if all(seq[-1] == "end" for seq, _ in beam):
            break

    best_seq = beam[0][0]

    if best_seq[0] == "<start>":
        best_seq = best_seq[1:]
    if best_seq[-1] == "end":
        best_seq = best_seq[:-1]

    return " ".join(best_seq).strip()


# -------------------------------------------------------------------
# 4. STREAMLIT UI
# -------------------------------------------------------------------
st.title("🖼️ Scenery Image Caption Generator")
st.caption("YOLOv8 object detection + ResNet50 features + LSTM caption generation (beam search)")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save to a temp path since YOLO + load_img both expect a file path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    image = Image.open(tmp_path).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Generate Caption"):
        with st.spinner("Detecting objects with YOLO..."):
            img_feature, yolo_results, detected_classes = extract_image_features(tmp_path)

        st.write(f"**Detected objects (conf > 0.40):** {detected_classes if detected_classes else 'None'}")

        with st.spinner("Generating caption (beam search)..."):
            caption = generate_caption_beam_search(
                lstm_model, tokenizer, img_feature, MAX_LENGTH,
                detected_classes, beam_width=BEAM_WIDTH, alpha=ALPHA
            )

        # Show YOLO-annotated image
        plotted_img_bgr = yolo_results.plot()
        plotted_img_rgb = cv2.cvtColor(plotted_img_bgr, cv2.COLOR_BGR2RGB)
        st.image(plotted_img_rgb, caption="YOLO Detections", use_container_width=True)

        st.subheader("Generated Caption")
        st.success(caption.capitalize() + ".")

    os.unlink(tmp_path) if False else None  # keep temp file alive until Streamlit reruns; cleaned by OS temp gc

st.sidebar.markdown("### Models used")
st.sidebar.markdown(
    "- **YOLOv8m** — object detection (`yolov8m.pt`, auto-downloaded)\n"
    "- **ResNet50** — image feature extraction (ImageNet weights, auto-downloaded)\n"
    "- **Custom LSTM** — caption decoder (`caption_model.keras`, your trained model)\n"
    "- **Tokenizer** — `tokenizer.pkl`, your trained tokenizer"
)
