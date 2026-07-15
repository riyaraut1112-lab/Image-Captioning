import os
import pickle
import uuid
import numpy as np
import cv2
import tensorflow as tf
from collections import Counter
import streamlit as st

# Keras / TensorFlow Imports
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# YOLOv8 Import
from ultralytics import YOLO

# Set Page Configuration
st.set_page_config(
    page_title="Hybrid Image Caption Generator",
    page_icon="📸",
    layout="centered"
)

# -------------------------------------------------------------------
# 1. CACHED MODEL LOADERS (Optimized for Streamlit Memory Usage)
# -------------------------------------------------------------------
@st.cache_resource
def load_yolo_model():
    # Will download 'yolov8m.pt' automatically if not present in the root directory
    return YOLO('yolov8m.pt')

@st.cache_resource
def load_resnet_model():
    return ResNet50(weights='imagenet', include_top=False, pooling='avg')

@st.cache_resource
def load_caption_components():
    # Load custom sequence items
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    lstm_model = load_model('caption_model.keras')
    return tokenizer, lstm_model

# --- Helper function for integer-to-word conversion ---
def int_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# -------------------------------------------------------------------
# 2. FEATURE EXTRACTION PIPELINE
# -------------------------------------------------------------------
def extract_image_features(temp_img_path, _yolo_model, _resnet_model):
    """Extracts and combines YOLOv8m and ResNet50 features for a single image."""
    # Run YOLOv8
    yolo_results = _yolo_model(temp_img_path, verbose=False)[0]
    boxes = yolo_results.boxes
    num_objs = len(boxes)
    yolo_feats = []
    detected_classes = []
    names = _yolo_model.names

    if num_objs > 0:
        conf_scores = boxes.conf.cpu().numpy()
        sorted_indices = np.argsort(conf_scores)[::-1]

        for i in range(min(15, num_objs)):
            idx = sorted_indices[i]
            b = boxes[idx]
            class_id = float(b.cls[0].cpu().numpy())
            conf = float(b.conf[0].cpu().numpy())

            # Only use highly confident detections for Semantic Injection
            if conf > 0.40:
                detected_classes.append(names[int(class_id)])

            cx, cy, w, h = b.xywhn[0].cpu().numpy().tolist()
            area = w * h
            yolo_feats.extend([class_id, conf, cx, cy, w, h, area, cx, cy])

    # Pad to 15 objects
    padding_needed = 135 - len(yolo_feats)
    yolo_feats.extend([0.0] * padding_needed)
    yolo_feats.append(float(num_objs))
    yolo_vector = np.array(yolo_feats)

    # ResNet50 Extraction
    # Read the image using OpenCV / Native Keras
    img_bgr = cv2.imread(temp_img_path)
    if img_bgr is None:
        raise ValueError("Could not read the uploaded image. Please try a different JPG/PNG file.")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    img_array = img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    resnet_vector = _resnet_model.predict(img_array, verbose=0).flatten()

    # Combine Vectors (Shape: 2184,)
    combined_vector = np.concatenate((resnet_vector, yolo_vector))
    return np.array([combined_vector]), yolo_results, detected_classes

# -------------------------------------------------------------------
# 3. TEXT GENERATION WITH BEAM SEARCH & SEMANTIC INJECTION
# -------------------------------------------------------------------
def generate_caption_beam_search(model, tokenizer, image_feature, max_length, detected_classes, beam_width=5, alpha=0.8):
    """Generates a caption using Beam Search, Length Normalization, and Count-Aware YOLO Semantic Injection."""
    start_word = '<start>'
    beam = [([start_word], 0.0)]
    img_tensor = tf.convert_to_tensor(image_feature)

    stop_words = {'a', 'the', 'and', 'is', 'in', 'on', 'of', 'with', 'at', 'to', 'by', 'an', 'are'}
    class_counts = Counter(detected_classes)

    for _ in range(max_length):
        candidates = []
        for seq, score in beam:
            if seq[-1] == 'end':
                candidates.append((seq, score))
                continue

            seq_str = ' '.join(seq)
            encoded_seq = tokenizer.texts_to_sequences([seq_str])[0]
            padded_seq = pad_sequences([encoded_seq], maxlen=max_length, padding='post')
            seq_tensor = tf.convert_to_tensor(padded_seq)

            yhat = model.predict_on_batch([img_tensor, seq_tensor])[0]

            # COUNT-AWARE DYNAMIC SEMANTIC INJECTION
            current_max_prob = np.max(yhat)
            yolo_seen_broad = set()

            for class_name, count in class_counts.items():
                boost_words = [class_name]
                if class_name == 'person':
                    if count > 1:
                        boost_words.extend(['women', 'men', 'people', 'group', 'girls', 'friends'])
                    else:
                        boost_words.extend(['man', 'woman', 'boy', 'girl', 'runner'])
                elif class_name == 'dog':
                    boost_words.extend(['dogs', 'puppies', 'pack'] if count > 1 else ['puppy', 'hound'])
                elif class_name == 'cat':
                    boost_words.extend(['cats', 'kittens'] if count > 1 else ['kitten', 'feline'])
                elif class_name == 'car':
                    boost_words.extend(['cars', 'vehicles'] if count > 1 else ['vehicle', 'automobile'])
                elif class_name == 'handbag':
                    boost_words.extend(['bag', 'purse'])

                yolo_seen_broad.update(boost_words)

                # Apply injection boost only if not mentioned yet
                if not any(w in seq for w in boost_words):
                    for w in boost_words:
                        if w in tokenizer.word_index:
                            word_idx = tokenizer.word_index[w]
                            yhat[word_idx] += (current_max_prob * 0.35)

            # ACTIVE SUPPRESSION (NEGATIVE INJECTION)
            common_biases = {'dog', 'dogs', 'man', 'woman', 'boy', 'girl', 'person', 'people', 'child', 'children'}
            hallucination_risks = common_biases - yolo_seen_broad

            for risk_word in hallucination_risks:
                if risk_word in tokenizer.word_index:
                    word_idx = tokenizer.word_index[risk_word]
                    yhat[word_idx] *= 0.001

            # ADVANCED GLOBAL REPETITION PENALTY
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

        if all(seq[-1] == 'end' for seq, _ in beam):
            break

    best_seq = beam[0][0]
    if best_seq[0] == '<start>':
        best_seq = best_seq[1:]
    if best_seq[-1] == 'end':
        best_seq = best_seq[:-1]

    return ' '.join(best_seq).strip()

# -------------------------------------------------------------------
# 4. STREAMLIT UI INTERFACE
# -------------------------------------------------------------------
st.title("📸 Hybrid Scenery Caption Generator")
st.write("Generate rich captions using **YOLOv8 Object Detection** combined with a **ResNet50 + LSTM Decoder** with Count-Aware Semantic Injection.")

# Sidebar Controls
st.sidebar.header("Model Settings")
beam_width = st.sidebar.slider("Beam Width", min_value=1, max_value=10, value=5, step=1)
alpha = st.sidebar.slider("Length Penalty (Alpha)", min_value=0.0, max_value=1.5, value=0.8, step=0.1)

# Status indicators
try:
    with st.spinner("Loading AI Models... (First run may take a minute)"):
        yolo_model = load_yolo_model()
        resnet_model = load_resnet_model()
        tokenizer, lstm_model = load_caption_components()
    st.success("All models loaded successfully!")
except Exception as e:
    st.error(f"Error loading models. Have you placed `caption_model.keras` and `tokenizer.pkl` in this directory? \nDetail: {e}")
    st.stop()

# File Uploader
uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save the uploaded file to a temporary location for OpenCV / YOLO processing
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    file_ext = os.path.splitext(uploaded_file.name)[1] or ".jpg"
    temp_path = os.path.join(temp_dir, f"{uuid.uuid4().hex}{file_ext}")

    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(uploaded_file, use_container_width=True)

    try:
        with st.spinner("Processing image and generating caption..."):
            # 1. Feature Extraction
            img_feature, yolo_results, detected_classes = extract_image_features(temp_path, yolo_model, resnet_model)

            # 2. Decode with Beam Search and Semantic Injection
            MAX_LENGTH = 34
            caption = generate_caption_beam_search(
                lstm_model,
                tokenizer,
                img_feature,
                MAX_LENGTH,
                detected_classes,
                beam_width=beam_width,
                alpha=alpha
            )

        with col2:
            st.subheader("YOLOv8 Detection")
            # Plot YOLO predictions onto the image
            plotted_bgr = yolo_results.plot()
            plotted_rgb = cv2.cvtColor(plotted_bgr, cv2.COLOR_BGR2RGB)
            st.image(plotted_rgb, use_container_width=True)

        # Display results
        st.markdown("---")
        st.subheader("Generated Caption:")
        st.info(f"👉 **{caption.capitalize()}.**")

        if detected_classes:
            st.write(f"**Detected tags for Semantic Injection:** `{', '.join(set(detected_classes))}`")
        else:
            st.write("*No high-confidence YOLOv8 tags detected for injection.*")
    except Exception as e:
        st.error(f"Something went wrong while processing this image: {e}")
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
