import os
import re
import pickle
import tempfile
import numpy as np
import cv2
import easyocr
import streamlit as st
from PIL import Image
from pdf2image import convert_from_path

# ── Page config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Receipt Fraud Detection",
    page_icon="🔍",
    layout="centered",
)

# ── Load model (cached so it only loads once) ───────────────────────
@st.cache_resource
def load_model():
    with open("trained_models.pkl", "rb") as f:
        data = pickle.load(f)
    best = data["rf_model"] if data["best_model_name"] == "Random Forest" else data["xgb_model"]
    return best, data["feature_cols"], data["best_model_name"]


@st.cache_resource
def load_ocr():
    return easyocr.Reader(["ar", "en"], gpu=False)


best_model, feature_cols, model_name = load_model()
reader = load_ocr()


# ── Feature extraction ──────────────────────────────────────────────
def extract_features(ocr_blocks):
    if not ocr_blocks:
        return None

    confidences = [b["confidence"] for b in ocr_blocks]
    text_lengths = [len(b["text"]) for b in ocr_blocks]
    all_x = [b["cx"] for b in ocr_blocks]
    all_y = [b["cy"] for b in ocr_blocks]
    widths = [b["x_max"] - b["x_min"] for b in ocr_blocks]
    heights = [b["y_max"] - b["y_min"] for b in ocr_blocks]
    all_text = " ".join([b["text"] for b in ocr_blocks])
    all_text_lower = all_text.lower()

    arabic_chars = len(re.findall(r"[\u0600-\u06FF]", all_text))
    english_chars = len(re.findall(r"[a-zA-Z]", all_text))
    total_alpha = arabic_chars + english_chars

    bank_keywords = [
        "الراجحي", "rajhi", "الأهلي", "الاهلي", "snb", "الرياض",
        "riyad", "sabb", "البريطاني", "الإنماء", "الانماء", "inma",
        "البلاد", "bilad",
    ]
    receipt_kw = [
        "transfer receipt", "إيصال التحويل", "إيصال", "ايصال",
        "transfer details", "تفاصيل التحويل",
    ]
    purpose_kw = ["purpose", "الغرض", "غرض", "remark", "ملاحظات"]
    amount_kw = ["amount", "المبلغ"]

    has_full_iban = bool(re.search(r"SA\d{22}", all_text.replace(" ", "")))
    has_masked_iban = bool(re.search(r"SA\*|[\*]{4,}", all_text))
    bottom_blocks = [b for b in ocr_blocks if b["cy"] > 0.8 * max(all_y)]

    features = {
        "conf_mean": np.mean(confidences),
        "conf_std": np.std(confidences),
        "conf_min": np.min(confidences),
        "conf_max": np.max(confidences),
        "conf_median": np.median(confidences),
        "low_conf_ratio": sum(1 for c in confidences if c < 0.5) / len(confidences),
        "num_blocks": len(ocr_blocks),
        "avg_text_len": np.mean(text_lengths),
        "max_text_len": np.max(text_lengths),
        "total_chars": sum(text_lengths),
        "x_spread": max(all_x) - min(all_x),
        "y_spread": max(all_y) - min(all_y),
        "x_std": np.std(all_x),
        "y_std": np.std(all_y),
        "avg_box_width": np.mean(widths),
        "avg_box_height": np.mean(heights),
        "box_width_std": np.std(widths),
        "box_height_std": np.std(heights),
        "arabic_ratio": arabic_chars / total_alpha if total_alpha > 0 else 0,
        "english_ratio": english_chars / total_alpha if total_alpha > 0 else 0,
        "has_bank_name": int(any(kw in all_text_lower for kw in bank_keywords)),
        "has_amount": int(bool(re.search(r"\d+[\.,]?\d*\s*(SAR|sar|ريال|SR)", all_text))),
        "has_date": int(bool(re.search(r"\d{4}[/\-]\d{1,2}[/\-]\d{1,2}", all_text))),
        "has_full_iban": int(has_full_iban),
        "has_masked_iban": int(has_masked_iban),
        "iban_is_masked": int(has_masked_iban and not has_full_iban),
        "has_reference": int(bool(re.search(r"[A-Z0-9]{15,}", all_text.replace(" ", "")))),
        "has_receipt_title": int(any(kw in all_text_lower for kw in receipt_kw)),
        "has_purpose": int(any(kw in all_text_lower for kw in purpose_kw)),
        "has_amount_label": int(any(kw in all_text_lower for kw in amount_kw)),
        "has_bottom_content": int(len(bottom_blocks) > 2),
        "digit_ratio": len(re.findall(r"\d", all_text)) / len(all_text) if all_text else 0,
        "special_ratio": len(re.findall(r"[*\-/\.:,]", all_text)) / len(all_text) if all_text else 0,
        "star_count": all_text.count("*"),
        "completeness": sum([
            int(any(kw in all_text_lower for kw in bank_keywords)),
            int(bool(re.search(r"\d+[\.,]?\d*\s*(SAR|sar|ريال|SR)", all_text))),
            int(bool(re.search(r"\d{4}[/\-]\d{1,2}[/\-]\d{1,2}", all_text))),
            int(has_full_iban or has_masked_iban),
            int(any(kw in all_text_lower for kw in receipt_kw)),
        ]) / 5.0,
    }
    return features


# ── Prediction ──────────────────────────────────────────────────────
def predict_receipt(file_path):
    if file_path.lower().endswith(".pdf"):
        pages = convert_from_path(file_path, dpi=200, first_page=1, last_page=1)
        img = cv2.cvtColor(np.array(pages[0]), cv2.COLOR_RGB2BGR)
    else:
        img = cv2.imread(file_path)
        if img is None:
            img = cv2.cvtColor(np.array(Image.open(file_path).convert("RGB")), cv2.COLOR_RGB2BGR)

    blocks = []
    for bbox, text, conf in reader.readtext(img):
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        blocks.append({
            "text": text.strip(), "confidence": float(conf), "bbox": bbox,
            "x_min": min(xs), "x_max": max(xs),
            "y_min": min(ys), "y_max": max(ys),
            "cx": sum(xs) / 4, "cy": sum(ys) / 4,
        })

    feats = extract_features(blocks)
    if feats is None:
        return None, None, 0

    prob = best_model.predict_proba(np.array([[feats[c] for c in feature_cols]]))[0]
    is_genuine = prob[0] > prob[1]
    confidence = round(max(prob) * 100, 1)

    return is_genuine, prob, len(blocks)


# ── UI ──────────────────────────────────────────────────────────────
st.title("🔍 Receipt Fraud Detection")
st.markdown(f"Using **{model_name}** model with OCR (Arabic + English)")

st.divider()

uploaded = st.file_uploader(
    "Upload a receipt image or PDF",
    type=["png", "jpg", "jpeg", "pdf"],
)

if uploaded:
    # Show preview
    if uploaded.type == "application/pdf":
        st.info("📄 PDF uploaded — first page will be analyzed.")
    else:
        st.image(uploaded, caption="Uploaded receipt", use_container_width=True)

    if st.button("🔎 Analyze Receipt", type="primary", use_container_width=True):
        suffix = "." + uploaded.name.rsplit(".", 1)[-1].lower()

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getvalue())
            tmp_path = tmp.name

        try:
            with st.spinner("Running OCR and analyzing... this may take a moment."):
                is_genuine, prob, num_blocks = predict_receipt(tmp_path)

            if prob is None:
                st.error("Could not extract any text from the receipt.")
            else:
                st.divider()

                prob_genuine = round(prob[0] * 100, 1)
                prob_fake = round(prob[1] * 100, 1)

                if is_genuine:
                    st.success(f"✅ **GENUINE** — {prob_genuine}% confidence")
                else:
                    st.error(f"🚨 **LIKELY FAKE** — {prob_fake}% confidence")

                col1, col2, col3 = st.columns(3)
                col1.metric("Genuine %", f"{prob_genuine}%")
                col2.metric("Fake %", f"{prob_fake}%")
                col3.metric("OCR Blocks", num_blocks)

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
