import os
import tempfile

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
from PIL import Image

from client.benchmark import run_benchmark
from client.cosine_similarity import calculate_similarity, find_best_matches
from client.metrics import compute_metrics_at_threshold
from preprocess.preprocessor import preprocess_image
from self_supervised.model import PalmprintEncoder

st.set_page_config(page_title="Palmprint Recognition", layout="wide")

st.markdown(
    """
    <style>
    .metric-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        border: 1px solid #e0e0e0;
    }
    .metric-label { font-size: 13px; color: #666; margin-bottom: 4px; font-weight: 600; }
    .metric-value { font-size: 28px; font-weight: 700; }
    .metric-sub   { font-size: 11px; color: #999; margin-top: 4px; }
    .score-text   { color: #2e7d32; font-weight: bold; font-size: 16px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── cached resource / data loaders ───────────────────────────────────────────

@st.cache_resource
def load_encoder():
    enc = PalmprintEncoder().encoder
    enc.load_state_dict(
        torch.load(
            "output/model/palmprint_encoder.pth",
            map_location=torch.device("cpu"),
            weights_only=True,
        )
    )
    enc.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc.to(device)
    return enc, device


@st.cache_data
def load_database():
    embeddings  = np.load("output/embeddings/all_embeddings.npy")
    names       = np.load("output/embeddings/image_names.npy").tolist()
    return embeddings, names


@st.cache_data(show_spinner=False)
def get_benchmark_data() -> dict:
    """
    Runs the full benchmark once and caches the result for the session.
    Returns raw genuine / impostor scores alongside curve data so the UI
    can recompute point metrics interactively when the threshold slider changes.
    """
    return run_benchmark()


# ── embed a single uploaded image ────────────────────────────────────────────

def embed_image(preprocessed_roi: np.ndarray, encoder, device) -> np.ndarray:
    roi = np.expand_dims(preprocessed_roi, axis=0)
    tensor = torch.from_numpy(roi).float().repeat(1, 3, 1, 1).to(device)
    with torch.no_grad():
        return encoder(tensor).cpu().numpy().flatten()


# ── plotting helpers ──────────────────────────────────────────────────────────

def _far_frr_plot(curve: dict, threshold: float, eer_thr: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(curve["thresholds"], curve["FAR"], color="#e53935", linewidth=2, label="FAR")
    ax.plot(curve["thresholds"], curve["FRR"], color="#1e88e5", linewidth=2, label="FRR")
    ax.axvline(eer_thr,   color="#757575", linestyle="--", linewidth=1.5,
               label=f"EER  @ {eer_thr:.3f}")
    ax.axvline(threshold, color="#fb8c00", linestyle="--", linewidth=1.5,
               label=f"Operating threshold @ {threshold:.2f}")
    ax.set_xlabel("Threshold", fontsize=11)
    ax.set_ylabel("Rate (%)", fontsize=11)
    ax.set_title("FAR and FRR vs. Decision Threshold", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25)
    ax.set_xlim(curve["thresholds"].min(), curve["thresholds"].max())
    ax.set_ylim(-2, 102)
    fig.tight_layout()
    return fig


def _score_distribution_plot(genuine: np.ndarray, impostor: np.ndarray, threshold: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(genuine,  bins=60, alpha=0.65, color="#43a047", label="Genuine pairs")
    ax.hist(impostor, bins=60, alpha=0.65, color="#e53935", label="Impostor pairs")
    ax.axvline(threshold, color="#fb8c00", linestyle="--", linewidth=2,
               label=f"Threshold ({threshold:.2f})")
    ax.set_xlabel("Cosine Similarity Score", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Score Distribution: Genuine vs. Impostor Pairs", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


# ── metric card HTML helper ───────────────────────────────────────────────────

def _metric_card(label: str, value: str, sub: str = "", color: str = "#1a1a1a") -> str:
    return (
        f"<div class='metric-card'>"
        f"<div class='metric-label'>{label}</div>"
        f"<div class='metric-value' style='color:{color}'>{value}</div>"
        f"<div class='metric-sub'>{sub}</div>"
        f"</div>"
    )


# ── app ───────────────────────────────────────────────────────────────────────

st.title("Palmprint Recognition")

encoder, device         = load_encoder()
all_embeddings, img_names = load_database()

tab_recognition, tab_benchmark = st.tabs(["🔍 Recognition", "📊 Benchmark"])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Recognition  (original functionality, unchanged)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_recognition:
    uploaded_file = st.file_uploader(
        "Choose a palmprint image…", type=["jpg", "jpeg", "png", "tiff"]
    )

    if uploaded_file is not None:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".tiff") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            image           = Image.open(uploaded_file).convert("RGB")
            image_np        = np.array(image)
            preprocessed    = preprocess_image(tmp_path)
            os.remove(tmp_path)

            if preprocessed is None:
                st.error("Could not preprocess the uploaded image. Please check the image quality.")
            else:
                col1, col2 = st.columns([6.5, 4])

                with col1:
                    st.subheader("Uploaded Image")
                    st.image(image_np, use_container_width=True)

                with col2:
                    query_emb         = embed_image(preprocessed, encoder, device)
                    similarity_scores = calculate_similarity(query_emb, all_embeddings)
                    best_matches, match_found = find_best_matches(
                        similarity_scores, img_names, threshold=0.61
                    )

                    st.subheader("Matching Results")
                    if match_found:
                        st.success("Match found!")
                    else:
                        st.warning("No match found above the threshold.")

                    for i in range(0, len(best_matches), 3):
                        row = best_matches[i : i + 3]
                        cols = st.columns(3)
                        for col, (name, score) in zip(cols, row):
                            img_path = os.path.join(
                                "dataset/preprocessed_images",
                                name.replace(".npy", "") + ".npy",
                            )
                            matched_img = np.load(img_path)
                            with col:
                                st.image(matched_img, use_container_width=True)
                                st.markdown(
                                    f"<div style='color:#2e7d32;font-weight:bold'>"
                                    f"Score: {score * 100:.2f}%</div>",
                                    unsafe_allow_html=True,
                                )
                                st.write(f"**Image:** {name}")

        except Exception as exc:
            st.error(f"An error occurred: {exc}")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Benchmark
# ═══════════════════════════════════════════════════════════════════════════════
with tab_benchmark:
    st.subheader("System-wide Benchmark Metrics")
    st.caption(
        "Metrics are computed over genuine pairs (same identity) and impostor pairs "
        "(different identities) drawn from the full embedding database."
    )

    try:
        with st.spinner("Computing benchmark — building pairs and scoring embeddings…"):
            bm = get_benchmark_data()

        # ── threshold slider ───────────────────────────────────────────────────
        st.markdown("#### Decision Threshold")
        threshold = st.slider(
            "Adjust the threshold to see how FAR, FRR, and Valid Accuracy change.",
            min_value=0.0, max_value=1.0,
            value=0.8, step=0.01,
            key="benchmark_threshold",
        )

        # Recompute point metrics for the selected threshold from cached scores
        pt = compute_metrics_at_threshold(
            bm["genuine_scores"], bm["impostor_scores"], threshold
        )

        # ── metric cards ───────────────────────────────────────────────────────
        st.markdown("#### Metrics at Selected Threshold")
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(
            _metric_card("✅ Valid Accuracy",
                         f"{pt['Valid_Accuracy']:.2f}%",
                         "True accepted / total genuine",
                         "#2e7d32"),
            unsafe_allow_html=True,
        )
        c2.markdown(
            _metric_card("🔴 FAR",
                         f"{pt['FAR']:.2f}%",
                         "False accepted / total impostor",
                         "#c62828"),
            unsafe_allow_html=True,
        )
        c3.markdown(
            _metric_card("🔵 FRR",
                         f"{pt['FRR']:.2f}%",
                         "False rejected / total genuine",
                         "#1565c0"),
            unsafe_allow_html=True,
        )
        c4.markdown(
            _metric_card("⚖️ EER",
                         f"{bm['EER']:.2f}%",
                         f"@ threshold {bm['EER_threshold']:.3f}",
                         "#6a1b9a"),
            unsafe_allow_html=True,
        )

        st.divider()

        # ── dataset summary ────────────────────────────────────────────────────
        st.markdown("#### Evaluation Set Summary")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Genuine Pairs",   f"{pt['total_genuine']:,}")
        s2.metric("Impostor Pairs",  f"{pt['total_impostors']:,}")
        s3.metric("True Acceptances (TA)", f"{pt['true_acceptances']:,}")
        s4.metric("True Rejections (TR)",  f"{pt['true_rejections']:,}")

        ia1, ia2 = st.columns(2)
        ia1.metric("False Acceptances (FA)", f"{pt['false_acceptances']:,}",
                   delta=f"FAR = {pt['FAR']:.2f}%", delta_color="inverse")
        ia2.metric("False Rejections (FR)",  f"{pt['false_rejections']:,}",
                   delta=f"FRR = {pt['FRR']:.2f}%", delta_color="inverse")

        st.divider()

        # ── FAR / FRR curve ────────────────────────────────────────────────────
        st.markdown("#### FAR / FRR vs. Decision Threshold")
        curve = {
            "thresholds": bm["curve_thresholds"],
            "FAR":        bm["curve_FAR"],
            "FRR":        bm["curve_FRR"],
        }
        fig1 = _far_frr_plot(curve, threshold, bm["EER_threshold"])
        st.pyplot(fig1, use_container_width=True)
        plt.close(fig1)

        st.divider()

        # ── score distribution ─────────────────────────────────────────────────
        st.markdown("#### Score Distribution")
        st.caption(
            "A well-separated distribution (green peak near 1, red peak near 0) "
            "indicates strong discriminative power."
        )
        fig2 = _score_distribution_plot(bm["genuine_scores"], bm["impostor_scores"], threshold)
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

        # ── formula reference ──────────────────────────────────────────────────
        with st.expander("📐 Metric definitions"):
            st.markdown(
                """
| Metric | Formula | Description |
|--------|---------|-------------|
| **Valid Accuracy** | (TA / total genuine) × 100 | % of genuine attempts correctly accepted |
| **FAR** | (FA / total impostor) × 100 | % of impostor attempts incorrectly accepted |
| **FRR** | (FR / total genuine) × 100 | % of genuine attempts incorrectly rejected |
| **EER** | (FAR + FRR) / 2 at crossover | Threshold-independent system error rate |

*TA = True Acceptances, FA = False Acceptances, FR = False Rejections*
                """
            )

    except FileNotFoundError:
        st.warning(
            "Embeddings not found. "
            "Run `client/extract_features.py` first to generate `output/embeddings/`."
        )
    except Exception as exc:
        st.error(f"Benchmark error: {exc}")
