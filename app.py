"""
Comment Toxicity Detection — Premium Streamlit Web Application
==============================================================
A visually stunning, dark-themed dashboard powered by a pre-trained BiLSTM
model.  Provides real-time single / bulk toxicity prediction, data insights,
and model-performance visualisation.
"""

import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# ── Page configuration (MUST be the first Streamlit command) ────────────────
st.set_page_config(
    page_title="Toxicity Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ───────────────────────────────────────────────────────────────
LABEL_COLUMNS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]
LABEL_DISPLAY = {
    "toxic": "☠️ Toxic",
    "severe_toxic": "💀 Severe Toxic",
    "obscene": "🤬 Obscene",
    "threat": "⚔️ Threat",
    "insult": "🗯️ Insult",
    "identity_hate": "🎭 Identity Hate",
}
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"


# ══════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS — Premium dark glassmorphism theme
# ══════════════════════════════════════════════════════════════════════════════
def inject_custom_css() -> None:
    st.markdown(
        """
    <style>
    /* ── Import font ─────────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* ── Global resets ───────────────────────────────────────────── */
    *, *::before, *::after { box-sizing: border-box; }
    html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(145deg, #0e1117 0%, #1a1a2e 50%, #16213e 100%);
        color: #e0e0e0;
    }
    [data-testid="stHeader"] { background: transparent; }

    /* ── Custom scrollbar ────────────────────────────────────────── */
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: #0e1117; }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #667eea, #764ba2);
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover { background: #8b5cf6; }

    /* ── Sidebar ─────────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29 0%, #302b63 50%, #24243e 100%) !important;
        border-right: 1px solid rgba(102,126,234,0.25);
    }
    [data-testid="stSidebar"] .stRadio label {
        color: #c4c4e0 !important;
        font-weight: 500;
        transition: color 0.3s ease;
    }
    [data-testid="stSidebar"] .stRadio label:hover { color: #fff !important; }

    /* ── Glassmorphism card ───────────────────────────────────────── */
    .glass-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 18px;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        box-shadow: 0 8px 32px rgba(0,0,0,0.35);
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(102,126,234,0.18);
    }

    /* ── Gradient heading ────────────────────────────────────────── */
    .gradient-text {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 800;
    }
    .hero-title {
        font-size: 3.2rem;
        line-height: 1.1;
        margin-bottom: 0;
    }
    .hero-subtitle {
        font-size: 1.15rem;
        color: #9ca3af;
        margin-top: 6px;
        font-weight: 400;
    }

    /* ── Metric card ─────────────────────────────────────────────── */
    .metric-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 22px 20px;
        text-align: center;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.25);
        transition: transform 0.2s ease;
    }
    .metric-card:hover { transform: translateY(-3px); }
    .metric-card .metric-value {
        font-size: 2.4rem;
        font-weight: 800;
        margin: 4px 0;
    }
    .metric-card .metric-label {
        font-size: 0.85rem;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* ── Animated result reveal ──────────────────────────────────── */
    @keyframes fadeSlideUp {
        from { opacity: 0; transform: translateY(24px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .result-anim {
        animation: fadeSlideUp 0.55s cubic-bezier(0.22,1,0.36,1) forwards;
    }

    /* ── Styled button override ──────────────────────────────────── */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.55rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        letter-spacing: 0.3px;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 14px rgba(102,126,234,0.35) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(102,126,234,0.5) !important;
    }
    .stButton > button:active { transform: translateY(0) !important; }

    /* ── Status badge ────────────────────────────────────────────── */
    .badge-pass {
        background: rgba(16,185,129,0.15);
        color: #10b981;
        padding: 3px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.82rem;
    }
    .badge-fail {
        background: rgba(239,68,68,0.15);
        color: #ef4444;
        padding: 3px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.82rem;
    }

    /* ── Divider ─────────────────────────────────────────────────── */
    .styled-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(102,126,234,0.4), transparent);
        border: none;
        margin: 28px 0;
    }

    /* ── Tech badge ───────────────────────────────────────────────── */
    .tech-badge {
        display: inline-block;
        background: rgba(102,126,234,0.12);
        color: #a5b4fc;
        border: 1px solid rgba(102,126,234,0.25);
        border-radius: 20px;
        padding: 6px 16px;
        margin: 4px;
        font-size: 0.85rem;
        font-weight: 500;
    }

    /* ── Example button row ──────────────────────────────────────── */
    .example-btn {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 8px;
        padding: 8px 16px;
        color: #c4c4e0;
        cursor: pointer;
        transition: all 0.2s ease;
        font-size: 0.85rem;
    }
    .example-btn:hover {
        background: rgba(102,126,234,0.15);
        border-color: rgba(102,126,234,0.4);
    }

    /* ── Hide default Streamlit branding ──────────────────────────── */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }

    /* ── Table styling ────────────────────────────────────────────── */
    .stDataFrame { border-radius: 12px; overflow: hidden; }
    </style>
    """,
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# HELPER WIDGETS
# ══════════════════════════════════════════════════════════════════════════════

def render_glass_card(content_html: str) -> None:
    st.markdown(f'<div class="glass-card result-anim">{content_html}</div>', unsafe_allow_html=True)


def render_metric_card(label: str, value: str, color: str = "#667eea") -> str:
    return (
        f'<div class="metric-card" style="border-top:3px solid {color};">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value" style="color:{color};">{value}</div>'
        f"</div>"
    )


def styled_divider() -> None:
    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING (cached)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def cached_load_model():
    """Load model once and cache across reruns."""
    from src.predict import load_model
    return load_model(model_dir=str(MODEL_DIR))


def get_model():
    """Try to load the model; return (model, vocab, config) or None."""
    try:
        return cached_load_model()
    except Exception as exc:
        st.error(f"⚠️ Could not load model: {exc}")
        return None


def show_model_missing_message():
    render_glass_card(
        "<h3 style='color:#f87171;'>🔧 Model Not Found</h3>"
        "<p>The pre-trained model files were not detected in <code>models/</code>.</p>"
        "<p style='color:#9ca3af;'>To train the model, run:</p>"
        "<pre style='background:rgba(0,0,0,0.3);padding:14px;border-radius:8px;"
        "color:#a5b4fc;'>python -m src.train</pre>"
        "<p style='color:#9ca3af;margin-top:8px;'>Expected files: "
        "<code>toxicity_model.pth</code>, <code>vocab.json</code></p>"
    )


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING (cached)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_training_data() -> pd.DataFrame | None:
    path = DATA_DIR / "train.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


# ══════════════════════════════════════════════════════════════════════════════
# PLOTLY THEME HELPER
# ══════════════════════════════════════════════════════════════════════════════
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#d1d5db"),
    margin=dict(l=40, r=40, t=50, b=40),
)

VIBRANT_COLORS = ["#667eea", "#764ba2", "#f093fb", "#4facfe", "#00f2fe", "#43e97b"]
SEVERITY_SCALE = ["#10b981", "#34d399", "#fbbf24", "#f97316", "#ef4444", "#dc2626"]


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: 🏠 Real-Time Detection
# ══════════════════════════════════════════════════════════════════════════════

def page_realtime():
    from src.predict import predict_single, get_toxicity_label, get_severity_color

    # Hero section
    st.markdown(
        '<h1 class="gradient-text hero-title">🛡️ Comment Toxicity Detector</h1>'
        '<p class="hero-subtitle">'
        "Analyse any comment for toxic language using a state-of-the-art BiLSTM deep-learning model. "
        "Paste or type a comment below to get instant predictions across six toxicity categories."
        "</p>",
        unsafe_allow_html=True,
    )
    styled_divider()

    # ── Preset examples ─────────────────────────────────────────────────────
    EXAMPLES = {
        "💚 Clean comment": "Great article! I really enjoyed reading this and learned a lot. Thanks for sharing!",
        "⚠️ Mildly toxic": "This is the stupidest thing I've ever read. You clearly have no idea what you're talking about.",
        "🚨 Toxic comment": "You are an absolute idiot and a waste of space. Shut up and go away, nobody wants you here!",
        "🎭 Identity attack": "All people of that group are terrible and should be removed from this platform permanently.",
    }

    st.markdown("##### 💡 Try a preset example")
    example_cols = st.columns(len(EXAMPLES))
    for idx, (btn_label, btn_text) in enumerate(EXAMPLES.items()):
        with example_cols[idx]:
            if st.button(btn_label, key=f"ex_{idx}", use_container_width=True):
                st.session_state["comment_input"] = btn_text

    st.markdown("")  # spacer

    # ── Text input ───────────────────────────────────────────────────────────
    comment = st.text_area(
        "✍️ Enter a comment to analyse",
        value=st.session_state.get("comment_input", ""),
        height=140,
        placeholder="Type or paste a comment here…",
        key="comment_box",
    )

    analyse_clicked = st.button("🔍  Analyse Comment", use_container_width=False)

    # ── Prediction ───────────────────────────────────────────────────────────
    if analyse_clicked:
        if not comment.strip():
            st.warning("Please enter a comment before analysing.")
            return

        result = get_model()
        if result is None:
            show_model_missing_message()
            return

        model, vocab, config = result

        with st.spinner("Analysing…"):
            predictions = predict_single(comment, model, vocab, config)
            label = get_toxicity_label(predictions)
            overall = max(predictions.values())

        # ── Results ──────────────────────────────────────────────────────────
        st.markdown('<div class="result-anim">', unsafe_allow_html=True)
        styled_divider()

        # Top-level score + label
        score_color = get_severity_color(overall)
        col_score, col_label, col_spacer = st.columns([1, 2, 1])
        with col_score:
            st.markdown(
                render_metric_card("Overall Toxicity", f"{overall:.0%}", score_color),
                unsafe_allow_html=True,
            )
        with col_label:
            label_color_map = {
                "Clean ✅": "#10b981",
                "Toxic ⚠️": "#f59e0b",
                "Highly Toxic 🚨": "#ef4444",
                "Threat Detected ⚔️": "#dc2626",
            }
            lc = label_color_map.get(label, "#667eea")
            st.markdown(
                render_metric_card("Verdict", label, lc),
                unsafe_allow_html=True,
            )

        st.markdown("")  # spacer

        # ── Horizontal bar chart ─────────────────────────────────────────────
        labels = [LABEL_DISPLAY.get(k, k) for k in LABEL_COLUMNS]
        scores = [predictions[k] for k in LABEL_COLUMNS]
        colors = [get_severity_color(s) for s in scores]

        fig = go.Figure(
            go.Bar(
                x=scores,
                y=labels,
                orientation="h",
                marker=dict(color=colors, line=dict(width=0)),
                text=[f"{s:.1%}" for s in scores],
                textposition="auto",
                textfont=dict(color="#fff", size=13, family="Inter"),
            )
        )
        fig.update_layout(
            **PLOTLY_LAYOUT,
            title=dict(text="Per-Label Probability", font=dict(size=18)),
            xaxis=dict(range=[0, 1], tickformat=".0%", showgrid=True,
                        gridcolor="rgba(255,255,255,0.06)"),
            yaxis=dict(autorange="reversed"),
            height=340,
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Severity table ───────────────────────────────────────────────────
        st.markdown("##### 📋 Detailed Scores")
        table_rows = ""
        for col in LABEL_COLUMNS:
            s = predictions[col]
            sc = get_severity_color(s)
            status = (
                '<span class="badge-fail">FAIL</span>'
                if s >= 0.5
                else '<span class="badge-pass">PASS</span>'
            )
            bar_pct = s * 100
            table_rows += (
                f"<tr>"
                f"<td style='padding:10px 14px;font-weight:500;'>{LABEL_DISPLAY[col]}</td>"
                f"<td style='padding:10px 14px;'>"
                f"  <div style='background:rgba(255,255,255,0.06);border-radius:6px;overflow:hidden;height:10px;width:100%;'>"
                f"    <div style='width:{bar_pct}%;height:100%;background:{sc};border-radius:6px;'></div>"
                f"  </div>"
                f"</td>"
                f"<td style='padding:10px 14px;text-align:center;color:{sc};font-weight:700;'>{s:.3f}</td>"
                f"<td style='padding:10px 14px;text-align:center;'>{status}</td>"
                f"</tr>"
            )

        render_glass_card(
            "<table style='width:100%;border-collapse:collapse;'>"
            "<thead><tr style='border-bottom:1px solid rgba(255,255,255,0.1);'>"
            "<th style='text-align:left;padding:10px 14px;color:#9ca3af;'>Category</th>"
            "<th style='text-align:left;padding:10px 14px;color:#9ca3af;width:35%;'>Bar</th>"
            "<th style='text-align:center;padding:10px 14px;color:#9ca3af;'>Score</th>"
            "<th style='text-align:center;padding:10px 14px;color:#9ca3af;'>Status</th>"
            "</tr></thead>"
            f"<tbody>{table_rows}</tbody></table>"
        )

        st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: 📊 Data Insights
# ══════════════════════════════════════════════════════════════════════════════

def page_data_insights():
    st.markdown(
        '<h1 class="gradient-text hero-title">📊 Data Insights</h1>'
        '<p class="hero-subtitle">Explore the training dataset and understand the distribution of toxic comments.</p>',
        unsafe_allow_html=True,
    )
    styled_divider()

    df = load_training_data()
    if df is None:
        render_glass_card(
            "<h3 style='color:#f87171;'>📂 Dataset Not Found</h3>"
            "<p>Could not locate <code>data/train.csv</code>. "
            "Please ensure the training data file is in the <code>data/</code> directory.</p>"
        )
        return

    # ── Top-level metrics ────────────────────────────────────────────────────
    total = len(df)
    has_any_toxic = (df[LABEL_COLUMNS].sum(axis=1) > 0).sum()
    clean_count = total - has_any_toxic
    toxic_pct = has_any_toxic / total * 100

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(render_metric_card("Total Comments", f"{total:,}", "#667eea"), unsafe_allow_html=True)
    with m2:
        st.markdown(render_metric_card("Toxic Comments", f"{has_any_toxic:,}", "#ef4444"), unsafe_allow_html=True)
    with m3:
        st.markdown(render_metric_card("Clean Comments", f"{clean_count:,}", "#10b981"), unsafe_allow_html=True)
    with m4:
        st.markdown(render_metric_card("Toxic Ratio", f"{toxic_pct:.1f}%", "#f59e0b"), unsafe_allow_html=True)

    styled_divider()

    # ── Label distribution bar chart ─────────────────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        label_counts = df[LABEL_COLUMNS].sum().sort_values(ascending=True)
        fig_lbl = go.Figure(
            go.Bar(
                x=label_counts.values,
                y=[LABEL_DISPLAY.get(c, c) for c in label_counts.index],
                orientation="h",
                marker=dict(
                    color=VIBRANT_COLORS[: len(label_counts)],
                    line=dict(width=0),
                ),
                text=[f"{v:,}" for v in label_counts.values],
                textposition="auto",
                textfont=dict(color="#fff", size=12),
            )
        )
        fig_lbl.update_layout(
            **PLOTLY_LAYOUT,
            title=dict(text="Label Distribution", font=dict(size=17)),
            height=380,
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
        )
        st.plotly_chart(fig_lbl, use_container_width=True)

    # ── Comment-length histogram ─────────────────────────────────────────────
    with col_b:
        if "comment_text" in df.columns:
            lengths = df["comment_text"].astype(str).str.len()
        else:
            first_text_col = df.select_dtypes(include="object").columns[0]
            lengths = df[first_text_col].astype(str).str.len()

        fig_hist = go.Figure(
            go.Histogram(
                x=lengths.clip(upper=2000),
                nbinsx=60,
                marker=dict(
                    color="rgba(102,126,234,0.55)",
                    line=dict(color="#667eea", width=1),
                ),
            )
        )
        fig_hist.update_layout(
            **PLOTLY_LAYOUT,
            title=dict(text="Comment Length Distribution", font=dict(size=17)),
            xaxis_title="Character Count",
            yaxis_title="Frequency",
            height=380,
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    styled_divider()

    # ── Correlation heatmap ──────────────────────────────────────────────────
    st.markdown("##### 🔗 Label Correlation Matrix")
    corr = df[LABEL_COLUMNS].corr()
    display_labels = [LABEL_DISPLAY.get(c, c) for c in LABEL_COLUMNS]

    fig_corr = go.Figure(
        go.Heatmap(
            z=corr.values,
            x=display_labels,
            y=display_labels,
            colorscale=[
                [0, "#0e1117"],
                [0.25, "#302b63"],
                [0.5, "#667eea"],
                [0.75, "#f093fb"],
                [1, "#ef4444"],
            ],
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            textfont=dict(size=12, color="#fff"),
            zmin=0,
            zmax=1,
            colorbar=dict(
                title="Corr",
                tickfont=dict(color="#9ca3af"),
                titlefont=dict(color="#9ca3af"),
            ),
        )
    )
    fig_corr.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Toxicity Label Correlations", font=dict(size=17)),
        height=480,
        xaxis=dict(tickangle=-35),
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    # ── Toxic vs Clean pie chart ─────────────────────────────────────────────
    st.markdown("##### 🍩 Toxic vs Clean Breakdown")
    fig_pie = go.Figure(
        go.Pie(
            labels=["Clean", "Toxic"],
            values=[clean_count, has_any_toxic],
            hole=0.55,
            marker=dict(colors=["#10b981", "#ef4444"], line=dict(color="#1a1a2e", width=3)),
            textfont=dict(size=14, color="#fff"),
            textinfo="label+percent",
        )
    )
    fig_pie.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Overall Composition", font=dict(size=17)),
        height=380,
        showlegend=False,
    )
    st.plotly_chart(fig_pie, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: 📈 Model Performance
# ══════════════════════════════════════════════════════════════════════════════

def page_model_performance():
    st.markdown(
        '<h1 class="gradient-text hero-title">📈 Model Performance</h1>'
        '<p class="hero-subtitle">Review evaluation metrics, confusion matrices, and ROC curves generated during training.</p>',
        unsafe_allow_html=True,
    )
    styled_divider()

    # ── Evaluation metrics JSON ──────────────────────────────────────────────
    metrics_path = MODEL_DIR / "evaluation_results.json"
    if metrics_path.exists():
        try:
            with open(metrics_path, "r") as f:
                metrics = json.load(f)

            # Per-label table
            st.markdown("##### 📋 Per-Label Metrics")

            table_data = []
            for lbl in LABEL_COLUMNS:
                lbl_metrics = metrics.get(lbl, metrics.get("per_label", {}).get(lbl, {}))
                if lbl_metrics:
                    table_data.append({
                        "Category": LABEL_DISPLAY.get(lbl, lbl),
                        "AUC-ROC": f"{lbl_metrics.get('auc_roc', lbl_metrics.get('roc_auc', 0)):.4f}",
                        "F1 Score": f"{lbl_metrics.get('f1', lbl_metrics.get('f1_score', 0)):.4f}",
                        "Precision": f"{lbl_metrics.get('precision', 0):.4f}",
                        "Recall": f"{lbl_metrics.get('recall', 0):.4f}",
                        "Accuracy": f"{lbl_metrics.get('accuracy', 0):.4f}",
                    })

            if table_data:
                metrics_df = pd.DataFrame(table_data)
                st.dataframe(
                    metrics_df,
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                # Try to display raw JSON in a nice card
                render_glass_card(
                    "<h4>Raw Evaluation Results</h4>"
                    f"<pre style='color:#a5b4fc;'>{json.dumps(metrics, indent=2)}</pre>"
                )

            # Overall metrics if present
            overall = metrics.get("overall", metrics.get("mean", {}))
            if overall:
                styled_divider()
                st.markdown("##### 🎯 Overall Metrics")
                ov_cols = st.columns(4)
                metric_keys = [
                    ("Mean AUC-ROC", "mean_auc_roc", "auc_roc", "roc_auc", "mean_roc_auc"),
                    ("Mean F1", "mean_f1", "f1", "f1_score", "mean_f1_score"),
                    ("Mean Precision", "mean_precision", "precision", "precision", "mean_precision"),
                    ("Mean Recall", "mean_recall", "recall", "recall", "mean_recall"),
                ]
                for i, keys in enumerate(metric_keys):
                    display_name = keys[0]
                    val = 0
                    for k in keys[1:]:
                        val = overall.get(k, 0)
                        if val:
                            break
                    with ov_cols[i]:
                        st.markdown(
                            render_metric_card(display_name, f"{val:.4f}", VIBRANT_COLORS[i]),
                            unsafe_allow_html=True,
                        )
        except Exception as exc:
            st.warning(f"Could not parse evaluation_results.json: {exc}")
    else:
        render_glass_card(
            "<p style='color:#9ca3af;'>📄 <code>models/evaluation_results.json</code> not found. "
            "Run evaluation after training to generate metrics.</p>"
        )

    styled_divider()

    # ── Visualisation images ─────────────────────────────────────────────────
    image_items = [
        ("Confusion Matrices", "confusion_matrices.png"),
        ("ROC Curves", "roc_curves.png"),
        ("Training History", "training_history.png"),
    ]

    for title, fname in image_items:
        img_path = MODEL_DIR / fname
        if img_path.exists():
            st.markdown(f"##### 🖼️ {title}")
            st.image(str(img_path), use_container_width=True)
            st.markdown("")
        else:
            render_glass_card(
                f"<p style='color:#9ca3af;'>🖼️ <code>models/{fname}</code> not found. "
                f"This image is generated during training/evaluation.</p>"
            )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: 📁 Bulk Prediction
# ══════════════════════════════════════════════════════════════════════════════

def page_bulk_prediction():
    from src.predict import predict_batch

    st.markdown(
        '<h1 class="gradient-text hero-title">📁 Bulk Prediction</h1>'
        '<p class="hero-subtitle">Upload a CSV file containing comments and get toxicity predictions for every row.</p>',
        unsafe_allow_html=True,
    )
    styled_divider()

    result = get_model()
    if result is None:
        show_model_missing_message()
        return

    model, vocab, config = result

    uploaded = st.file_uploader(
        "📤 Upload CSV file",
        type=["csv"],
        help="The file should contain a text column with comments.",
    )

    if uploaded is None:
        render_glass_card(
            "<p style='color:#9ca3af;text-align:center;padding:30px 0;'>"
            "⬆️ Upload a CSV to get started. The file must contain a column with comment text."
            "</p>"
        )
        return

    try:
        df = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Failed to read CSV: {exc}")
        return

    if df.empty:
        st.warning("The uploaded CSV is empty.")
        return

    st.markdown(f"✅ Loaded **{len(df):,}** rows &nbsp;|&nbsp; Columns: `{'`, `'.join(df.columns)}`")

    # Column selection
    text_cols = df.select_dtypes(include="object").columns.tolist()
    default_col = "comment_text" if "comment_text" in text_cols else (text_cols[0] if text_cols else None)

    if not text_cols:
        st.error("No text columns found in the uploaded CSV.")
        return

    selected_col = st.selectbox("Select the text column", text_cols, index=text_cols.index(default_col) if default_col else 0)

    if st.button("🚀  Run Bulk Prediction", use_container_width=True):
        texts = df[selected_col].astype(str).tolist()
        progress = st.progress(0, text="Processing comments…")

        # Process in chunks for progress display
        chunk_size = max(1, len(texts) // 20)
        all_chunks = [texts[i : i + chunk_size] for i in range(0, len(texts), chunk_size)]
        result_frames = []

        for idx, chunk in enumerate(all_chunks):
            chunk_df = predict_batch(chunk, model, vocab, config)
            result_frames.append(chunk_df)
            progress.progress((idx + 1) / len(all_chunks), text=f"Processing… {(idx + 1) * chunk_size}/{len(texts)}")

        progress.empty()

        results_df = pd.concat(result_frames, ignore_index=True)
        st.session_state["bulk_results"] = results_df

    # ── Display results ──────────────────────────────────────────────────────
    if "bulk_results" in st.session_state:
        results_df = st.session_state["bulk_results"]
        styled_divider()

        # Summary metrics
        total = len(results_df)
        if "overall_toxicity" in results_df.columns:
            toxic_count = (results_df["overall_toxicity"] >= 0.5).sum()
        else:
            toxic_count = (results_df[LABEL_COLUMNS].max(axis=1) >= 0.5).sum()
        clean_count = total - toxic_count
        toxic_pct = toxic_count / total * 100

        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.markdown(render_metric_card("Total", f"{total:,}", "#667eea"), unsafe_allow_html=True)
        with s2:
            st.markdown(render_metric_card("Toxic", f"{toxic_count:,}", "#ef4444"), unsafe_allow_html=True)
        with s3:
            st.markdown(render_metric_card("Clean", f"{clean_count:,}", "#10b981"), unsafe_allow_html=True)
        with s4:
            st.markdown(render_metric_card("Toxic %", f"{toxic_pct:.1f}%", "#f59e0b"), unsafe_allow_html=True)

        # Distribution chart
        if "label" in results_df.columns:
            label_dist = results_df["label"].value_counts()
            fig_dist = go.Figure(
                go.Bar(
                    x=label_dist.index.tolist(),
                    y=label_dist.values.tolist(),
                    marker=dict(
                        color=["#10b981", "#f59e0b", "#ef4444"][: len(label_dist)],
                        line=dict(width=0),
                    ),
                    text=label_dist.values.tolist(),
                    textposition="auto",
                    textfont=dict(color="#fff", size=13),
                )
            )
            fig_dist.update_layout(
                **PLOTLY_LAYOUT,
                title=dict(text="Prediction Distribution", font=dict(size=17)),
                height=340,
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
            )
            st.plotly_chart(fig_dist, use_container_width=True)

        # Dataframe display
        st.markdown("##### 📄 Results Table")
        st.dataframe(results_df, use_container_width=True, height=420)

        # Download
        csv_bytes = results_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️  Download Results CSV",
            data=csv_bytes,
            file_name="toxicity_predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ℹ️ About
# ══════════════════════════════════════════════════════════════════════════════

def page_about():
    st.markdown(
        '<h1 class="gradient-text hero-title">ℹ️ About This Project</h1>'
        '<p class="hero-subtitle">Understanding the architecture, tech stack, and usage of the Comment Toxicity Detector.</p>',
        unsafe_allow_html=True,
    )
    styled_divider()

    # ── Project overview ─────────────────────────────────────────────────────
    render_glass_card(
        "<h3 style='margin-top:0;'>🎯 Project Overview</h3>"
        "<p>The <strong>Comment Toxicity Detector</strong> is a deep-learning application that "
        "classifies user-generated comments into six categories of toxicity:</p>"
        "<ul>"
        "<li><strong>Toxic</strong> — generally rude or disrespectful</li>"
        "<li><strong>Severe Toxic</strong> — extremely hateful or aggressive</li>"
        "<li><strong>Obscene</strong> — contains obscene language</li>"
        "<li><strong>Threat</strong> — contains threatening language</li>"
        "<li><strong>Insult</strong> — insulting or demeaning</li>"
        "<li><strong>Identity Hate</strong> — attacks based on identity</li>"
        "</ul>"
        "<p style='color:#9ca3af;'>This is a <em>multi-label classification</em> task — "
        "a single comment may belong to multiple categories simultaneously.</p>"
    )

    # ── Architecture diagram ─────────────────────────────────────────────────
    st.markdown("##### 🏗️ Model Architecture")
    render_glass_card(
        "<pre style='color:#a5b4fc;font-size:0.95rem;line-height:1.7;text-align:center;"
        "background:rgba(0,0,0,0.2);padding:20px;border-radius:10px;overflow-x:auto;'>"
        "┌─────────────┐    ┌──────────────┐    ┌───────────────┐    ┌──────────────┐    ┌────────────┐\n"
        "│  Raw Text   │───▶│  Tokeniser   │───▶│   Embedding   │───▶│   BiLSTM     │───▶│  FC Layers │\n"
        "│  (Comment)  │    │  + Padding   │    │  (128-dim)    │    │  (2 layers)  │    │  + Sigmoid │\n"
        "└─────────────┘    └──────────────┘    └───────────────┘    └──────────────┘    └────────────┘\n"
        "                                                                  │\n"
        "                                                           ┌──────┴──────┐\n"
        "                                                           │   Dropout   │\n"
        "                                                           │   (0.3)     │\n"
        "                                                           └─────────────┘\n"
        "</pre>"
    )

    # ── Tech stack ───────────────────────────────────────────────────────────
    st.markdown("##### 🛠️ Tech Stack")
    badges = [
        "🐍 Python 3.10+",
        "🔥 PyTorch",
        "🎈 Streamlit 1.54",
        "📊 Plotly",
        "🐼 Pandas",
        "🔢 NumPy",
        "📝 NLTK / Regex",
        "🧠 BiLSTM",
    ]
    badge_html = " ".join(f'<span class="tech-badge">{b}</span>' for b in badges)
    st.markdown(f'<div style="margin:10px 0 20px;">{badge_html}</div>', unsafe_allow_html=True)

    # ── Usage ────────────────────────────────────────────────────────────────
    render_glass_card(
        "<h3 style='margin-top:0;'>🚀 Getting Started</h3>"
        "<ol style='line-height:2;'>"
        "<li>Install dependencies: <code>pip install -r requirements.txt</code></li>"
        "<li>Train the model: <code>python -m src.train</code></li>"
        "<li>Launch the app: <code>streamlit run app.py</code></li>"
        "</ol>"
        "<p style='color:#9ca3af;'>The model trains on the Jigsaw Toxic Comment dataset and saves "
        "weights to the <code>models/</code> directory.</p>"
    )

    # ── GitHub ───────────────────────────────────────────────────────────────
    render_glass_card(
        "<h3 style='margin-top:0;'>🔗 Links</h3>"
        "<p>"
        "📂 <a href='#' style='color:#667eea;text-decoration:none;'>GitHub Repository</a> &nbsp;|&nbsp; "
        "📄 <a href='https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge' "
        "style='color:#667eea;text-decoration:none;' target='_blank'>Kaggle Competition</a>"
        "</p>"
    )

    # ── Footer ───────────────────────────────────────────────────────────────
    styled_divider()
    st.markdown(
        "<p style='text-align:center;color:#6b7280;font-size:0.85rem;'>"
        "Built with ❤️ using Streamlit &nbsp;•&nbsp; © 2026"
        "</p>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR & NAVIGATION
# ══════════════════════════════════════════════════════════════════════════════

def main():
    inject_custom_css()

    with st.sidebar:
        st.markdown(
            '<div style="text-align:center;padding:16px 0 8px;">'
            '<span style="font-size:2.4rem;">🛡️</span><br>'
            '<span class="gradient-text" style="font-size:1.3rem;font-weight:700;">'
            "Toxicity Detector</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

        page = st.radio(
            "Navigation",
            [
                "🏠 Real-Time Detection",
                "📊 Data Insights",
                "📈 Model Performance",
                "📁 Bulk Prediction",
                "ℹ️ About",
            ],
            label_visibility="collapsed",
        )

        st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

        # Sidebar footer
        st.markdown(
            "<div style='position:fixed;bottom:20px;padding:0 16px;'>"
            "<p style='color:#6b7280;font-size:0.75rem;'>"
            "🛡️ Toxicity Detector v1.0<br>"
            "Powered by BiLSTM + PyTorch"
            "</p></div>",
            unsafe_allow_html=True,
        )

    # Route to selected page
    pages = {
        "🏠 Real-Time Detection": page_realtime,
        "📊 Data Insights": page_data_insights,
        "📈 Model Performance": page_model_performance,
        "📁 Bulk Prediction": page_bulk_prediction,
        "ℹ️ About": page_about,
    }

    pages[page]()


if __name__ == "__main__":
    main()
