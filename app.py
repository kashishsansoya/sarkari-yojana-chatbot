import streamlit as st
import pandas as pd
from langdetect import detect
from deep_translator import GoogleTranslator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import speech_recognition as sr
from streamlit_mic_recorder import speech_to_text

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="सरकारी योजना सहायक",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ──────────────────────────────────────────────
# SESSION STATE INIT
# ──────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "query" not in st.session_state:
    st.session_state.query = ""

# ──────────────────────────────────────────────
# CUSTOM CSS — Modern India-inspired Design
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Devanagari:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700;800&family=Playfair+Display:wght@700;800&display=swap');

/* ── Root Variables ── */
:root {
  --saffron:      #FF6B2B;
  --saffron-lt:   #FFF0E8;
  --saffron-mid:  #FF8C55;
  --deep-green:   #0F3D2B;
  --leaf:         #1A5C3A;
  --leaf-lt:      #E8F5EE;
  --gold:         #F5A623;
  --gold-lt:      #FFF8E6;
  --cream:        #FDFAF5;
  --sand:         #F7F0E3;
  --white:        #FFFFFF;
  --charcoal:     #1A1A2E;
  --body-text:    #374151;
  --muted:        #6B7280;
  --lighter:      #9CA3AF;
  --border:       #E5D9C6;
  --border-light: #F0E8D8;
  --success:      #16A34A;
  --blue-pill:    #EEF2FF;
  --blue-accent:  #4F46E5;
  --pink-accent:  #DB2777;
  --shadow-sm:    0 2px 8px rgba(0,0,0,0.06);
  --shadow-md:    0 6px 24px rgba(0,0,0,0.08);
  --shadow-lg:    0 12px 48px rgba(0,0,0,0.12);
  --radius-sm:    10px;
  --radius-md:    16px;
  --radius-lg:    24px;
  --radius-xl:    32px;
}

/* ── Global Reset ── */
html, body, [class*="css"] {
    font-family: 'Inter', 'Noto Sans Devanagari', sans-serif;
    background-color: var(--cream) !important;
    color: var(--charcoal);
}
.stApp {
    background: linear-gradient(145deg, #FDFAF5 0%, #F7F0E3 60%, #F0E8D8 100%) !important;
}

/* ── Hide Streamlit Chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding-top: 1.2rem !important;
    padding-bottom: 2rem !important;
    max-width: 1080px !important;
}

/* ═══════════════════════════════════════
   HERO HEADER
═══════════════════════════════════════ */
.hero-outer {
    position: relative;
    border-radius: var(--radius-xl);
    overflow: hidden;
    margin-bottom: 1.6rem;
    box-shadow: 0 16px 60px rgba(15,61,43,0.28);
}
.hero-bg {
    background: linear-gradient(135deg, #0F3D2B 0%, #1A5C3A 45%, #2D8653 80%, #3AA876 100%);
    padding: 2.8rem 2.6rem 2.2rem;
    position: relative;
}
/* Decorative Ashoka Chakra watermark */
.hero-bg::before {
    content: "⊕";
    position: absolute;
    font-size: 260px;
    right: -20px;
    top: -60px;
    opacity: 0.04;
    line-height: 1;
    color: #fff;
}
/* Tricolor accent bar at bottom */
.hero-tricolor {
    display: flex;
    height: 6px;
}
.tricolor-saffron { flex: 1; background: #FF9933; }
.tricolor-white   { flex: 1; background: #FFFFFF; }
.tricolor-green   { flex: 1; background: #138808; }

.hero-eyebrow {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: rgba(255,255,255,0.55);
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.hero-eyebrow::before {
    content: "";
    display: inline-block;
    width: 24px;
    height: 2px;
    background: var(--gold);
    border-radius: 2px;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.8rem;
    font-weight: 800;
    color: #FFFFFF;
    margin: 0 0 0.3rem 0;
    line-height: 1.1;
    letter-spacing: -0.5px;
}
.hero-title .accent { color: var(--gold); }
.hero-sub {
    color: rgba(255,255,255,0.7);
    font-size: 1rem;
    margin: 0 0 1.6rem 0;
    font-weight: 400;
    max-width: 520px;
    line-height: 1.6;
}
.hero-stats {
    display: flex;
    gap: 2rem;
    flex-wrap: wrap;
}
.hero-stat {
    text-align: center;
}
.hero-stat-num {
    font-size: 1.5rem;
    font-weight: 800;
    color: var(--gold);
    display: block;
    line-height: 1;
}
.hero-stat-label {
    font-size: 0.72rem;
    color: rgba(255,255,255,0.55);
    font-weight: 500;
    letter-spacing: 0.5px;
    margin-top: 4px;
    display: block;
}
.hero-badges-row {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-top: 1.4rem;
}
.hero-badge {
    background: rgba(255,255,255,0.1);
    border: 1px solid rgba(255,255,255,0.2);
    color: rgba(255,255,255,0.9);
    padding: 5px 14px;
    border-radius: 50px;
    font-size: 0.8rem;
    font-weight: 500;
    backdrop-filter: blur(6px);
    display: flex;
    align-items: center;
    gap: 5px;
}

/* ═══════════════════════════════════════
   CATEGORY QUICK-SELECT
═══════════════════════════════════════ */
.cat-section-label {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.7rem;
}
.cat-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 10px;
    margin-bottom: 1.6rem;
}
.cat-card {
    background: var(--white);
    border: 2px solid var(--border-light);
    border-radius: var(--radius-md);
    padding: 14px 10px;
    text-align: center;
    cursor: pointer;
    transition: all 0.22s ease;
    box-shadow: var(--shadow-sm);
    text-decoration: none;
}
.cat-card:hover {
    border-color: var(--saffron);
    background: var(--saffron-lt);
    transform: translateY(-3px);
    box-shadow: 0 8px 24px rgba(255,107,43,0.15);
}
.cat-icon { font-size: 1.6rem; display: block; margin-bottom: 6px; }
.cat-name {
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--charcoal);
    line-height: 1.3;
}
.cat-hint { font-size: 0.68rem; color: var(--muted); margin-top: 2px; }

/* ═══════════════════════════════════════
   INPUT SECTION
═══════════════════════════════════════ */
.input-section {
    background: var(--white);
    border-radius: var(--radius-lg);
    padding: 1.8rem 2rem;
    border: 2px solid var(--border-light);
    box-shadow: var(--shadow-md);
    margin-bottom: 1.6rem;
    position: relative;
}
.input-section::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 4px;
    border-radius: var(--radius-lg) var(--radius-lg) 0 0;
    background: linear-gradient(90deg, #FF6B2B, #F5A623, #138808);
}
.input-label-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 0.8rem;
}
.input-label-icon {
    width: 32px; height: 32px;
    background: var(--leaf-lt);
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem;
}
.input-label-text {
    font-size: 0.85rem;
    font-weight: 700;
    color: var(--charcoal);
}
.input-label-sub {
    font-size: 0.75rem;
    color: var(--muted);
    margin-left: 2px;
}

/* Streamlit text input override */
.stTextInput > div > div > input {
    border-radius: var(--radius-md) !important;
    border: 2px solid var(--border) !important;
    background: var(--sand) !important;
    padding: 15px 20px !important;
    font-size: 1.05rem !important;
    font-family: 'Inter', 'Noto Sans Devanagari', sans-serif !important;
    color: var(--charcoal) !important;
    transition: all 0.2s !important;
    box-shadow: inset 0 2px 6px rgba(0,0,0,0.04) !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--leaf) !important;
    background: var(--white) !important;
    box-shadow: 0 0 0 4px rgba(26,92,58,0.1) !important;
}
.stTextInput > div > div > input::placeholder {
    color: var(--lighter) !important;
    font-style: italic;
}

/* Buttons */
.stButton > button {
    border-radius: var(--radius-md) !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    padding: 13px 26px !important;
    transition: all 0.22s ease !important;
    border: none !important;
    letter-spacing: 0.2px !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #FF6B2B 0%, #F5A623 100%) !important;
    color: white !important;
    box-shadow: 0 4px 18px rgba(255,107,43,0.35) !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(255,107,43,0.45) !important;
}
.stButton > button[kind="secondary"] {
    background: var(--white) !important;
    border: 2px solid var(--border) !important;
    color: var(--body-text) !important;
}
.stButton > button[kind="secondary"]:hover {
    border-color: var(--saffron) !important;
    color: var(--saffron) !important;
    background: var(--saffron-lt) !important;
}

/* ═══════════════════════════════════════
   RESULT SECTION
═══════════════════════════════════════ */
.result-wrapper {
    animation: slideUp 0.4s cubic-bezier(0.22,1,0.36,1);
}
@keyframes slideUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}
.result-header {
    display: flex;
    align-items: center;
    gap: 12px;
    background: var(--leaf-lt);
    border: 2px solid #C3E8D4;
    border-radius: var(--radius-md);
    padding: 1rem 1.4rem;
    margin-bottom: 1.2rem;
}
.result-icon {
    width: 44px; height: 44px;
    background: var(--leaf);
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.3rem;
    flex-shrink: 0;
}
.result-found-label {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: var(--leaf);
    margin-bottom: 2px;
}
.result-category {
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--deep-green);
}

/* Scheme Card */
.scheme-card {
    background: var(--white);
    border: 1.5px solid var(--border-light);
    border-left: 5px solid var(--saffron);
    border-radius: var(--radius-md);
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.85rem;
    transition: all 0.22s ease;
    position: relative;
    overflow: hidden;
}
.scheme-card::after {
    content: "";
    position: absolute;
    top: 0; right: 0;
    width: 80px; height: 80px;
    background: radial-gradient(circle, rgba(255,107,43,0.06) 0%, transparent 70%);
    border-radius: 0 var(--radius-md) 0 80px;
}
.scheme-card:hover {
    border-left-color: var(--leaf);
    box-shadow: var(--shadow-md);
    transform: translateX(4px);
    background: #FAFFFE;
}
.scheme-name {
    font-weight: 700;
    font-size: 1rem;
    color: var(--charcoal);
    margin-bottom: 5px;
    line-height: 1.4;
}
.scheme-eligibility {
    font-size: 0.86rem;
    color: var(--leaf);
    font-weight: 500;
    margin-bottom: 8px;
    display: flex;
    align-items: flex-start;
    gap: 6px;
    line-height: 1.5;
}
.scheme-link a {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: var(--saffron-lt);
    color: var(--saffron) !important;
    font-size: 0.82rem;
    font-weight: 700;
    text-decoration: none;
    padding: 5px 12px;
    border-radius: 50px;
    border: 1.5px solid rgba(255,107,43,0.25);
    transition: all 0.2s;
}
.scheme-link a:hover {
    background: var(--saffron) !important;
    color: white !important;
    border-color: transparent;
}

/* Confidence Bar */
.confidence-wrap {
    background: var(--sand);
    border-radius: var(--radius-md);
    padding: 1rem 1.2rem;
    margin-top: 1rem;
    border: 1px solid var(--border-light);
}
.conf-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
}
.conf-label { font-size: 0.78rem; color: var(--muted); font-weight: 600; }
.conf-percent { font-size: 0.9rem; font-weight: 800; color: var(--leaf); }
.conf-bar-bg {
    background: var(--border-light);
    border-radius: 50px;
    height: 8px;
    overflow: hidden;
}
.conf-bar-fill {
    height: 8px;
    border-radius: 50px;
    background: linear-gradient(90deg, #FF6B2B, #F5A623, #138808);
    transition: width 1s cubic-bezier(0.22,1,0.36,1);
}

/* Helpline Pill */
.helpline-row {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin-top: 1rem;
}
.helpline-pill {
    background: var(--white);
    border: 1.5px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 8px 14px;
    font-size: 0.82rem;
    font-weight: 500;
    color: var(--body-text);
    display: flex;
    align-items: center;
    gap: 6px;
}
.helpline-pill strong { color: var(--deep-green); }

/* ═══════════════════════════════════════
   EXAMPLE QUESTIONS
═══════════════════════════════════════ */
.examples-wrap {
    background: var(--white);
    border-radius: var(--radius-lg);
    padding: 1.4rem 1.6rem;
    border: 2px solid var(--border-light);
    box-shadow: var(--shadow-sm);
    margin-bottom: 1.4rem;
}
.ex-section-label {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.9rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.ex-section-label::after {
    content: "";
    flex: 1;
    height: 1px;
    background: var(--border-light);
}

/* Override Streamlit example buttons */
div[data-testid="column"] .stButton > button {
    font-size: 0.78rem !important;
    padding: 7px 12px !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    text-align: left !important;
    height: auto !important;
    white-space: normal !important;
    word-break: break-word !important;
    line-height: 1.4 !important;
    background: var(--blue-pill) !important;
    border: 1.5px solid #C7D2FE !important;
    color: var(--blue-accent) !important;
    box-shadow: none !important;
}
div[data-testid="column"] .stButton > button:hover {
    background: var(--blue-accent) !important;
    color: white !important;
    border-color: var(--blue-accent) !important;
    transform: none !important;
}

/* ═══════════════════════════════════════
   VOICE TAB
═══════════════════════════════════════ */
.voice-panel {
    background: var(--white);
    border-radius: var(--radius-lg);
    padding: 2rem;
    border: 2px solid var(--border-light);
    box-shadow: var(--shadow-sm);
    text-align: center;
}
.voice-icon-ring {
    width: 80px; height: 80px;
    border-radius: 50%;
    background: linear-gradient(135deg, var(--leaf-lt), #C3E8D4);
    border: 3px solid #C3E8D4;
    margin: 0 auto 1rem;
    display: flex; align-items: center; justify-content: center;
    font-size: 2rem;
    animation: pulse-ring 2s ease infinite;
}
@keyframes pulse-ring {
    0%, 100% { box-shadow: 0 0 0 0 rgba(26,92,58,0.2); }
    50% { box-shadow: 0 0 0 12px rgba(26,92,58,0.0); }
}
.voice-title {
    font-size: 1.2rem;
    font-weight: 700;
    color: var(--charcoal);
    margin-bottom: 0.4rem;
}
.voice-sub {
    font-size: 0.88rem;
    color: var(--muted);
    margin-bottom: 1.4rem;
    line-height: 1.6;
}
.voice-lang-chips {
    display: flex;
    gap: 8px;
    justify-content: center;
    flex-wrap: wrap;
    margin-bottom: 1.4rem;
}
.lang-chip {
    background: var(--gold-lt);
    border: 1.5px solid #F5D78A;
    color: #8B6400;
    border-radius: 50px;
    padding: 4px 14px;
    font-size: 0.78rem;
    font-weight: 600;
}

/* ═══════════════════════════════════════
   TABS
═══════════════════════════════════════ */
.stTabs [data-baseweb="tab-list"] {
    background: var(--white) !important;
    border-radius: var(--radius-md) !important;
    padding: 4px !important;
    border: 2px solid var(--border-light) !important;
    gap: 4px !important;
    margin-bottom: 1rem !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: var(--radius-sm) !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    color: var(--muted) !important;
    padding: 10px 20px !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #1A5C3A, #2D8653) !important;
    color: white !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 0.8rem !important; }

/* ═══════════════════════════════════════
   SIDEBAR CARDS
═══════════════════════════════════════ */
.side-card {
    background: var(--white);
    border-radius: var(--radius-md);
    padding: 1.2rem 1.4rem;
    border: 2px solid var(--border-light);
    margin-bottom: 1.1rem;
    box-shadow: var(--shadow-sm);
}
.side-card-label {
    font-size: 0.7rem;
    font-weight: 800;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 6px;
}
.side-card-label::after {
    content: "";
    flex: 1;
    height: 1px;
    background: var(--border-light);
}
.model-stat-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 7px 0;
    border-bottom: 1px solid var(--border-light);
    font-size: 0.86rem;
}
.model-stat-row:last-child { border-bottom: none; }
.model-stat-key { color: var(--muted); }
.model-stat-val { font-weight: 700; color: var(--deep-green); }

.cat-side-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 0;
    border-bottom: 1px dashed var(--border-light);
}
.cat-side-item:last-child { border-bottom: none; }
.cat-side-icon {
    width: 36px; height: 36px;
    background: var(--sand);
    border-radius: 9px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1rem;
    flex-shrink: 0;
}
.cat-side-name { font-weight: 600; font-size: 0.88rem; color: var(--charcoal); }
.cat-side-hint { font-size: 0.72rem; color: var(--lighter); margin-top: 1px; }

.history-chip {
    background: var(--sand);
    border: 1.5px solid var(--border-light);
    border-radius: var(--radius-sm);
    padding: 8px 11px;
    margin-bottom: 7px;
    cursor: pointer;
    transition: border-color 0.2s;
}
.history-chip:hover { border-color: var(--saffron); }
.history-query {
    font-size: 0.82rem;
    color: var(--body-text);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    font-weight: 500;
}
.history-cat {
    font-size: 0.7rem;
    color: var(--lighter);
    margin-top: 2px;
}

/* ═══════════════════════════════════════
   FOOTER
═══════════════════════════════════════ */
.footer-wrap {
    background: linear-gradient(135deg, #0F3D2B 0%, #1A5C3A 100%);
    border-radius: var(--radius-lg);
    padding: 1.2rem 1.8rem;
    color: white;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 10px;
    font-size: 0.84rem;
    margin-top: 1.2rem;
    box-shadow: 0 4px 20px rgba(15,61,43,0.3);
}
.footer-brand {
    font-weight: 700;
    display: flex;
    align-items: center;
    gap: 8px;
}
.footer-links { display: flex; gap: 1.5rem; flex-wrap: wrap; }
.footer-links a {
    color: var(--gold);
    text-decoration: none;
    font-weight: 700;
    display: flex;
    align-items: center;
    gap: 4px;
}
.footer-made {
    font-size: 0.75rem;
    opacity: 0.5;
}
.footer-tribar {
    height: 3px;
    border-radius: 0 0 var(--radius-lg) var(--radius-lg);
    background: linear-gradient(90deg, #FF9933 33%, #FFFFFF 33% 66%, #138808 66%);
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: var(--saffron) !important; }

/* ── Alerts ── */
.stSuccess { background: #F0FFF4 !important; border-radius: var(--radius-md) !important; }
.stWarning { background: #FFFBEB !important; border-radius: var(--radius-md) !important; }
.stInfo    { background: #EEF2FF !important; border-radius: var(--radius-md) !important; }

/* ── File uploader ── */
.stFileUploader > div {
    border-radius: var(--radius-md) !important;
    border: 2px dashed var(--border) !important;
    background: var(--sand) !important;
}

/* ── Responsive ── */
@media (max-width: 768px) {
    .hero-title { font-size: 1.9rem !important; }
    .cat-grid { grid-template-columns: repeat(3, 1fr) !important; }
    .hero-stats { gap: 1.2rem; }
    .footer-wrap { flex-direction: column; text-align: center; }
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# TRAINING DATA
# ──────────────────────────────────────────────
@st.cache_resource
def train_model():
    data = {
        "text": [
            # SCHOLARSHIP (80)
            "I need scholarship","money for studies","financial help for education","student scholarship scheme",
            "fee support for college","I cannot pay my fees","education funding needed","scholarship for poor students",
            "fees nahi bhar pa raha","study ke paise nahi hai","college fees bahut high hai help chahiye",
            "mujhe padhai ke liye paise chahiye","mujhe fees bharne mein madad chahiye","chhatravritti chahiye",
            "education loan ya scholarship chahiye","books ke liye paise chahiye","hostel fees ke liye help",
            "student hu paise nahi hai","garib chhatra ke liye yojana","school fees bhar nahi pa raha",
            "college ke liye financial help","scholarship kaise milegi","need help for studies urgently",
            "padhai ruk jayegi paise nahi hai","tuition fees help chahiye","govt scholarship batao",
            "education support scheme batao","meri fees pending hai help chahiye","study ke liye fund chahiye",
            "mere pass apni beti ko padhane ke paise nahi hai","student scholarship apply kaise kare",
            "education ke liye govt help chahiye","meri padhai ke liye support chahiye",
            "scholarship eligibility kya hai","higher studies ke liye fund chahiye",
            "meri fees due hai kya help milegi","education ke liye loan ya grant chahiye",
            "गरीब छात्र के लिए scholarship","padhai ke liye koi yojana batao",
            "college admission ke liye paisa chahiye","student ke liye govt aid",
            "free education scheme hai kya","study ke liye paise arrange nahi ho rahe",
            "scholarship form kaha milega","education ke liye financial support",
            "mujhe padhne ke liye madad chahiye","fees waiver scheme batao",
            "school ke liye scholarship chahiye","meri education continue karni hai help karo",
            "student ke liye paisa kaise milega","mujhe apni beti ko padhane ke liye paise chahiye",
            "meri fees bharne ke liye koi scheme hai kya","padhai ke liye loan lena padega kya",
            "govt student ke liye kya help deti hai","education ke liye paisa arrange nahi ho raha",
            "scholarship milne ka process kya hai","12th ke baad scholarship milegi kya",
            "college admission ke liye fund nahi hai","गरीब छात्रों के लिए कौन सी योजना है",
            "mujhe engineering ke liye paisa chahiye","higher education ke liye loan chahiye",
            "meri padhai beech me ruk jayegi help karo","scholarship ke liye apply kaha kare",
            "govt se padhai ke liye madad chahiye","meri beti ki padhai ke liye paisa chahiye",
            "hostel aur mess fees ke liye help chahiye","student hu earning nahi hai help chahiye",
            "padhai ke liye koi govt support hai kya","scholarship ke liye documents kya chahiye",
            "education ke liye free scheme hai kya","school se college tak support milega kya",
            "meri padhai continue karne ke liye madad karo","study loan lena safe hai kya",
            "govt scholarship kitni milti hai","padhai ke liye paisa kaha se milega",
            "scholarship late aati hai kya","meri fees pending hai kya govt help karegi",
            "study ke liye sponsor chahiye","financial problem ki wajah se padhai ruk gayi",
            "education ke liye govt scheme batao detail me",

            # AGRICULTURE (80)
            "farmer loan scheme","crop subsidy","agriculture support scheme","kisan yojana",
            "farmer needs money","crop damage help","subsidy for seeds","kheti ke liye paise chahiye",
            "fasal kharab ho gayi","kisan loan chahiye","beej ke liye paise chahiye",
            "tractor ke liye loan chahiye","kisan ke liye yojana","kheti ke liye paisa chahiye",
            "fasal kharab ho gayi","kisanon ke liye sahayata","kheti me nuksaan ho gaya",
            "crop insurance chahiye","pm kisan scheme kya hai","fasal barbaad ho gayi",
            "kisan ko subsidy milegi kya","irrigation ke liye help","kheti ke liye loan",
            "seed subsidy chahiye","agriculture scheme batao","farming support chahiye",
            "kisan ke liye govt help","crop loss compensation","soil improvement scheme",
            "kisan credit card kaise banega","farming ke liye irrigation support",
            "drip irrigation scheme hai kya","agriculture loan interest kitna hai",
            "kheti ke liye machine subsidy","harvest ke baad loss ho gaya help",
            "fertilizer subsidy kaise milegi","soil testing scheme kya hai",
            "organic farming ke liye support","kisan ke liye insurance details",
            "crop ke liye protection scheme","farming ke liye govt training",
            "agriculture equipment loan chahiye","kheti ke liye pani ki problem hai help",
            "weather se fasal kharab ho gayi help","kisan ke liye best yojana batao",
            "farming me profit kaise badhaye help","govt support for small farmers",
            "kisan ke liye financial aid","kheti me profit nahi ho raha help chahiye",
            "govt farming ke liye kya support deti hai","kisan ke liye loan ka process kya hai",
            "fasal barbad ho gayi insurance milega kya","crop ke liye best subsidy kaunsi hai",
            "small farmer ke liye scheme batao","kheti ke liye pani ka issue hai kya kare",
            "tractor kharidne ke liye loan chahiye","govt se kisan ko kitna paisa milta hai",
            "fertilizer mehenga hai subsidy milegi kya","kisan credit card ka benefit kya hai",
            "agriculture me loss ho gaya help chahiye","kheti ke liye modern tools chahiye",
            "govt farming training deti hai kya","crop protection ke liye scheme hai kya",
            "beej aur khaad ke liye loan chahiye","farming ke liye govt grant hai kya",
            "kisan ke liye monthly income scheme","fasal bechne me problem aa rahi hai help",
            "mandi rate ka issue hai kya kare","govt se irrigation ke liye support milega kya",
            "soil health card kya hota hai","organic farming ke liye scheme batao",
            "agriculture me technology use karne ke liye help","kisan ke liye best govt plan kaunsa hai",
            "crop ke liye weather protection scheme","farming business start karna hai help",
            "govt se kisan ko subsidy kaise milti hai","kheti ke liye paise kaha se milege",
            "agriculture ke liye free training hai kya","kisan ke liye naye govt schemes kya hai",
            "fasal ke liye protection ka best tarika kya hai",

            # EMPLOYMENT (80)
            "I need a job","job for unemployed","employment scheme","work opportunities",
            "job assistance","unemployment help","looking for work","need work urgently",
            "mujhe naukri chahiye","job dilado","kaam chahiye bhai","koi kaam milega kya",
            "naukri nahi mil rahi","berozgar hoon madad chahiye","kaam chahiye",
            "rojgar yojana batao","job nahi mil rahi help karo","part time job chahiye",
            "freshers ke liye job","skill development scheme batao","kaam nahi mil raha",
            "income source chahiye","self employment scheme","startup help chahiye",
            "govt job scheme","daily wage kaam chahiye","training ke baad job milegi kya",
            "skill seekh ke job chahiye","rojgar ke liye help","mujhe rozgaar chahiye",
            "job ke liye training chahiye","government job kaise milegi",
            "online job opportunities batao","skill development ka course chahiye",
            "earning ka source chahiye","work from home job chahiye",
            "job ke liye resume kaise banaye","rojgar ke liye registration kaise kare",
            "nayi job ke liye apply kaise kare","career start karne ke liye help",
            "internship chahiye freshers ke liye","skill training free hai kya",
            "job ke liye govt portal batao","self business start karna hai help",
            "income badhane ke liye kya kare","job ke liye guidance chahiye",
            "rojgar ka form kaha milega","daily income ke liye kaam chahiye",
            "job ke liye financial support","kaam ke liye training chahiye",
            "job ke liye kaha apply kare","mujhe ghar ke paas job chahiye",
            "govt job ke liye kya process hai","private job nahi mil rahi kya kare",
            "online earning kaise kare","work from home ka option hai kya",
            "skill sikh ke job milegi kya","govt free training deti hai kya",
            "rojgar ke liye best scheme batao","job ke liye resume strong kaise banaye",
            "mujhe part time earning chahiye","ghar baithe kaam mil sakta hai kya",
            "startup ke liye fund kaise milega","business ke liye loan lena hai",
            "job ke liye qualification kam hai help","kaam ke liye experience nahi hai kya kare",
            "mujhe daily income chahiye urgently","job ke liye placement help chahiye",
            "govt rojgar mela kya hota hai","skill india program kaise join kare",
            "job ke liye guidance kaun dega","career kaise choose kare help",
            "mujhe stable income chahiye","freelancing kaise start kare",
            "online job safe hai kya","govt job ke liye coaching free hai kya",
            "training ke baad placement milega kya","job ke liye registration kaha kare",
            "employment exchange kya hota hai","mujhe apna kaam start karna hai help",

            # WOMEN SUPPORT (80)
            "help for women","women empowerment scheme","support for girls","scheme for women safety",
            "financial help for women","mahila yojana","ladkiyon ke liye yojana",
            "women ke liye loan hai kya","mahila ke liye sarkari help","beti bachao scheme",
            "mahila ko naukri dilao","pregnant women help","mahila ka ilaj chahiye",
            "ladki ki padhai ke liye help","women safety scheme","mahila ke liye free training",
            "beti ke liye yojana","govt ladies ke liye kya scheme hai","women ke liye koi yojana",
            "mahila ko paisa chahiye","ladki ke liye education help","widow support scheme",
            "vidhwa pension scheme","mahila ke liye business loan","women self help group",
            "SHG loan chahiye","mahila ke liye koi govt scheme","women ke liye sarkari yojana",
            "beti ka vivah ke liye madad","shadi ke liye help chahiye","mahila ke liye job scheme",
            "pregnant ke liye free hospital","mahila ke liye free medicine","ladki ka support chahiye",
            "aangan wadi scheme","mahila ke liye poshan scheme","nutrition support for women",
            "महिला के लिए योजना","बेटी बचाओ बेटी पढ़ाओ","ujjwala yojana kya hai",
            "free gas connection chahiye","mahila ke liye ujjwala scheme",
            "women farmer ke liye scheme","ladki ko scholarship chahiye",
            "mahila ke liye koi training programme","domestic violence help chahiye",
            "mahila ke liye free legal help","women helpline number kya hai",
            "beti ke rishte ke liye govt help","ladki ki shadi ke liye fund",
            "mahila entrepreneur ke liye loan","women ke liye startup fund",
            "mahila ko aarthik madad chahiye","govt se mahila ko kya milta hai",
            "mahila suraksha yojana","safe city scheme for women",
            "anganwadi worker ke liye scheme","mahila ke liye pension",
            "vidhwa pension kaise milegi","single mother ke liye help",
            "mahila ke liye housing scheme","ladki ke liye free hostel",
            "women ke liye health checkup free","mahila ke liye ration card",
            "mahila ke liye free silai training","women ke liye skill development",
            "mahila ke liye bank account scheme","jan dhan yojana for women",
            "mahila ke liye microfinance","mahila ke liye kisan scheme",
            "women ke liye govt grant","mahila ke liye insurance scheme",
            "ladki ke liye best sarkari scheme","beti ke liye govt help detail me",
            "mahila ke liye free education scheme kya hai","working women ke liye govt support kya hai",
            "mahila self employment ke liye kya help milti hai","ladkiyon ke liye safety apps ya schemes kya hai",
            "women ke liye financial independence kaise milegi","help for women",

            # HEALTH (80)
            "health support","need medical help","hospital fees are too high","no money for treatment",
            "hospital ka kharcha nahi hai","doctor ke liye paise chahiye",
            "mujhe ilaj ke liye paise chahiye","aspataal kharche mein madad chahiye",
            "swasthya yojana","ilaj ke liye sahayata","insurance scheme batao",
            "ayushman card kaise banega","free treatment scheme","medical emergency help",
            "hospital bill bahut zyada hai","operation ke liye paise chahiye",
            "health support scheme batao","garib ke ilaj ke liye yojana",
            "medicine ke paise nahi hai","treatment afford nahi kar pa raha",
            "govt hospital scheme","health card kaise milega","bimari ke liye financial help",
            "doctor fees nahi hai","emergency medical support","healthcare subsidy",
            "free surgery scheme","insurance claim help","free checkup scheme hai kya",
            "health card banwana hai kaise kare","insurance ka claim kaise kare",
            "hospital me free treatment milega kya","emergency ke liye govt help",
            "medical loan chahiye","health ke liye subsidy milegi kya",
            "doctor consultation free hai kya","operation ke liye govt scheme",
            "medicine free milegi kya","healthcare ke liye financial support",
            "bimari ke liye insurance kaise milega","health ke liye govt program",
            "treatment ke liye loan chahiye","free ambulance service hai kya",
            "serious illness ke liye help","health benefit schemes india",
            "govt hospital me free ilaj kaise milega","health scheme apply kaise kare",
            "treatment ke liye paise arrange nahi ho rahe","mujhe free treatment kaha milega",
            "govt hospital me ilaj free hai kya","insurance lena zaroori hai kya",
            "health ke liye govt kya help deti hai","emergency me paisa kaha se milega",
            "operation ke liye loan lena padega kya","medicine bahut mehengi hai help chahiye",
            "govt se health card kaise banega","serious disease ke liye scheme hai kya",
            "doctor ka kharcha kaise manage kare","health ke liye free camp kaha lagta hai",
            "govt se ilaj ke liye madad milegi kya","hospital me admission ke liye help",
            "treatment ke liye donation kaise milega","health ke liye best scheme kaunsi hai",
            "insurance claim reject ho gaya kya kare","bimari ke liye paisa arrange nahi ho raha",
            "govt hospital ka process kya hai","health ke liye free ambulance number kya hai",
            "emergency medical loan chahiye","treatment ke liye govt grant hai kya",
            "healthcare ke liye support chahiye","doctor consultation free kaha milega",
            "insurance policy ka benefit kya hai","govt health scheme kaise check kare",
            "ilaj ke liye financial help kaise milegi","health ke liye subsidy ka process kya hai",
            "hospital bill reduce kaise kare","free surgery kaha hoti hai",
            "govt health support kaise le"," free health checkup kaha available hai",
            "govt hospital me treatment kaise start kare"
        ],

        "intent": (
            ["scholarship"] * 80 +
            ["agriculture"] * 80 +
            ["employment"] * 80 +
            ["women_support"] * 80 +
            ["health"] * 80
        )
    }

    df = pd.DataFrame(data)

    def clean_text(text):
        text = text.lower()
        cleaned = ""
        for char in text:
            if char.isalnum() or char == " " or ("\u0900" <= char <= "\u097F"):
                cleaned += char
        return cleaned

    df["clean_text"] = df["text"].apply(clean_text)
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["intent"], test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec  = vectorizer.transform(X_test)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)

    return vectorizer, model, acc


# ──────────────────────────────────────────────
# SCHEME DATABASE
# ──────────────────────────────────────────────
GT = "https://translate.google.com/translate?sl=en&tl=hi&u="

schemes = {
    "scholarship": {
        "label": "📚 शिक्षा / छात्रवृत्ति",
        "icon": "📚",
        "color": "#4338CA",
        "schemes": [
            ("📚 नेशनल स्कॉलरशिप पोर्टल (NSP)",
             "✅ पात्रता: कक्षा 1 से PG | परिवार की आय ₹2.5 लाख से कम",
             GT + "scholarships.gov.in"),
            ("📚 पोस्ट मैट्रिक छात्रवृत्ति",
             "✅ पात्रता: SC/ST/OBC छात्र | कक्षा 11 के बाद",
             GT + "socialjustice.gov.in"),
            ("📚 PM यशस्वी योजना — ₹75,000–₹1,25,000",
             "✅ पात्रता: OBC/EBC/DNT छात्र | कक्षा 9 या 11",
             GT + "yet.nta.ac.in"),
            ("📚 विद्या लक्ष्मी — एजुकेशन लोन",
             "✅ पात्रता: कोई भी छात्र जिसे लोन चाहिए",
             GT + "vidyalakshmi.co.in"),
            ("📚 केंद्रीय क्षेत्र छात्रवृत्ति",
             "✅ पात्रता: 12वीं में 80%+ | परिवार की आय ₹4.5 लाख से कम",
             GT + "scholarships.gov.in"),
        ]
    },
    "agriculture": {
        "label": "🌾 खेती / किसान",
        "icon": "🌾",
        "color": "#16A34A",
        "schemes": [
            ("🌾 PM किसान सम्मान निधि — हर साल ₹6,000",
             "✅ पात्रता: सभी किसान जिनके पास खेती की ज़मीन है",
             GT + "pmkisan.gov.in"),
            ("🌾 PM फसल बीमा योजना",
             "✅ पात्रता: सभी किसान | फसल खराब होने पर मुआवजा",
             GT + "pmfby.gov.in"),
            ("🌾 किसान क्रेडिट कार्ड (KCC)",
             "✅ पात्रता: सभी किसान | 4% ब्याज दर पर लोन",
             GT + "pmkisan.gov.in"),
            ("🌾 PM कृषि सिंचाई योजना",
             "✅ पात्रता: सभी किसान | सिंचाई उपकरण पर सब्सिडी",
             GT + "pmksy.gov.in"),
            ("🌾 सॉइल हेल्थ कार्ड — मुफ्त मिट्टी जाँच",
             "✅ पात्रता: सभी किसान | हर 2 साल में मुफ्त",
             GT + "soilhealth.dac.gov.in"),
        ]
    },
    "employment": {
        "label": "💼 रोजगार / नौकरी",
        "icon": "💼",
        "color": "#D97706",
        "schemes": [
            ("💼 मनरेगा — 100 दिन काम की गारंटी",
             "✅ पात्रता: ग्रामीण क्षेत्र के वयस्क | जॉब कार्ड ज़रूरी",
             GT + "nrega.nic.in"),
            ("💼 PM कौशल विकास योजना — मुफ्त ट्रेनिंग",
             "✅ पात्रता: 15–45 साल के युवा | 10वीं पास या ड्रॉपआउट",
             GT + "pmkvyofficial.org"),
            ("💼 DDU-GKY — ग्रामीण युवाओं के लिए",
             "✅ पात्रता: 15–35 साल | BPL परिवार को प्राथमिकता",
             GT + "ddugky.gov.in"),
            ("💼 PM रोजगार सृजन कार्यक्रम (PMEGP)",
             "✅ पात्रता: 18+ साल | खुद का व्यवसाय के लिए ₹25 लाख तक",
             GT + "kviconline.gov.in"),
            ("💼 नेशनल करियर सर्विस पोर्टल",
             "✅ पात्रता: सभी नौकरी ढूंढने वाले | मुफ्त रजिस्ट्रेशन",
             GT + "ncs.gov.in"),
        ]
    },
    "women_support": {
        "label": "👩 महिला सशक्तिकरण",
        "icon": "👩",
        "color": "#DB2777",
        "schemes": [
            ("👩 बेटी बचाओ बेटी पढ़ाओ",
             "✅ पात्रता: 0–10 साल की बेटियाँ | सभी परिवार",
             GT + "wcd.nic.in"),
            ("👩 PM उज्ज्वला योजना — मुफ्त गैस कनेक्शन",
             "✅ पात्रता: BPL परिवार की महिलाएं | APL भी पात्र",
             GT + "pmuy.gov.in"),
            ("👩 PM मातृत्व वंदना योजना — ₹5,000",
             "✅ पात्रता: गर्भवती व स्तनपान कराने वाली महिलाएं",
             GT + "pmmvy.wcd.gov.in"),
            ("👩 महिला शक्ति केंद्र",
             "✅ पात्रता: सभी ग्रामीण महिलाएं | कौशल व रोजगार सहायता",
             GT + "wcd.nic.in"),
            ("👩 स्वयं सहायता समूह (SHG) ऋण",
             "✅ पात्रता: महिलाएं | SHG के माध्यम से सस्ती ब्याज दर पर ऋण",
             GT + "nrlm.gov.in"),
        ]
    },
    "health": {
        "label": "🏥 स्वास्थ्य / इलाज",
        "icon": "🏥",
        "color": "#DC2626",
        "schemes": [
            ("🏥 आयुष्मान भारत — ₹5 लाख तक मुफ्त इलाज",
             "✅ पात्रता: SECC 2011 सूची में नाम | BPL परिवार",
             GT + "pmjay.gov.in"),
            ("🏥 आयुष्मान कार्ड — यहाँ बनवाएं",
             "✅ अपना नाम चेक करें और कार्ड बनवाएं",
             GT + "beneficiary.nha.gov.in"),
            ("🏥 राष्ट्रीय स्वास्थ्य बीमा योजना (RSBY)",
             "✅ पात्रता: BPL परिवार | ₹30,000 तक बीमा",
             GT + "rsby.gov.in"),
            ("🏥 जननी सुरक्षा योजना",
             "✅ पात्रता: गर्भवती महिलाएं | BPL / SC / ST",
             GT + "nhm.gov.in"),
            ("🏥 Helpline: Ayushman 📞 14555 | NHM 📞 1800-180-1104",
             "✅ किसी भी सवाल के लिए — बिल्कुल मुफ्त",
             None),
        ]
    }
}


# ──────────────────────────────────────────────
# CHATBOT LOGIC
# ──────────────────────────────────────────────
def clean_text(text):
    text = text.lower()
    cleaned = ""
    for char in text:
        if char.isalnum() or char == " " or ("\u0900" <= char <= "\u097F"):
            cleaned += char
    return cleaned


def chatbot_response(user_input, vectorizer, model):
    if not user_input.strip():
        return None, None, 0
    try:
        lang = detect(user_input)
        if lang == "en":
            translated = GoogleTranslator(source="en", target="hi").translate(user_input)
        else:
            translated = user_input
    except Exception:
        translated = user_input

    cleaned = clean_text(translated)
    vec = vectorizer.transform([cleaned])
    intent = model.predict(vec)[0]
    confidence = float(model.predict_proba(vec).max()) * 100
    return intent, schemes[intent], confidence


def speech_to_text_from_file(audio_file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file_path) as source:
        audio_data = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio_data, language="hi-IN")
        return text
    except Exception:
        return None


# ──────────────────────────────────────────────
# TRAIN MODEL
# ──────────────────────────────────────────────
vectorizer, model, model_acc = train_model()


# ══════════════════════════════════════════════
# ███  HERO HEADER
# ══════════════════════════════════════════════
st.markdown("""
<div class="hero-outer">
  <div class="hero-bg">
    <div class="hero-eyebrow">🇮🇳 भारत सरकार &nbsp;·&nbsp; Government of India</div>
    <div class="hero-title">सरकारी <span class="accent">योजना</span> सहायक</div>
    <div class="hero-sub">ग्रामीण नागरिकों के लिए — हिंदी, हिंग्लिश या English में अपनी ज़रूरत बताएं और सही सरकारी योजना खोजें</div>
    <div class="hero-stats">
      <div class="hero-stat">
        <span class="hero-stat-num">400+</span>
        <span class="hero-stat-label">प्रशिक्षण वाक्य</span>
      </div>
      <div class="hero-stat">
        <span class="hero-stat-num">5</span>
        <span class="hero-stat-label">योजना श्रेणियाँ</span>
      </div>
      <div class="hero-stat">
        <span class="hero-stat-num">25+</span>
        <span class="hero-stat-label">सरकारी योजनाएं</span>
      </div>
      <div class="hero-stat">
        <span class="hero-stat-num">3</span>
        <span class="hero-stat-label">भाषाएं</span>
      </div>
    </div>
    <div class="hero-badges-row">
      <span class="hero-badge">📚 शिक्षा</span>
      <span class="hero-badge">🌾 कृषि</span>
      <span class="hero-badge">💼 रोजगार</span>
      <span class="hero-badge">👩 महिला</span>
      <span class="hero-badge">🏥 स्वास्थ्य</span>
    </div>
  </div>
  <div class="hero-tricolor">
    <div class="tricolor-saffron"></div>
    <div class="tricolor-white"></div>
    <div class="tricolor-green"></div>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# ███  LAYOUT: Main + Sidebar
# ══════════════════════════════════════════════
col_main, col_side = st.columns([2.7, 1], gap="large")

with col_main:

    # ── CATEGORY QUICK-SELECT ──
    st.markdown('<div class="cat-section-label">⚡ श्रेणी चुनें — Quick Select</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="cat-grid">
      <div class="cat-card">
        <span class="cat-icon">📚</span>
        <div class="cat-name">शिक्षा</div>
        <div class="cat-hint">Scholarship · Fees</div>
      </div>
      <div class="cat-card">
        <span class="cat-icon">🌾</span>
        <div class="cat-name">कृषि</div>
        <div class="cat-hint">Kisan · Loan</div>
      </div>
      <div class="cat-card">
        <span class="cat-icon">💼</span>
        <div class="cat-name">रोजगार</div>
        <div class="cat-hint">Job · Training</div>
      </div>
      <div class="cat-card">
        <span class="cat-icon">👩</span>
        <div class="cat-name">महिला</div>
        <div class="cat-hint">Beti · Mahila</div>
      </div>
      <div class="cat-card">
        <span class="cat-icon">🏥</span>
        <div class="cat-name">स्वास्थ्य</div>
        <div class="cat-hint">Ilaj · Hospital</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── TABS ──
    tab1, tab2 = st.tabs(["⌨️  टेक्स्ट से पूछें", "🎤  आवाज़ से पूछें"])

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TAB 1: TEXT INPUT
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with tab1:
        st.markdown("""
        <div class="input-section">
          <div class="input-label-row">
            <div class="input-label-icon">✍️</div>
            <div>
              <div class="input-label-text">अपनी ज़रूरत लिखें</div>
              <div class="input-label-sub">हिंदी, हिंग्लिश या English — तीनों चलेगी</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        user_input = st.text_input(
            label="query",
            label_visibility="collapsed",
            placeholder="जैसे: मुझे पढ़ाई के लिए पैसे चाहिए  |  kisan loan chahiye  |  I need a job",
            value=st.session_state.query,
            key="text_input"
        )

        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            search_btn = st.button("🔍  योजना खोजें", type="primary", use_container_width=True)
        with c2:
            clear_btn = st.button("🗑  साफ करें", type="secondary", use_container_width=True)
        with c3:
            pass  # spacer

        if clear_btn:
            st.session_state.query = ""
            st.rerun()

        if search_btn and user_input.strip():
            with st.spinner("🔄 योजनाएं ढूंढ रहे हैं…"):
                intent, scheme_data, confidence = chatbot_response(user_input, vectorizer, model)

            if intent:
                st.session_state.history.insert(0, {"query": user_input, "intent": intent})
                if len(st.session_state.history) > 8:
                    st.session_state.history = st.session_state.history[:8]

                # Result header
                st.markdown(f"""
                <div class="result-wrapper">
                  <div class="result-header">
                    <div class="result-icon">{scheme_data['icon']}</div>
                    <div>
                      <div class="result-found-label">✅ योजनाएं मिलीं</div>
                      <div class="result-category">{scheme_data['label']}</div>
                    </div>
                  </div>
                """, unsafe_allow_html=True)

                # Scheme cards
                for name, elig, link in scheme_data["schemes"]:
                    link_html = f'<div class="scheme-link"><a href="{link}" target="_blank">🔗 हिंदी वेबसाइट खोलें</a></div>' if link else ""
                    st.markdown(f"""
                    <div class="scheme-card">
                        <div class="scheme-name">{name}</div>
                        <div class="scheme-eligibility">{elig}</div>
                        {link_html}
                    </div>
                    """, unsafe_allow_html=True)

                # Confidence bar
                bar_width = int(confidence)
                st.markdown(f"""
                <div class="confidence-wrap">
                  <div class="conf-top">
                    <div class="conf-label">🤖 AI विश्वास स्तर</div>
                    <div class="conf-percent">{bar_width:.0f}%</div>
                  </div>
                  <div class="conf-bar-bg">
                    <div class="conf-bar-fill" style="width:{bar_width}%"></div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                # Helpline
                st.markdown("""
                <div class="helpline-row">
                  <div class="helpline-pill">💡 नज़दीकी <strong>CSC</strong> या <strong>ग्राम पंचायत</strong> जाएं</div>
                  <div class="helpline-pill">📞 PM Kisan: <strong>1800-11-0001</strong></div>
                  <div class="helpline-pill">📞 Ayushman: <strong>14555</strong></div>
                </div>
                </div>
                """, unsafe_allow_html=True)

        elif search_btn and not user_input.strip():
            st.warning("⚠️ कृपया पहले अपनी ज़रूरत लिखें।")

        # ── EXAMPLE QUESTIONS ──
        st.markdown("""
        <div class="examples-wrap">
          <div class="ex-section-label">📝 उदाहरण सवाल — क्लिक करके देखें</div>
        </div>
        """, unsafe_allow_html=True)

        examples = [
            "मुझे पढ़ाई के लिए पैसे चाहिए",
            "kisan loan chahiye fasal kharab ho gayi",
            "naukri nahi mil rahi help chahiye",
            "mahila ke liye koi yojana hai kya",
            "hospital ka kharcha bahut zyada hai",
            "beti ko padhane ke paise nahi hai",
            "I need a job urgently",
            "free medical treatment kaha milega",
        ]

        ex_cols = st.columns(4)
        for i, ex in enumerate(examples):
            with ex_cols[i % 4]:
                if st.button(ex, key=f"ex_{i}", help=ex, use_container_width=True):
                    st.session_state.query = ex
                    st.rerun()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TAB 2: VOICE INPUT
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with tab2:
        st.markdown("""
        <div class="voice-panel">
          <div class="voice-icon-ring">🎙️</div>
          <div class="voice-title">आवाज़ से पूछें</div>
          <div class="voice-sub">नीचे दिए बटन को दबाएं और अपनी ज़रूरत हिंदी में बोलें।<br>AI आपकी आवाज़ सुनकर सही योजना बताएगा।</div>
          <div class="voice-lang-chips">
            <span class="lang-chip">🇮🇳 हिंदी</span>
            <span class="lang-chip">🗣 हिंग्लिश</span>
            <span class="lang-chip">🌐 English</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        voice_text = speech_to_text(
            language='hi-IN',
            start_prompt="🎙️  बोलना शुरू करें",
            stop_prompt="⏹️  रोकें",
            just_once=True,
            use_container_width=True,
            key="mic"
        )

        if voice_text:
            st.success(f"🎤 आपने बोला: **{voice_text}**")
            with st.spinner("🔄 योजनाएं ढूंढ रहे हैं…"):
                intent, scheme_data, confidence = chatbot_response(voice_text, vectorizer, model)

            if intent:
                st.markdown(f"""
                <div class="result-header" style="margin-top:1rem;">
                  <div class="result-icon">{scheme_data['icon']}</div>
                  <div>
                    <div class="result-found-label">✅ योजनाएं मिलीं</div>
                    <div class="result-category">{scheme_data['label']}</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                for name, elig, link in scheme_data["schemes"]:
                    link_html = f'<div class="scheme-link"><a href="{link}" target="_blank">🔗 हिंदी वेबसाइट खोलें</a></div>' if link else ""
                    st.markdown(f"""
                    <div class="scheme-card">
                        <div class="scheme-name">{name}</div>
                        <div class="scheme-eligibility">{elig}</div>
                        {link_html}
                    </div>
                    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# ███  SIDEBAR
# ══════════════════════════════════════════════
with col_side:

    # Model Stats Card
    st.markdown(f"""
    <div class="side-card">
      <div class="side-card-label">🤖 मॉडल जानकारी</div>
      <div class="model-stat-row">
        <span class="model-stat-key">सटीकता</span>
        <span class="model-stat-val">{model_acc*100:.1f}%</span>
      </div>
      <div class="model-stat-row">
        <span class="model-stat-key">श्रेणियाँ</span>
        <span class="model-stat-val">5</span>
      </div>
      <div class="model-stat-row">
        <span class="model-stat-key">भाषाएं</span>
        <span class="model-stat-val">हिंदी · हिंग्लिश · EN</span>
      </div>
      <div class="model-stat-row">
        <span class="model-stat-key">एल्गोरिदम</span>
        <span class="model-stat-val">TF-IDF + LR</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Category Guide Card
    cat_info = [
        ("📚", "शिक्षा", "Scholarship, fees, study"),
        ("🌾", "कृषि", "Kisan, farming, loan"),
        ("💼", "रोजगार", "Job, training, naukri"),
        ("👩", "महिला", "Women, beti, mahila"),
        ("🏥", "स्वास्थ्य", "Hospital, ilaj, treatment"),
    ]
    st.markdown("""
    <div class="side-card">
      <div class="side-card-label">📂 श्रेणियाँ</div>
    """, unsafe_allow_html=True)
    for icon, name, hint in cat_info:
        st.markdown(f"""
        <div class="cat-side-item">
          <div class="cat-side-icon">{icon}</div>
          <div>
            <div class="cat-side-name">{name}</div>
            <div class="cat-side-hint">{hint}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Helpline Card
    st.markdown("""
    <div class="side-card">
      <div class="side-card-label">📞 हेल्पलाइन</div>
      <div class="model-stat-row">
        <span class="model-stat-key">PM Kisan</span>
        <span class="model-stat-val">1800-11-0001</span>
      </div>
      <div class="model-stat-row">
        <span class="model-stat-key">Ayushman</span>
        <span class="model-stat-val">14555</span>
      </div>
      <div class="model-stat-row">
        <span class="model-stat-key">NHM</span>
        <span class="model-stat-val">1800-180-1104</span>
      </div>
      <div class="model-stat-row">
        <span class="model-stat-key">PMKVY</span>
        <span class="model-stat-val">1800-123-9626</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Recent History Card
    if st.session_state.history:
        st.markdown("""
        <div class="side-card">
          <div class="side-card-label">🕐 हाल की खोजें</div>
        """, unsafe_allow_html=True)
        for item in st.session_state.history[:5]:
            label = schemes[item["intent"]]["label"]
            q = item['query'][:38] + ("…" if len(item['query']) > 38 else "")
            st.markdown(f"""
            <div class="history-chip">
              <div class="history-query">{q}</div>
              <div class="history-cat">{label}</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if st.button("🗑  इतिहास साफ करें", key="clear_hist"):
            st.session_state.history = []
            st.rerun()


# ══════════════════════════════════════════════
# ███  FOOTER
# ══════════════════════════════════════════════
st.markdown("""
<div class="footer-wrap">
  <div class="footer-brand">🇮🇳 सरकारी योजना सहायक — ग्रामीण भारत के लिए</div>
  <div class="footer-links">
    <a href="tel:18001110001">📞 PM Kisan: 1800-11-0001</a>
    <a href="tel:14555">📞 Ayushman: 14555</a>
  </div>
  <div class="footer-made">Built with ❤️ for Rural India</div>
</div>
<div class="footer-tribar"></div>
""", unsafe_allow_html=True)
