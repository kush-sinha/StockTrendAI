import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight
import hashlib
import json
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="StockTrendAI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# AUTHENTICATION HELPERS
# ─────────────────────────────────────────────
USERS_FILE = Path(__file__).with_name("users.json")
REMEMBER_FILE = Path(__file__).with_name("remember_me.json")

def _password_hash(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()

def _normalize_answer(answer: str) -> str:
    return answer.strip().lower()

def _load_users() -> dict:
    if not USERS_FILE.exists():
        return {}
    try:
        with USERS_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            normalized = {}
            for username, record in data.items():
                # Backward compatibility:
                # - old format: {"user": "password_hash"}
                # - role format: {"user": {"password_hash": "...", "role": "...", ...}}
                if isinstance(record, str):
                    normalized[username] = {
                        "password_hash": record,
                        "security_question": "",
                        "security_answer_hash": ""
                    }
                elif isinstance(record, dict):
                    normalized[username] = {
                        "password_hash": record.get("password_hash", ""),
                        "security_question": record.get("security_question", ""),
                        "security_answer_hash": record.get("security_answer_hash", "")
                    }
            return normalized
    except Exception:
        pass
    return {}

def _save_users(users: dict) -> None:
    with USERS_FILE.open("w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)

def _load_remembered_user() -> str:
    if not REMEMBER_FILE.exists():
        return ""
    try:
        with REMEMBER_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return str(data.get("username", "")).strip()
    except Exception:
        pass
    return ""

def _save_remembered_user(username: str) -> None:
    with REMEMBER_FILE.open("w", encoding="utf-8") as f:
        json.dump({"username": username}, f, indent=2)

def _clear_remembered_user() -> None:
    try:
        if REMEMBER_FILE.exists():
            REMEMBER_FILE.unlink()
    except Exception:
        pass

def _is_password_strong(password: str):
    checks = [
        (len(password) >= 8, "at least 8 characters"),
        (any(ch.isupper() for ch in password), "an uppercase letter"),
        (any(ch.islower() for ch in password), "a lowercase letter"),
        (any(ch.isdigit() for ch in password), "a number"),
        (any(not ch.isalnum() for ch in password), "a symbol (e.g. @, #, !)"),
    ]
    missing = [rule for ok, rule in checks if not ok]
    return len(missing) == 0, missing

def _init_auth_state() -> None:
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "current_user" not in st.session_state:
        st.session_state.current_user = ""
    if "auth_view" not in st.session_state:
        st.session_state.auth_view = "login"
    users = _load_users()
    if not users and not st.session_state.logged_in:
        st.session_state.auth_view = "register"
    if not st.session_state.logged_in:
        remembered_user = _load_remembered_user()
        if remembered_user and remembered_user in users:
            st.session_state.logged_in = True
            st.session_state.current_user = remembered_user

def _render_auth_screen() -> None:
    st.title("🔐 StockTrendAI Authentication")
    st.markdown(
        '<p class="page-subtitle">Register/Login required before accessing dashboard features</p>',
        unsafe_allow_html=True
    )

    if st.session_state.auth_view == "login":
        st.markdown("### Login")
        with st.form("login_form", clear_on_submit=False):
            login_user = st.text_input("Username", key="login_username")
            login_pass = st.text_input("Password", type="password", key="login_password")
            remember_me = st.checkbox("Remember me on this device")
            login_submit = st.form_submit_button("Login")
            if login_submit:
                users = _load_users()
                if login_user in users and users[login_user].get("password_hash") == _password_hash(login_pass):
                    st.session_state.logged_in = True
                    st.session_state.current_user = login_user
                    if remember_me:
                        _save_remembered_user(login_user)
                    else:
                        _clear_remembered_user()
                    st.success("Login successful.")
                    st.rerun()
                else:
                    st.error("Invalid username or password.")

        st.markdown("Not registered yet?")
        if st.button("Go to Register"):
            st.session_state.auth_view = "register"
            st.rerun()
        if st.button("Forgot Password?"):
            st.session_state.auth_view = "forgot"
            st.rerun()

    elif st.session_state.auth_view == "register":
        st.markdown("### Register")
        with st.form("register_form", clear_on_submit=True):
            new_user = st.text_input("Create Username", key="register_username")
            new_pass = st.text_input("Create Password", type="password", key="register_password")
            confirm_pass = st.text_input("Confirm Password", type="password", key="register_confirm")
            security_question = st.selectbox(
                "Security Question (for password reset)",
                [
                    "What is your favorite color?",
                    "What is your birth city?",
                    "What is your favorite food?",
                ],
            )
            security_answer = st.text_input("Security Answer", key="register_security_answer")
            users = _load_users()
            register_submit = st.form_submit_button("Register")
            if register_submit:
                if len(new_user.strip()) < 3:
                    st.error("Username must be at least 3 characters.")
                elif not security_answer.strip():
                    st.error("Security answer is required.")
                elif new_pass != confirm_pass:
                    st.error("Passwords do not match.")
                elif new_user in users:
                    st.error("Username already exists.")
                else:
                    strong, missing_rules = _is_password_strong(new_pass)
                    if not strong:
                        st.error("Password is weak. It must include: " + ", ".join(missing_rules) + ".")
                        st.stop()
                    users[new_user] = {
                        "password_hash": _password_hash(new_pass),
                        "security_question": security_question.strip(),
                        "security_answer_hash": _password_hash(_normalize_answer(security_answer))
                    }
                    _save_users(users)
                    st.success("Registration successful. Please log in now.")
                    st.session_state.auth_view = "login"
                    st.rerun()

        if st.button("Back to Login"):
            st.session_state.auth_view = "login"
            st.rerun()

    else:
        st.markdown("### Forgot Password")
        with st.form("forgot_form", clear_on_submit=False):
            fp_user = st.text_input("Username", key="forgot_username")
            users = _load_users()
            record = users.get(fp_user, {})
            question = record.get("security_question", "")
            if question:
                st.info(f"Security Question: {question}")
            fp_answer = st.text_input("Security Answer", key="forgot_answer")
            fp_new_pass = st.text_input("New Password", type="password", key="forgot_new_pass")
            fp_confirm_pass = st.text_input("Confirm New Password", type="password", key="forgot_confirm_pass")
            forgot_submit = st.form_submit_button("Reset Password")
            if forgot_submit:
                if fp_user not in users:
                    st.error("Username not found.")
                elif not users[fp_user].get("security_question"):
                    st.error("Password reset is not configured for this account.")
                elif users[fp_user].get("security_answer_hash") != _password_hash(_normalize_answer(fp_answer)):
                    st.error("Security answer is incorrect.")
                elif fp_new_pass != fp_confirm_pass:
                    st.error("Passwords do not match.")
                else:
                    strong, missing_rules = _is_password_strong(fp_new_pass)
                    if not strong:
                        st.error("Password is weak. It must include: " + ", ".join(missing_rules) + ".")
                        st.stop()
                    users[fp_user]["password_hash"] = _password_hash(fp_new_pass)
                    _save_users(users)
                    st.success("Password reset successful. Please log in.")
                    st.session_state.auth_view = "login"
                    st.rerun()

        if st.button("Back to Login"):
            st.session_state.auth_view = "login"
            st.rerun()

_init_auth_state()

# ─────────────────────────────────────────────
# GLOBAL PLOTLY THEME
# ─────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(10,15,30,0.0)",
    plot_bgcolor="rgba(10,15,30,0.0)",
    font=dict(family="Syne, sans-serif", color="#f1f5f9"),
    margin=dict(l=40, r=20, t=50, b=40),
    xaxis=dict(gridcolor="rgba(56,189,248,0.08)", zerolinecolor="rgba(56,189,248,0.15)"),
    yaxis=dict(gridcolor="rgba(56,189,248,0.08)", zerolinecolor="rgba(56,189,248,0.15)"),
    legend=dict(bgcolor="rgba(15,23,42,0.95)", bordercolor="rgba(56,189,248,0.35)", borderwidth=1,
                font=dict(color="#f1f5f9", size=13),
                itemsizing="constant")
)

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            header {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Base ── */
*, *::before, *::after { box-sizing: border-box; }

[data-testid="stAppViewContainer"] {
    font-family: 'DM Sans', sans-serif;
    background: #FAF7F2;
    color: #1e293b;
}

/* Animated grid background */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(56,189,248,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(56,189,248,0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
}

[data-testid="stAppViewContainer"]::after {
    content: '';
    position: fixed;
    top: -40%;
    right: -20%;
    width: 700px;
    height: 700px;
    background: radial-gradient(circle, rgba(56,189,248,0.06) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
}

.main .block-container {
    padding-top: 2rem !important;
    position: relative;
    z-index: 1;
}

/* ── Header ── */
header[data-testid="stHeader"] {
    background: rgba(250,247,242,0.85) !important;
    backdrop-filter: blur(20px);
    border-bottom: 1px solid rgba(56,189,248,0.15);
}

h1 {
    font-family: 'Syne', sans-serif !important;
    font-weight: 800 !important;
    font-size: 2.4rem !important;
    background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #38bdf8 100%);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    animation: shimmer 4s linear infinite;
    letter-spacing: -0.5px;
}

@keyframes shimmer {
    0% { background-position: 0% center; }
    100% { background-position: 200% center; }
}

h2, h3 {
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    color: #1e293b !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #FAF7F2 0%, #F0EAE0 100%) !important;
    border-right: 1px solid rgba(56,189,248,0.15) !important;
}

/* Don't apply DM Sans to icon ligatures (fixes "keyboard_double_arrow_left" text) */
section[data-testid="stSidebar"] :not(.material-icons):not([data-testid="stIconMaterial"]):not(svg):not(path) {
    font-family: 'DM Sans', sans-serif !important;
}

/* Ensure Material icons keep their icon font */
.material-icons,
[data-testid="stIconMaterial"] {
    font-family: 'Material Icons' !important;
    font-weight: normal !important;
    font-style: normal !important;
    letter-spacing: normal !important;
    text-transform: none !important;
    display: inline-block !important;
    white-space: nowrap !important;
    word-wrap: normal !important;
    direction: ltr !important;
    -webkit-font-feature-settings: 'liga' !important;
    -webkit-font-smoothing: antialiased !important;
}

[data-testid="stSidebarUserContent"] label {
    color: #38bdf8 !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 1.5px;
}

/* Fix for Selectbox (Removes the blinking vertical line/cursor) */
div[data-baseweb="select"] input {
    caret-color: transparent !important;
    color: #1e293b !important;
}

[data-testid="stSidebar"] input {
    background: rgba(240,234,224,0.8) !important;
    border: 1px solid rgba(56,189,248,0.2) !important;
    border-radius: 8px !important;
    color: #1e293b !important;
}

/* Sidebar "Select Stock & Date Range" form visibility */
section[data-testid="stSidebar"] form {
    background: rgba(250, 247, 242, 0.92) !important;
    border: 1px solid rgba(56, 189, 248, 0.25) !important;
    border-radius: 14px !important;
    padding: 14px 14px 10px !important;
    box-shadow: 0 10px 24px rgba(0, 0, 0, 0.06) !important;
}

section[data-testid="stSidebar"] form label {
    color: #0369a1 !important;
    font-weight: 700 !important;
}

section[data-testid="stSidebar"] form input {
    background: #ffffff !important;
    border: 1px solid rgba(56, 189, 248, 0.35) !important;
    border-radius: 10px !important;
}

/* Make BaseWeb input wrappers white too (covers date picker inputs) */
section[data-testid="stSidebar"] form [data-baseweb="input"] {
    background: #ffffff !important;
    border-radius: 10px !important;
}

section[data-testid="stSidebar"] form [data-baseweb="input"] input {
    background: #ffffff !important;
}

section[data-testid="stSidebar"] form button {
    width: 100% !important;
    border-radius: 10px !important;
}

div[data-baseweb="select"] input { caret-color: transparent !important; }
div[data-baseweb="select"] * { caret-color: transparent !important; }
div[data-baseweb="select"] [contenteditable] { caret-color: transparent !important; }
div[data-baseweb="popover"] input { caret-color: transparent !important; }

/* ── Metric Cards ── */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(240,234,224,0.9) 0%, rgba(235,226,212,0.9) 100%) !important;
    border: 1px solid rgba(56,189,248,0.25) !important;
    border-radius: 14px !important;
    padding: 16px 18px !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.08), inset 0 1px 0 rgba(56,189,248,0.1) !important;
    transition: transform 0.2s, box-shadow 0.2s;
    position: relative;
    overflow: hidden;
}

[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(56,189,248,0.15), inset 0 1px 0 rgba(56,189,248,0.2) !important;
}

/* ── Metric Cards ── */
[data-testid="stMetricLabel"] {
    color: #1e293b !important;        /* was #64748b */
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 1px;
}

[data-testid="stMetricValue"] {
    color: #0369a1 !important;        /* was #38bdf8 — deeper sky blue */
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.45rem !important;
    font-weight: 500 !important;
}
[data-testid="stMetricDelta"] svg { display: none !important; }

/* ── Dataframes ── */
[data-testid="stDataFrame"] {
    border-radius: 12px !important;
    overflow: hidden !important;
    border: 1px solid rgba(56,189,248,0.15) !important;
    background: rgba(240,234,224,0.6) !important;
}

/* ── Section Cards ── */
.section-card {
    background: linear-gradient(135deg, rgba(240,234,224,0.7) 0%, rgba(235,226,212,0.7) 100%);
    border: 1px solid rgba(56,189,248,0.15);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    backdrop-filter: blur(10px);
}

.section-card-accent {
    border-left: 3px solid #38bdf8;
}

/* ── Tag Badges ── */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}
.badge-up { background: rgba(34,197,94,0.15); color: #22c55e; border: 1px solid rgba(34,197,94,0.3); }
.badge-down { background: rgba(239,68,68,0.15); color: #ef4444; border: 1px solid rgba(239,68,68,0.3); }
.badge-blue { background: rgba(56,189,248,0.15); color: #38bdf8; border: 1px solid rgba(56,189,248,0.3); }
.badge-purple { background: rgba(129,140,248,0.15); color: #818cf8; border: 1px solid rgba(129,140,248,0.3); }

/* ── Divider ── */
.glow-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(56,189,248,0.4), transparent);
    margin: 28px 0;
    border: none;
}

/* ── Page subtitle ── */
.page-subtitle {
    text-align: center;
    color: #64748b;
    font-size: 0.9rem;
    margin-top: -10px;
    margin-bottom: 28px;
    letter-spacing: 0.5px;
}

/* ── Info box ── */
.info-box {
    background: rgba(56,189,248,0.06);
    border: 1px solid rgba(56,189,248,0.2);
    border-radius: 10px;
    padding: 14px 18px;
    font-size: 0.88rem;
    color: #64748b;
    line-height: 1.6;
}

/* ── Download button ── */
div.stDownloadButton > button {
    background: linear-gradient(90deg, #38bdf8, #6366f1) !important;
    color: white !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 24px !important;
    font-size: 0.9rem !important;
    box-shadow: 0 0 20px rgba(56,189,248,0.3) !important;
    transition: all 0.25s ease !important;
    font-family: 'DM Sans', sans-serif !important;
}
div.stDownloadButton > button:hover {
    box-shadow: 0 0 30px rgba(56,189,248,0.6) !important;
    transform: translateY(-2px) !important;
}

/* ── Slider ── */
[data-testid="stSlider"] { padding: 0 4px; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #FAF7F2; }
::-webkit-scrollbar-thumb { background: rgba(56,189,248,0.3); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(56,189,248,0.6); }

label { color: #475569 !important; font-size: 0.9rem !important; }
</style>
""", unsafe_allow_html=True)

if not st.session_state.logged_in:
    _render_auth_screen()
    st.stop()

# ─────────────────────────────────────────────
# TITLE
# ─────────────────────────────────────────────
st.title("⚡ StockTrendAI")
st.markdown('<p class="page-subtitle">Intelligent Stock Market Forecasting & Trend Analysis Platform</p>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 10px 0 20px;'>
        <div style='font-family:Syne,sans-serif; font-size:1.3rem; font-weight:800;
                    background:linear-gradient(135deg,#38bdf8,#818cf8);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
            StockTrendAI
        </div>
        <div style='color:#475569; font-size:0.72rem; letter-spacing:2px; text-transform:uppercase; margin-top:4px;'>
            Forecasting Platform
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        f"<div style='color:#475569;font-size:0.82rem;text-align:center;'>Logged in as <strong>{st.session_state.current_user}</strong></div>",
        unsafe_allow_html=True
    )

    page = st.sidebar.selectbox(
        "Navigate",
        [
            "📥  Data Collection & Preprocessing",
            "🧹  Data Cleaning & Linear Forecasting",
            "🤖  Advanced Model Comparison",
            "📊  Trend Analysis & Classification",
        ],
    )

    st.markdown("<hr style='border-color:rgba(56,189,248,0.15);margin:16px 0;'>", unsafe_allow_html=True)

    st.markdown("#### 📌 Select Stock & Date Range")
    with st.form("data_inputs_form", clear_on_submit=False):
        ticker = st.text_input("Ticker Symbol", value=st.session_state.get("ticker", ""))
        # Streamlit date inputs show a calendar picker.
        # We keep them "empty" by only setting session state after user clicks Load Data.
        start_date = st.date_input(
            "Start Date",
            value=st.session_state.get("start_date", None),
        )
        end_date = st.date_input(
            "End Date",
            value=st.session_state.get("end_date", None),
        )
        load_clicked = st.form_submit_button("Load Data")

    if load_clicked:
        t = ticker.strip().upper()
        if not t:
            st.warning("Please enter a ticker symbol.")
        elif start_date is None or end_date is None:
            st.warning("Please select both Start Date and End Date from the calendar.")
        elif pd.to_datetime(end_date) <= pd.to_datetime(start_date):
            st.warning("End Date must be after Start Date.")
        else:
            st.session_state["ticker"] = t
            st.session_state["start_date"] = start_date
            st.session_state["end_date"] = end_date
            st.session_state["data_loaded"] = True
            st.rerun()

    st.markdown("<hr style='border-color:rgba(56,189,248,0.15);margin:16px 0;'>", unsafe_allow_html=True)

    st.markdown("""
    <div style='color:#475569; font-size:0.72rem; text-align:center; line-height:1.8;'>
        Data sourced from Yahoo Finance<br>
        Models: LR · RF · GB · XGBoost<br>
        <span style='color:#38bdf8;'>v1.0 — StockTrendAI</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:rgba(56,189,248,0.15);margin:16px 0;'>", unsafe_allow_html=True)

    if st.button("🚪 Logout", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.current_user = ""
        _clear_remembered_user()
        st.rerun()

if not st.session_state.get("data_loaded"):
    st.info("Enter ticker + dates, then click **Load Data**.")
    st.stop()

ticker = st.session_state.get("ticker", "").strip().upper()
start_date = st.session_state.get("start_date", None)
end_date = st.session_state.get("end_date", None)

if not ticker:
    st.warning("Please enter a ticker symbol (example: NVDA, AAPL, TSLA).")
    st.stop()

if start_date is None or end_date is None:
    st.warning("Please select Start Date and End Date, then click Load Data.")
    st.stop()

if pd.to_datetime(end_date) <= pd.to_datetime(start_date):
    st.warning("End Date must be after Start Date.")
    st.stop()

# ─────────────────────────────────────────────
# DATA LOADING & BASE PREPROCESSING
# ─────────────────────────────────────────────
@st.cache_data(ttl=300, max_entries=10)
def load_data(ticker, start, end):
    try:
        # yfinance is picky about date types; normalize to YYYY-MM-DD strings.
        start_ts = pd.to_datetime(start)
        end_ts = pd.to_datetime(end)
        # `yf.download` is generally more reliable than `Ticker().history`.
        raw_df = yf.download(
            ticker,
            start=start_ts.strftime("%Y-%m-%d"),
            end=end_ts.strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=False,
            threads=False,
        )
        # `yfinance` can sometimes return None on network/ticker errors.
        if raw_df is None:
            return pd.DataFrame()
        # `yf.download` can return MultiIndex columns (e.g., ('Close','NVDA')).
        # Flatten to standard OHLCV columns so the rest of the app works.
        if isinstance(raw_df.columns, pd.MultiIndex):
            # Prefer the price field level (Open/High/Low/Close/Adj Close/Volume)
            raw_df.columns = raw_df.columns.get_level_values(0)
        return raw_df
    except Exception:
        return pd.DataFrame()

raw_df = load_data(ticker, start_date, end_date)

required_cols = ["Open", "High", "Low", "Close", "Volume"]
if raw_df is None or raw_df.empty:
    st.error(
        "⚠️ No data received from Yahoo Finance right now.\n"
        f"Ticker: {ticker}\n"
        f"Start: {start_date}\n"
        f"End: {end_date}\n"
        "Check your internet connection and try again (Load Data)."
    )
    st.stop()

missing = [c for c in required_cols if c not in raw_df.columns]
if missing:
    st.error(f"⚠️ Yahoo Finance did not return required columns: {', '.join(missing)}")
    st.stop()

df = raw_df[required_cols].copy()
df["Adj Close"] = raw_df["Close"]
if "Adj Close" in raw_df.columns:
    df["Adj Close"] = raw_df["Adj Close"]

df.index = pd.to_datetime(df.index)
df = df.sort_index()
df = df[~df.index.duplicated(keep="first")]
for col in ['Open','High','Close','Low','Volume','Adj Close']:
    if df[col].dtype == object:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df['MA_20'] = df['Close'].rolling(window=20).mean()
df['MA_50'] = df['Close'].rolling(window=50).mean()

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (np.array(y_true) + 1e-9))) * 100

# ─────────────────────────────────────────────
# ██  PAGE 1 — Data Collection & Preprocessing
# ─────────────────────────────────────────────
if page == "📥  Data Collection & Preprocessing":

    st.markdown("## 📥 Data Collection & Preprocessing")
    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

    # KPI Row
    daily_return = df['Close'].pct_change().mean() * 100
    total_return = ((df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1) * 100
    volatility   = df['Close'].pct_change().std() * 100

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Ticker", ticker)
    c2.metric("Trading Days", f"{len(df):,}")
    c3.metric("Features", len(df.columns))
    c4.metric("Total Return", f"{total_return:.1f}%")
    c5.metric("Avg Daily Vol", f"{volatility:.2f}%")

    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

    # Data Preview & Stats side by side
    left, right = st.columns([1, 1])
    with left:
        st.markdown("###  Data Preview")
        st.dataframe(
            df.head(10).style.background_gradient(cmap="Blues", subset=["Close", "Volume"]),
            width='stretch', height=320
        )
    with right:
        st.markdown("### 📐 Statistical Summary")
        st.dataframe(
            df.describe().style.background_gradient(cmap="Blues"),
            width='stretch', height=320
        )

    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

    # Missing values
    st.markdown("### 🔍 Missing Values Present")
    null_df = df.isnull().sum().reset_index()
    null_df.columns = ["Column", "Missing Values"]
    null_df["Status"] = null_df["Missing Values"].apply(lambda x: "✅ Clean" if x == 0 else f"⚠️ {x} missing")
    c1, c2 = st.columns([3, 2])
    with c1:
        st.dataframe(
            null_df,
            width='stretch',
            column_config={
                "Column":         st.column_config.TextColumn("Column",         width="short"),
                "Missing Values": st.column_config.NumberColumn("Missing Values", width="short"),
                "Status":         st.column_config.TextColumn("Status",          width="large"),
            },
            hide_index=False,
        )
    with c2:
        st.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)
        st.markdown('<div class="info-box" style="margin-top:10px;">Moving averages (MA_20, MA_50) will contain NaN values for the first 19 and 49 rows respectively — this is expected behavior as they require a minimum window of prior data points to compute. These are handled in Second part using an expanding mean fill strategy.</div>', unsafe_allow_html=True)

    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

    # Chart
    st.markdown("### 📈 Price Chart with Moving Averages")
    recent_days = st.slider("Trading days to display", 50, len(df), 252)
    recent_df   = df.tail(recent_days)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recent_df.index, y=recent_df['Close'],
                             name="Close", line=dict(color="#38bdf8", width=2)))
    fig.add_trace(go.Scatter(x=recent_df.index, y=recent_df['MA_20'],
                             name="MA 20", line=dict(color="#818cf8", width=1.5, dash="dot")))
    fig.add_trace(go.Scatter(x=recent_df.index, y=recent_df['MA_50'],
                             name="MA 50", line=dict(color="#f472b6", width=1.5, dash="dash")))

    # ✅ Volume bars — light warm gray, no dark border
    fig.add_trace(go.Bar(x=recent_df.index, y=recent_df['Volume'],
                         name="Volume", yaxis="y2",
                         marker=dict(
                             color='rgba(203, 189, 172, 0.35)',        # warm gray for cream bg
                             line=dict(color='rgba(0,0,0,0)', width=0) # no dark border
                         ),
                         showlegend=True))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=520,
        yaxis2=dict(overlaying="y", side="right", showgrid=False,
                    title="Volume", tickfont=dict(color="#475569")),
        title=dict(text=f"{ticker} — Price & Volume Overview",
                   font=dict(family="Syne,sans-serif", size=16, color="#1e293b"))  # ✅ dark title
    )

    # ✅ Legend — cream background, dark text
    fig.update_layout(legend=dict(
        bgcolor="rgba(250, 247, 242, 0.90)",        # cream
        bordercolor="rgba(56, 189, 248, 0.30)",
        borderwidth=1,
        font=dict(color="#1e293b", size=13),        # dark text
        itemsizing="constant",
        orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5
    ))

    st.plotly_chart(fig, width='stretch')

    # OHLC Candlestick
    st.markdown("### 🕯 Candlestick Chart (Last 90 Days)")
    candle_df = df.tail(90)
    fig_c = go.Figure(go.Candlestick(
        x=candle_df.index,
        open=candle_df['Open'], high=candle_df['High'],
        low=candle_df['Low'],   close=candle_df['Close'],
        increasing_line_color="#22c55e", decreasing_line_color="#ef4444",
        name="OHLC"
    ))
    fig_c.update_layout(**PLOTLY_LAYOUT, height=420,
                        title=dict(text=f"{ticker} — Candlestick (Last 90 Days)",
                                   font=dict(family="Syne,sans-serif", size=16, color="#415e7a")))
    fig_c.update_xaxes(rangeslider_visible=False)
    st.plotly_chart(fig_c, width='stretch')

# ─────────────────────────────────────────────
# ██  PAGE 2 — Data Cleaning & Linear Forecasting
# ─────────────────────────────────────────────
elif page == "🧹  Data Cleaning & Linear Forecasting":

    st.markdown("## 🧹 Data Cleaning & Linear Forecasting")
    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

    # Fill MAs
    df['MA_20'] = df['MA_20'].fillna(df['Close'].expanding().mean())
    df['MA_50'] = df['MA_50'].fillna(df['Close'].expanding().mean())

    # Outlier handling
    def rolling_zscore_smooth(series, window=30, threshold=3.0):
        min_p     = max(1, window // 2)
        roll_mean = series.rolling(window, min_periods=min_p).mean()
        roll_std  = series.rolling(window, min_periods=min_p).std()
        roll_mean = roll_mean.fillna(series.expanding().mean())
        roll_std  = roll_std.fillna(series.expanding().std()).fillna(0)
        roll_std_safe = roll_std.replace(0, np.nan)
        z = (series - roll_mean) / roll_std_safe
        z = z.fillna(0)
        mask = z.abs() > threshold
        adjusted = series.copy()
        adjusted[mask] = roll_mean[mask] + threshold * roll_std[mask] * np.sign(z[mask])
        return adjusted, int(mask.sum())

    df["Close_cleaned"], close_out = rolling_zscore_smooth(df["Close"])
    df["MA20_cleaned"],  ma20_out  = rolling_zscore_smooth(df["MA_20"])
    df["MA50_cleaned"],  ma50_out  = rolling_zscore_smooth(df["MA_50"])

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Missing Values", "0")
    c2.metric("Close Outliers Adjusted", close_out)
    c3.metric("MA20 Outliers Adjusted",  ma20_out)
    c4.metric("MA50 Outliers Adjusted",  ma50_out)

    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

    # Before / After comparison chart
    st.markdown("### 🔄 Before vs After Cleaning")
    fig_ba = make_subplots(rows=1, cols=2,
                           subplot_titles=("Close Price: Raw vs Cleaned", "Distribution Comparison"))
    fig_ba.add_trace(go.Scatter(x=df.index, y=df["Close"],
                                name="Raw", line=dict(color="#ef4444", width=1.2)), row=1, col=1)
    fig_ba.add_trace(go.Scatter(x=df.index, y=df["Close_cleaned"],
                                name="Cleaned", line=dict(color="#22c55e", width=1.5, dash="dot")), row=1, col=1)
    fig_ba.add_trace(go.Histogram(x=df["Close"], name="Raw Dist",
                                  marker_color="rgba(239,68,68,0.4)", nbinsx=60), row=1, col=2)
    fig_ba.add_trace(go.Histogram(x=df["Close_cleaned"], name="Cleaned Dist",
                                  marker_color="rgba(34,197,94,0.4)", nbinsx=60), row=1, col=2)
    fig_ba.update_layout(**PLOTLY_LAYOUT, height=420, showlegend=True)
    fig_ba.update_annotations(font=dict(color="#1e293b", size=13))
    fig_ba.update_xaxes(gridcolor="rgba(56,189,248,0.08)")
    fig_ba.update_yaxes(gridcolor="rgba(56,189,248,0.08)")
    st.plotly_chart(fig_ba, width='stretch')

    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

    # Linear Regression
    st.markdown("### 📉 Linear Regression Forecast")

    model_df = df.copy()
    model_df["target"] = model_df["Close_cleaned"].shift(-1)
    model_df = model_df.dropna()

    feature_cols = ["Close_cleaned", "MA20_cleaned", "MA50_cleaned", "Volume"]
    X = model_df[feature_cols]
    y = model_df["target"]

    split70 = int(len(model_df) * 0.70)
    split85 = int(len(model_df) * 0.85)

    X_train = X[:split70];   y_train = y[:split70]
    X_val   = X[split70:split85]; y_val = y[split70:split85]
    X_test  = X[split85:];  y_test  = y[split85:]

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_val_pred  = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    mae_val  = mean_absolute_error(y_val, y_val_pred)
    rmse_val = rmse(y_val, y_val_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    rmse_test= rmse(y_test, y_test_pred)

    left, right = st.columns([1, 2])
    with left:
        st.markdown("#### Model Metrics")
        mc1, mc2 = st.columns(2)
        mc1.metric("Val MAE",  round(mae_val, 3))
        mc2.metric("Val RMSE", round(rmse_val, 3))
        mc3, mc4 = st.columns(2)
        mc3.metric("Test MAE",  round(mae_test, 3))
        mc4.metric("Test RMSE", round(rmse_test, 3))
        st.markdown('<div class="info-box" style="margin-top:16px;">Linear Regression predicts next-day closing price using current Close, MA20, MA50, and Volume as features. Train/Val/Test split: 70% / 15% / 15%.</div>', unsafe_allow_html=True)
    with right:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=model_df.index[split85:], y=y_test,
                                  name="Actual", line=dict(color="#94a3b8", width=2)))
        fig2.add_trace(go.Scatter(x=model_df.index[split85:], y=y_test_pred,
                                  name="Predicted", line=dict(color="#38bdf8", width=2, dash="dash")))
        fig2.update_layout(**PLOTLY_LAYOUT, height=380,
                           title=dict(text="Actual vs Predicted — Test Set",
                                      font=dict(family="Syne,sans-serif", size=15, color="#7db1e6")))
        st.plotly_chart(fig2, width='stretch')

# ─────────────────────────────────────────────
# ██  PAGE 3 — Advanced Model Comparison
# ─────────────────────────────────────────────
elif page == "🤖  Advanced Model Comparison":

    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBRegressor

    st.markdown("## 🤖 Advanced Model Comparison")
    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

    # ── Feature Engineering ──
    df['MA_20'] = df['MA_20'].fillna(df['Close'].expanding().mean())
    df['MA_50'] = df['MA_50'].fillna(df['Close'].expanding().mean())

    def rolling_zscore_smooth(series, window=30, threshold=3.0):
        min_p = max(1, window // 2)
        roll_mean = series.rolling(window, min_periods=min_p).mean().fillna(series.expanding().mean())
        roll_std  = series.rolling(window, min_periods=min_p).std().fillna(series.expanding().std()).fillna(0)
        roll_std_safe = roll_std.replace(0, np.nan)
        z = ((series - roll_mean) / roll_std_safe).fillna(0)
        mask = z.abs() > threshold
        adjusted = series.copy()
        adjusted[mask] = roll_mean[mask] + threshold * roll_std[mask] * np.sign(z[mask])
        return adjusted, int(mask.sum())

    df["Close_cleaned"], _ = rolling_zscore_smooth(df["Close"])

    output_df = df.copy()
    output_df["Return"]     = output_df["Close_cleaned"].pct_change()
    output_df["MA_10"]      = output_df["Close_cleaned"].rolling(10).mean()
    output_df["MA_20"]      = output_df["Close_cleaned"].rolling(20).mean()
    output_df["MA_50"]      = output_df["Close_cleaned"].rolling(50).mean()
    output_df["Volatility"] = output_df["Return"].rolling(10).std()
    output_df["Momentum"]   = output_df["Close_cleaned"] - output_df["Close_cleaned"].shift(5)
    output_df["Lag1"]       = output_df["Close_cleaned"].shift(1)
    output_df["Lag2"]       = output_df["Close_cleaned"].shift(2)
    output_df["Lag3"]       = output_df["Close_cleaned"].shift(3)

    delta     = output_df["Close_cleaned"].diff()
    gain      = delta.where(delta > 0, 0)
    loss      = -delta.where(delta < 0, 0)
    avg_gain  = gain.rolling(14).mean()
    avg_loss  = loss.rolling(14).mean()
    rs        = avg_gain / avg_loss
    output_df["RSI"] = 100 - (100 / (1 + rs))
    output_df = output_df.dropna()

    feature_cols = ["Close_cleaned","MA_10","MA_20","MA_50","Volume",
                    "Return","Volatility","Momentum","RSI","Lag1","Lag2","Lag3"]

    model_df = output_df.copy()
    model_df["y_next_close"] = model_df["Close_cleaned"].shift(-1)
    model_df = model_df.dropna()

    n = len(model_df)
    train_end = int(n * 0.70)
    val_end   = int(n * 0.85)

    X = model_df[feature_cols]
    y = model_df["y_next_close"]

    X_train = X[:train_end];  y_train = y[:train_end]
    X_val   = X[train_end:val_end]; y_val = y[train_end:val_end]
    X_test  = X[val_end:];   y_test  = y[val_end:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled   = scaler.transform(X_val)
    X_test_scaled  = scaler.transform(X_test)

    with st.spinner("Training models: "):
        lin = LinearRegression()
        lin.fit(X_train_scaled, y_train)
        lin_val_pred  = lin.predict(X_val_scaled)
        lin_test_pred = lin.predict(X_test_scaled)

        rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        rf_val_pred  = rf.predict(X_val)
        rf_test_pred = rf.predict(X_test)

        gb = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
        gb.fit(X_train_scaled, y_train)
        gb_val_pred  = gb.predict(X_val_scaled)
        gb_test_pred = gb.predict(X_test_scaled)

        xgb = XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=5,
                            subsample=0.9, colsample_bytree=0.9, random_state=42, verbosity=0)
        xgb.fit(X_train_scaled, y_train)
        xgb_val_pred  = xgb.predict(X_val_scaled)
        xgb_test_pred = xgb.predict(X_test_scaled)

    def accuracy_pct(y_true, y_pred):
        return 100 - mape(y_true, y_pred)

    def rating(acc):
        if acc > 95: return "🌟 Excellent"
        elif acc > 90: return "✅ Very Good"
        elif acc > 80: return "👍 Good"
        else: return "⚡ Average"

    results = pd.DataFrame({
        "Model"   : ["Linear Regression","Random Forest","Gradient Boosting","XGBoost"],
        "MAE"     : [mean_absolute_error(y_val, lin_val_pred),
                     mean_absolute_error(y_val, rf_val_pred),
                     mean_absolute_error(y_val, gb_val_pred),
                     mean_absolute_error(y_val, xgb_val_pred)],
        "RMSE"    : [rmse(y_val, lin_val_pred), rmse(y_val, rf_val_pred),
                     rmse(y_val, gb_val_pred),  rmse(y_val, xgb_val_pred)],
        "MAPE(%)" : [mape(y_val, lin_val_pred), mape(y_val, rf_val_pred),
                     mape(y_val, gb_val_pred),  mape(y_val, xgb_val_pred)],
        "Accuracy": [accuracy_pct(y_val, lin_val_pred), accuracy_pct(y_val, rf_val_pred),
                     accuracy_pct(y_val, gb_val_pred),  accuracy_pct(y_val, xgb_val_pred)]
    })
    results["Rating"] = results["Accuracy"].apply(rating)
    results = results.sort_values("Accuracy", ascending=False).reset_index(drop=True)

    best = results.iloc[0]
    preds_map = {
        "Linear Regression": lin_test_pred,
        "Random Forest":     rf_test_pred,
        "Gradient Boosting": gb_test_pred,
        "XGBoost":           xgb_test_pred
    }
    best_test_pred = preds_map[best["Model"]]

    # ── Best Model Banner ──
    st.markdown(f"### 🏆 Best Model: {best['Model']}")
    b1,b2,b3,b4,b5,b6 = st.columns(6)
    b1.metric("Model",    best["Model"])
    b2.metric("Accuracy", f"{best['Accuracy']:.2f}%")
    b3.metric("MAE",      round(best["MAE"], 3))
    b4.metric("RMSE",     round(best["RMSE"], 3))
    b5.metric("MAPE",     f"{best['MAPE(%)']:.3f}%")
    b6.metric("Rating",   best["Rating"])

    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

    # ── Table + Bar chart ──
    left, right = st.columns([1, 1])
    with left:
        st.markdown("#### 📊 All Models Performance")
        st.dataframe(results.round(3), width='stretch', height=220)
    with right:
        fig_bar = go.Figure()
        colors = ["#38bdf8","#818cf8","#6366f1","#f472b6"]
        fig_bar.add_trace(go.Bar(
            x=results["Model"], y=results["Accuracy"],
            text=results["Accuracy"].round(2).astype(str) + "%",
            textposition="outside",
            marker_color=colors[:len(results)],
            marker_line_color="rgba(255,255,255,0.1)",
            marker_line_width=1
        ))
        fig_bar.update_layout(**PLOTLY_LAYOUT, height=320,
                              title=dict(text="Model Accuracy Comparison",
                                         font=dict(family="Syne,sans-serif", size=15, color="#5780a8")))
        fig_bar.update_layout(margin=dict(l=40, r=20, t=70, b=100))
        fig_bar.update_yaxes(range=[
            max(0, results["Accuracy"].min() - 5),
            results["Accuracy"].max() + 12
        ])
        st.plotly_chart(fig_bar, width='stretch')

    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

    #st.markdown(f"#### 📈 Actual vs Predicted — {best['Model']} (Test Set)")
    fig_pred = go.Figure()
    test_idx = model_df.index[val_end:]

    fig_pred.add_trace(go.Scatter(x=test_idx, y=y_test.values,
                              name="Actual", line=dict(color="#cbd5e1", width=2)))  # ← lighter

    fig_pred.add_trace(go.Scatter(x=test_idx, y=best_test_pred,
                              name="Predicted", line=dict(color="#38bdf8", width=2, dash="dash")))

    residuals = y_test.values - best_test_pred
    fig_pred.add_trace(go.Bar(x=test_idx, y=residuals,
                          name="Residuals", yaxis="y2",
                          marker=dict(
                              color=[
                                  'rgba(56, 189, 248, 0.20)' if v >= 0 else 'rgba(251, 113, 133, 0.20)'
                                  for v in residuals
                              ],
                              line=dict(color='rgba(0,0,0,0)', width=0)  # ← no dark border
                          )))

    fig_pred.update_layout(
    **PLOTLY_LAYOUT, height=460,
    yaxis2=dict(overlaying="y", side="right", showgrid=False,
                title="Residuals", tickfont=dict(color="#9298d4")),
    title=dict(text=f"{best['Model']} — Test Set Predictions + Residuals",
               font=dict(family="Syne,sans-serif", size=15, color="#1e293b"))  # ← dark title
)
    st.plotly_chart(fig_pred, width='stretch')

    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

    # ── 7-Day Forecast ──
    st.markdown("### 📅 7-Day Future Price Forecast")

    forecast_df_loop = output_df.copy()
    future_prices = []
    future_dates  = []

    best_model_obj = {"Linear Regression": lin, "Random Forest": rf,
                      "Gradient Boosting": gb, "XGBoost": xgb}[best["Model"]]
    use_scaler = best["Model"] != "Random Forest"

    for i in range(7):
        latest   = forecast_df_loop.iloc[-1]
        features = latest[feature_cols].to_frame().T
        if use_scaler:
            pred = best_model_obj.predict(scaler.transform(features))[0]
        else:
            pred = best_model_obj.predict(features)[0]
        future_prices.append(pred)

        next_row = latest.copy()
        next_row["Close_cleaned"] = pred
        next_row["Lag3"] = latest["Lag2"]
        next_row["Lag2"] = latest["Lag1"]
        next_row["Lag1"] = pred
        forecast_df_loop = pd.concat([forecast_df_loop, pd.DataFrame([next_row])])

        forecast_df_loop["Return"]     = forecast_df_loop["Close_cleaned"].pct_change()
        forecast_df_loop["MA_10"]      = forecast_df_loop["Close_cleaned"].rolling(10).mean()
        forecast_df_loop["MA_20"]      = forecast_df_loop["Close_cleaned"].rolling(20).mean()
        forecast_df_loop["MA_50"]      = forecast_df_loop["Close_cleaned"].rolling(50).mean()
        forecast_df_loop["Volatility"] = forecast_df_loop["Return"].rolling(10).std()
        forecast_df_loop["Momentum"]   = forecast_df_loop["Close_cleaned"] - forecast_df_loop["Close_cleaned"].shift(5)

        delta    = forecast_df_loop["Close_cleaned"].diff()
        gain     = delta.where(delta > 0, 0)
        loss     = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        forecast_df_loop["RSI"] = 100 - (100 / (1 + avg_gain / avg_loss))

    last_date    = model_df.index[-1]
    future_dates = pd.date_range(start=last_date + pd.tseries.offsets.BDay(1), periods=7, freq="B")

    forecast_result = pd.DataFrame({
        "Date": future_dates.strftime("%Y-%m-%d"),
        "Forecast Price": [round(p, 2) for p in future_prices],
        "Change":  [round(future_prices[i] - (future_prices[i-1] if i > 0 else output_df["Close_cleaned"].iloc[-1]), 2)
                    for i in range(7)],
        "Trend":   ["▲ Up" if (future_prices[i] > (future_prices[i-1] if i > 0 else output_df["Close_cleaned"].iloc[-1]))
                    else "▼ Down" for i in range(7)]
    })

    left, right = st.columns([1, 2])
    with left:
        st.markdown("#### Forecast Table")
        st.dataframe(forecast_result, width='stretch')
        csv = forecast_result.to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Download Forecast CSV", csv, "forecast.csv", "text/csv")
    with right:
        fig3 = go.Figure()
        hist_tail = df.tail(90)
        fig3.add_trace(go.Scatter(x=hist_tail.index, y=hist_tail["Close"],
                                  name="Historical", line=dict(color="#475569", width=2)))
        fig3.add_trace(go.Scatter(
            x=pd.to_datetime(forecast_result["Date"]),
            y=forecast_result["Forecast Price"],
            name="7-Day Forecast", mode="lines+markers",
            line=dict(color="#38bdf8", width=2.5, dash="dash"),
            marker=dict(size=8, color="#38bdf8",
                        line=dict(color="white", width=1.5))
        ))
        fig3.update_layout(**PLOTLY_LAYOUT, height=360,
                           title=dict(text="7-Day Price Forecast",
                                      font=dict(family="Syne,sans-serif", size=15, color="#4f7192")))
        st.plotly_chart(fig3, width='stretch')

    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

    # ── Performance Report ──
    st.markdown("### 📋 Performance Report")
    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown(f"""
        <div class="section-card section-card-accent">
            <div style="font-family:Syne,sans-serif;font-weight:700;color:#38bdf8;margin-bottom:10px;">Executive Summary</div>
            <div style="color:#94a3b8;font-size:0.88rem;line-height:1.7;">
                Best model: <strong style="color:#38bdf8;">{best['Model']}</strong><br>
                Accuracy: <strong style="color:#38bdf8;">{best['Accuracy']:.2f}%</strong><br>
                Rating: {best['Rating']}
            </div>
        </div>
        """, unsafe_allow_html=True)
    with r2:
        st.markdown(f"""
        <div class="section-card section-card-accent">
            <div style="font-family:Syne,sans-serif;font-weight:700;color:#818cf8;margin-bottom:10px;">Key Metrics</div>
            <div style="color:#94a3b8;font-size:0.88rem;line-height:1.7;">
                Best MAE: <strong style="color:#38bdf8;">{best['MAE']:.3f}</strong><br>
                Best RMSE: <strong style="color:#38bdf8;">{best['RMSE']:.3f}</strong><br>
                Avg Accuracy: <strong style="color:#38bdf8;">{results['Accuracy'].mean():.2f}%</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with r3:
        st.markdown(f"""
        <div class="section-card section-card-accent">
            <div style="font-family:Syne,sans-serif;font-weight:700;color:#f472b6;margin-bottom:10px;">Recommendations</div>
            <div style="color:#94a3b8;font-size:0.88rem;line-height:1.7;">
                • Deploy <strong style="color:#38bdf8;">{best['Model']}</strong> for production<br>
                • Add MACD & Bollinger Bands<br>
                • Consider LSTM for sequence modeling
            </div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# ██  PAGE 4 — Trend Analysis & Classification
# ─────────────────────────────────────────────
elif page == "📊  Trend Analysis & Classification":

    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBRegressor

    # ─────────────────────────────────────────────────────────────────────────
    # CACHED FUNCTION — trains all models once, reuses on every re-render
    # Cache key = ticker + dates, so retrains only when inputs change
    # ─────────────────────────────────────────────────────────────────────────
    @st.cache_resource(show_spinner=False)
    def build_and_train(ticker, start_str, end_str):
        stock = yf.Ticker(ticker)
        raw   = stock.history(start=start_str, end=end_str)
        dfc   = raw[['Open','High','Low','Close','Volume']].copy()
        dfc.index = pd.to_datetime(dfc.index)
        dfc   = dfc.sort_index()
        dfc   = dfc[~dfc.index.duplicated(keep="first")]

        dfc['MA_20'] = dfc['Close'].rolling(20).mean().fillna(dfc['Close'].expanding().mean())
        dfc['MA_50'] = dfc['Close'].rolling(50).mean().fillna(dfc['Close'].expanding().mean())

        def _smooth(series, window=30, threshold=3.0):
            min_p     = max(1, window // 2)
            roll_mean = series.rolling(window, min_periods=min_p).mean().fillna(series.expanding().mean())
            roll_std  = series.rolling(window, min_periods=min_p).std().fillna(series.expanding().std()).fillna(0)
            z         = ((series - roll_mean) / roll_std.replace(0, np.nan)).fillna(0)
            mask      = z.abs() > threshold
            out       = series.copy()
            out[mask] = roll_mean[mask] + threshold * roll_std[mask] * np.sign(z[mask])
            return out

        dfc["Close_cleaned"] = _smooth(dfc["Close"])

        out = dfc.copy()
        out["Return"]     = out["Close_cleaned"].pct_change()
        out["MA_10"]      = out["Close_cleaned"].rolling(10).mean()
        out["MA_20"]      = out["Close_cleaned"].rolling(20).mean()
        out["MA_50"]      = out["Close_cleaned"].rolling(50).mean()
        out["Volatility"] = out["Return"].rolling(10).std()
        out["Momentum"]   = out["Close_cleaned"] - out["Close_cleaned"].shift(5)
        out["Lag1"]       = out["Close_cleaned"].shift(1)
        out["Lag2"]       = out["Close_cleaned"].shift(2)
        out["Lag3"]       = out["Close_cleaned"].shift(3)
        delta             = out["Close_cleaned"].diff()
        gain              = delta.where(delta > 0, 0)
        loss              = -delta.where(delta < 0, 0)
        out["RSI"]        = 100 - (100 / (1 + gain.rolling(14).mean() / loss.rolling(14).mean()))
        out               = out.dropna()

        fcols = ["Close_cleaned","MA_10","MA_20","MA_50","Volume",
                 "Return","Volatility","Momentum","RSI","Lag1","Lag2","Lag3"]

        mdf = out.copy()
        mdf["y_next_close"] = mdf["Close_cleaned"].shift(-1)
        mdf = mdf.dropna()

        n         = len(mdf)
        train_end = int(n * 0.70)
        val_end   = int(n * 0.85)

        X = mdf[fcols]; y_reg = mdf["y_next_close"]
        Xtr = X[:train_end];        ytr = y_reg[:train_end]
        Xvl = X[train_end:val_end]; yvl = y_reg[train_end:val_end]
        Xts = X[val_end:];          yts = y_reg[val_end:]

        sc    = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr)
        Xvl_s = sc.transform(Xvl)
        Xts_s = sc.transform(Xts)

        # Regressors (reduced estimators for speed)
        lin     = LinearRegression().fit(Xtr_s, ytr)
        lin_p   = lin.predict(Xts_s)

        rf_reg  = RandomForestRegressor(n_estimators=100, max_depth=10,
                                        random_state=42, n_jobs=-1).fit(Xtr, ytr)
        rf_p    = rf_reg.predict(Xts)

        gb_reg  = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05,
                                            max_depth=4, random_state=42).fit(Xtr_s, ytr)
        gb_p    = gb_reg.predict(Xts_s)

        xgb_reg = XGBRegressor(n_estimators=150, learning_rate=0.03, max_depth=5,
                               subsample=0.9, colsample_bytree=0.9,
                               random_state=42, verbosity=0).fit(Xtr_s, ytr)
        xgb_p   = xgb_reg.predict(Xts_s)

        # Classification target
        y_cls  = (mdf["y_next_close"] > mdf["Close_cleaned"]).astype(int)
        yc_tr  = y_cls.iloc[:train_end]
        yc_vl  = y_cls.iloc[train_end:val_end]
        yc_ts  = y_cls.iloc[val_end:]

        log_clf  = LogisticRegression(max_iter=1000, random_state=42,
                                      class_weight='balanced').fit(Xtr_s, yc_tr)
        log_p    = log_clf.predict(Xts_s)
        log_acc  = accuracy_score(yc_ts, log_p) * 100

        rf_clf   = RandomForestClassifier(n_estimators=100, max_depth=10,
                                          random_state=42, n_jobs=-1,
                                          class_weight='balanced').fit(Xtr, yc_tr)
        rf_cp    = rf_clf.predict(Xts)
        rf_cacc  = accuracy_score(yc_ts, rf_cp) * 100

        sw       = compute_sample_weight('balanced', y=yc_tr)
        gb_clf   = GradientBoostingClassifier(n_estimators=150, learning_rate=0.05,
                                              max_depth=4, random_state=42).fit(Xtr_s, yc_tr,
                                                                                sample_weight=sw)
        gb_cp    = gb_clf.predict(Xts_s)
        gb_cacc  = accuracy_score(yc_ts, gb_cp) * 100

        return dict(
            output_df=out, model_df=mdf, scaler=sc,
            feature_cols=fcols, train_end=train_end, val_end=val_end,
            X_test=Xts, y_test=yts, X_test_scaled=Xts_s,
            lin=lin, rf_reg=rf_reg, gb_reg=gb_reg, xgb_reg=xgb_reg,
            lin_pred=lin_p, rf_pred=rf_p, gb_pred=gb_p, xgb_pred=xgb_p,
            y_cls_test=yc_ts,
            log_pred=log_p,  log_acc=log_acc,
            rf_clf_pred=rf_cp, rf_clf_acc=rf_cacc,
            gb_clf_pred=gb_cp, gb_clf_acc=gb_cacc,
        )

    # ── Load (spinner shown only on first run) ────────────────────────────────
    with st.spinner("⚙️ Training models — "):
        C = build_and_train(ticker, str(start_date), str(end_date))

    # ── Unpack all cached results ─────────────────────────────────────────────
    output_df     = C["output_df"]
    model_df      = C["model_df"]
    scaler        = C["scaler"]
    feature_cols  = C["feature_cols"]
    train_end     = C["train_end"]
    val_end       = C["val_end"]
    X_test        = C["X_test"]
    y_test        = C["y_test"]
    X_test_scaled = C["X_test_scaled"]
    lin_test_pred = C["lin_pred"];  rf_test_pred  = C["rf_pred"]
    gb_test_pred  = C["gb_pred"];   xgb_test_pred = C["xgb_pred"]
    lin    = C["lin"];    rf_reg  = C["rf_reg"]
    gb_reg = C["gb_reg"]; xgb_reg = C["xgb_reg"]
    y_cls_test       = C["y_cls_test"]
    log_test_pred    = C["log_pred"];    log_acc    = C["log_acc"]
    rf_clf_test_pred = C["rf_clf_pred"]; rf_clf_acc = C["rf_clf_acc"]
    gb_clf_test_pred = C["gb_clf_pred"]; gb_clf_acc = C["gb_clf_acc"]

    # ── Best classifier ───────────────────────────────────────────────────────
    clf_map = {
        "Logistic Regression": (log_acc,    log_test_pred),
        "RF Classifier":       (rf_clf_acc, rf_clf_test_pred),
        "GB Classifier":       (gb_clf_acc, gb_clf_test_pred),
    }
    best_clf_name = max(clf_map, key=lambda k: clf_map[k][0])
    best_clf_acc  = clf_map[best_clf_name][0]
    best_clf_pred = clf_map[best_clf_name][1]

    # ── Naive directional accuracy ────────────────────────────────────────────
    actual_prices_test = X_test["Close_cleaned"].values
    actual_trend_test  = (y_test.values > actual_prices_test).astype(int)
    lin_naive_acc = accuracy_score(actual_trend_test, (lin_test_pred  > actual_prices_test).astype(int)) * 100
    rf_naive_acc  = accuracy_score(actual_trend_test, (rf_test_pred   > actual_prices_test).astype(int)) * 100
    gb_naive_acc  = accuracy_score(actual_trend_test, (gb_test_pred   > actual_prices_test).astype(int)) * 100
    xgb_naive_acc = accuracy_score(actual_trend_test, (xgb_test_pred  > actual_prices_test).astype(int)) * 100

    # ─────────────────────────────────────────────────────────────────────────
    # UI STARTS HERE
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("## 📊 Trend Analysis & Classification")
    st.markdown('<p style="color:#64748b;font-size:0.88rem;margin-top:-10px;">Combines insights from all three milestones — cleaning, modeling, and advanced trend direction prediction</p>', unsafe_allow_html=True)
    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

    # ── KPI Row ──────────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Best Classifier",   best_clf_name)
    k2.metric("Directional Acc",   f"{best_clf_acc:.1f}%")
    k3.metric("Improvement vs LR", f"+{best_clf_acc - lin_naive_acc:.1f}%")
    k4.metric("Up Days (Test)",    int(y_cls_test.sum()))
    k5.metric("Down Days (Test)",  int(len(y_cls_test) - y_cls_test.sum()))

    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

    # ── Section A: Historical Trend Regions ──────────────────────────────────
    st.markdown("### 🟢 Historical Trend Regions")
    st.markdown('<div class="info-box">Green shading = Bullish period (MA10 &gt; MA50) &nbsp;|&nbsp; Red shading = Bearish period (MA10 &lt; MA50)</div>', unsafe_allow_html=True)
    st.markdown("")

    dates_plot = output_df.index
    bullish    = output_df["MA_10"] > output_df["MA_50"]

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(x=dates_plot, y=output_df["Close_cleaned"],
                                  name="Close", line=dict(color="#5693e3", width=1.8)))
    fig_hist.add_trace(go.Scatter(x=dates_plot, y=output_df["MA_10"],
                                  name="MA10", line=dict(color="#38bdf8", width=1.2, dash="dot")))
    fig_hist.add_trace(go.Scatter(x=dates_plot, y=output_df["MA_50"],
                                  name="MA50", line=dict(color="#f472b6", width=1.2, dash="dash")))
    # Group contiguous bull/bear runs — only ~10-20 vrects instead of ~1500
    region_start = dates_plot[0]
    region_bull  = bullish.iloc[0]
    for i in range(1, len(output_df)):
        changed  = (bullish.iloc[i] != region_bull)
        last_row = (i == len(output_df) - 1)
        if changed or last_row:
            clr = "rgba(34,197,94,0.10)" if region_bull else "rgba(239,68,68,0.10)"
            fig_hist.add_vrect(x0=region_start, x1=dates_plot[i],
                               fillcolor=clr, layer="below", line_width=0)
            region_start = dates_plot[i]
            region_bull  = bullish.iloc[i]
    fig_hist.update_layout(**PLOTLY_LAYOUT, height=420,
                           title=dict(text=f"{ticker} — Bullish/Bearish Trend Regions (Full History)",
                                      font=dict(family="Syne,sans-serif", size=15, color="#61a4e7")))
    st.plotly_chart(fig_hist, width='stretch')

    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

    # ── Section B: RSI Chart ─────────────────────────────────────────────────
    st.markdown("### 📡 RSI Indicator")
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=dates_plot, y=output_df["RSI"],
                                 name="RSI", line=dict(color="#838ded", width=1.5)))
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="rgba(239,68,68,0.8)",
                      annotation_text="⚠ Overbought (70)",
                      annotation_position="top left",
                      annotation_font=dict(color="#ef4444", size=12, family="JetBrains Mono"))
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="rgba(34,197,94,0.8)",
                      annotation_text="✓ Oversold (30)",
                      annotation_position="bottom left",
                      annotation_font=dict(color="#22c55e", size=12, family="JetBrains Mono"))
    fig_rsi.add_hrect(y0=70, y1=100, fillcolor="rgba(239,68,68,0.05)", layer="below", line_width=0)
    fig_rsi.add_hrect(y0=0,  y1=30,  fillcolor="rgba(34,197,94,0.05)",  layer="below", line_width=0)
    fig_rsi.update_layout(**PLOTLY_LAYOUT, height=300,
                          title=dict(text="RSI — Relative Strength Index",
                                     font=dict(family="Syne,sans-serif", size=15, color="#60aaf5")))
    fig_rsi.update_layout(margin=dict(l=40, r=40, t=50, b=40))
    fig_rsi.update_yaxes(range=[0, 100])
    st.plotly_chart(fig_rsi, width='stretch')

    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

    # ── Section C: Directional Accuracy Comparison ───────────────────────────
    st.markdown("### 🎯 Directional Accuracy Comparison")

    summary_data = {
        "Method":         ["LR (naive)","RF (naive)","GB (naive)","XGB (naive)",
                           "Logistic Clf","RF Classifier","GB Classifier"],
        "Accuracy (%)":   [lin_naive_acc, rf_naive_acc, gb_naive_acc, xgb_naive_acc,
                           log_acc, rf_clf_acc, gb_clf_acc],
        "Type":           ["Regressor→Trend"]*4 + ["Dedicated Classifier"]*3
    }
    summary_df = pd.DataFrame(summary_data).sort_values("Accuracy (%)", ascending=False).reset_index(drop=True)

    left, right = st.columns([1, 1])
    with left:
        st.dataframe(summary_df.round(2), width='stretch')
    with right:
        base_colors  = ["#475569"]*4 + ["#38bdf8","#818cf8","#6366f1"]
        method_order = summary_data["Method"]
        sorted_colors= [base_colors[method_order.index(m)] for m in summary_df["Method"]]
        fig_acc = go.Figure()
        fig_acc.add_trace(go.Bar(
            x=summary_df["Method"],
            y=summary_df["Accuracy (%)"],
            marker_color=sorted_colors,
            text=summary_df["Accuracy (%)"].round(1).astype(str) + "%",
            textposition="outside"
        ))
        fig_acc.add_hline(y=50, line_dash="dash", line_color="rgba(239,68,68,0.5)",
                          annotation_text="Random (50%)")
        fig_acc.add_hline(y=60, line_dash="dot",  line_color="rgba(251,191,36,0.5)",
                          annotation_text="Good (60%)")
        fig_acc.update_layout(
            **PLOTLY_LAYOUT, height=320,
            title=dict(text="Directional Accuracy: Naive vs Classifiers",
                       font=dict(family="Syne,sans-serif", size=15, color="#6bb0f6"))
        )
        fig_acc.update_yaxes(range=[40, min(100, summary_df["Accuracy (%)"].max() + 10)])
        st.plotly_chart(fig_acc, width='stretch')

    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

    # ── Section D: Test Set Trend Dots ───────────────────────────────────────
    st.markdown(f"### 🔵 Test Set Predictions — {best_clf_name}")
    test_idx  = model_df.index[val_end:]
    correct_m = best_clf_pred == y_cls_test.values
    wrong_m   = ~correct_m

    fig_dots = go.Figure()
    fig_dots.add_trace(go.Scatter(x=test_idx, y=y_test.values,
                                  name="Actual Price", line=dict(color="#94a3b8", width=2)))
    fig_dots.add_trace(go.Scatter(
        x=test_idx[correct_m], y=y_test.values[correct_m],
        mode="markers", name="Correct Trend",
        marker=dict(color="#22c55e", size=7, symbol="circle",
                    line=dict(color="white", width=0.8))
    ))
    fig_dots.add_trace(go.Scatter(
        x=test_idx[wrong_m], y=y_test.values[wrong_m],
        mode="markers", name="Wrong Trend",
        marker=dict(color="#ef4444", size=7, symbol="circle",
                    line=dict(color="white", width=0.8))
    ))
    fig_dots.update_layout(
        **PLOTLY_LAYOUT, height=420,
        title=dict(
            text=f"{best_clf_name} — Correct (green) vs Wrong (red) Trend  |  Acc: {best_clf_acc:.1f}%",
            font=dict(family="Syne,sans-serif", size=14, color="#71aae2")
        )
    )
    st.plotly_chart(fig_dots, width='stretch')

    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

    # ── Section E: 7-Day Forecast ────────────────────────────────────────────
    st.markdown("### 🚀 7-Day Forecast with Trend Direction")

    reg_acc_map = {
        "Linear Regression": accuracy_score(actual_trend_test, (lin_test_pred  > actual_prices_test).astype(int)),
        "Random Forest":     accuracy_score(actual_trend_test, (rf_test_pred   > actual_prices_test).astype(int)),
        "Gradient Boosting": accuracy_score(actual_trend_test, (gb_test_pred   > actual_prices_test).astype(int)),
        "XGBoost":           accuracy_score(actual_trend_test, (xgb_test_pred  > actual_prices_test).astype(int)),
    }
    best_reg_name = max(reg_acc_map, key=reg_acc_map.get)
    reg_obj_map   = {"Linear Regression": lin, "Random Forest": rf_reg,
                     "Gradient Boosting": gb_reg, "XGBoost": xgb_reg}
    best_reg_obj  = reg_obj_map[best_reg_name]
    use_sc        = best_reg_name != "Random Forest"

    forecast_df_loop = output_df.copy()
    future_prices    = []
    for _ in range(7):
        latest   = forecast_df_loop.iloc[-1]
        feat_row = latest[feature_cols].to_frame().T
        pred = best_reg_obj.predict(scaler.transform(feat_row))[0] if use_sc \
               else best_reg_obj.predict(feat_row)[0]
        future_prices.append(pred)

        nr = latest.copy()
        nr["Close_cleaned"] = pred
        nr["Lag3"] = latest["Lag2"]; nr["Lag2"] = latest["Lag1"]; nr["Lag1"] = pred
        forecast_df_loop = pd.concat([forecast_df_loop, pd.DataFrame([nr])])
        forecast_df_loop["Return"]     = forecast_df_loop["Close_cleaned"].pct_change()
        forecast_df_loop["MA_10"]      = forecast_df_loop["Close_cleaned"].rolling(10).mean()
        forecast_df_loop["MA_20"]      = forecast_df_loop["Close_cleaned"].rolling(20).mean()
        forecast_df_loop["MA_50"]      = forecast_df_loop["Close_cleaned"].rolling(50).mean()
        forecast_df_loop["Volatility"] = forecast_df_loop["Return"].rolling(10).std()
        forecast_df_loop["Momentum"]   = forecast_df_loop["Close_cleaned"] - forecast_df_loop["Close_cleaned"].shift(5)
        d = forecast_df_loop["Close_cleaned"].diff()
        g = d.where(d > 0, 0); lo = -d.where(d < 0, 0)
        forecast_df_loop["RSI"] = 100 - (100 / (1 + g.rolling(14).mean() / lo.rolling(14).mean()))

    last_date    = model_df.index[-1]
    future_dates = pd.date_range(start=last_date + pd.tseries.offsets.BDay(1), periods=7, freq="B")
    prev_prices  = [output_df["Close_cleaned"].iloc[-1]] + future_prices[:-1]
    trend_labels = ["▲ Up" if future_prices[i] > prev_prices[i] else "▼ Down" for i in range(7)]
    trend_colors = ["#22c55e" if "Up" in t else "#ef4444" for t in trend_labels]

    forecast_result = pd.DataFrame({
        "Date":           future_dates.strftime("%Y-%m-%d"),
        "Forecast Price": [round(p, 2) for p in future_prices],
        "Change":         [round(future_prices[i] - prev_prices[i], 2) for i in range(7)],
        "Trend":          trend_labels
    })

    left, right = st.columns([1, 2])
    up_count   = trend_labels.count("▲ Up")
    down_count = trend_labels.count("▼ Down")
    with left:
        st.dataframe(forecast_result, width='stretch', height=300)
        st.markdown(f"""
        <div class="info-box" style="margin-top:12px;">
            <strong>7-Day Outlook</strong><br>
            📈 Up days: <span style="color:#22c55e;font-weight:600;">{up_count}</span> &nbsp;|&nbsp;
            📉 Down days: <span style="color:#ef4444;font-weight:600;">{down_count}</span><br>
            Overall bias: <strong style="color:{'#22c55e' if up_count >= down_count else '#ef4444'};">
            {'🐂 Bullish' if up_count >= down_count else '🐻 Bearish'}</strong>
        </div>
        """, unsafe_allow_html=True)
    with right:
        fig_fore = go.Figure()
        hist_tail = output_df.tail(60)
        fig_fore.add_trace(go.Scatter(x=hist_tail.index, y=hist_tail["Close_cleaned"],
                                      name="Historical", line=dict(color="#475569", width=2)))
        fig_fore.add_trace(go.Scatter(
            x=pd.to_datetime(forecast_result["Date"]),
            y=forecast_result["Forecast Price"],
            name="Forecast", mode="lines+markers",
            line=dict(color="#38bdf8", width=2.5, dash="dash"),
            marker=dict(color=trend_colors, size=10, line=dict(color="white", width=1.5))
        ))
        fig_fore.update_layout(
            **PLOTLY_LAYOUT, height=360,
            title=dict(text="7-Day Forecast — Green=Up, Red=Down",
                       font=dict(family="Syne,sans-serif", size=15, color="#61acf7"))
        )
        st.plotly_chart(fig_fore, width='stretch')

    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

    # ── Section F: Classification Report ─────────────────────────────────────
    st.markdown(f"### 📋 Classification Report — {best_clf_name}")
    report    = classification_report(y_cls_test, best_clf_pred,
                                      target_names=["Down","Up"],
                                      zero_division=0, output_dict=True)
    report_df = pd.DataFrame(report).transpose().round(3)

    left, right = st.columns([1, 2])
    with left:
        st.dataframe(report_df, width='stretch')
    with right:
        st.markdown(f"""
        <div class="section-card section-card-accent">
            <div style="font-family:Syne,sans-serif;font-weight:700;color:#38bdf8;font-size:1.05rem;margin-bottom:14px;">
                Trend Analysis Summary
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;">
                <div style="background:rgba(56,189,248,0.08);border-radius:10px;padding:12px;">
                    <div style="color:#64748b;font-size:0.72rem;text-transform:uppercase;letter-spacing:1px;">Best Classifier</div>
                    <div style="color:#38bdf8;font-family:'JetBrains Mono',monospace;font-weight:600;margin-top:4px;">{best_clf_name}</div>
                </div>
                <div style="background:rgba(34,197,94,0.08);border-radius:10px;padding:12px;">
                    <div style="color:#64748b;font-size:0.72rem;text-transform:uppercase;letter-spacing:1px;">Directional Acc</div>
                    <div style="color:#22c55e;font-family:'JetBrains Mono',monospace;font-weight:600;margin-top:4px;">{best_clf_acc:.2f}%</div>
                </div>
                <div style="background:rgba(129,140,248,0.08);border-radius:10px;padding:12px;">
                    <div style="color:#64748b;font-size:0.72rem;text-transform:uppercase;letter-spacing:1px;">Improvement</div>
                    <div style="color:#818cf8;font-family:'JetBrains Mono',monospace;font-weight:600;margin-top:4px;">+{best_clf_acc - lin_naive_acc:.2f}% vs LR</div>
                </div>
                <div style="background:rgba(244,114,182,0.08);border-radius:10px;padding:12px;">
                    <div style="color:#64748b;font-size:0.72rem;text-transform:uppercase;letter-spacing:1px;">7-Day Bias</div>
                    <div style="color:{'#22c55e' if up_count >= 4 else '#ef4444'};font-family:'JetBrains Mono',monospace;font-weight:600;margin-top:4px;">
                        {'🐂 Bullish' if up_count >= 4 else '🐻 Bearish'}
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
