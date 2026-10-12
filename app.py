"""
MarketOpt – Intelligent Marketing Budget Optimization and ROI Prediction System
Run: streamlit run app.py
"""
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.neighbors import KNeighborsRegressor

# ----------------------------------------------------------------------------
# SUPABASE CLIENT INITIALIZATION
# ----------------------------------------------------------------------------
try:
    from supabase import create_client
    supabase_url = st.secrets.get("SUPABASE_URL", "")
    supabase_key = st.secrets.get("SUPABASE_KEY", "")
    supabase = create_client(supabase_url, supabase_key) if supabase_url and supabase_key else None
except Exception:
    supabase = None

st.set_page_config(
    page_title="MarketOpt – Marketing Budget Optimization",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------------------------------
# SESSION STATE INITIALIZATION
# ----------------------------------------------------------------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user_email" not in st.session_state:
    st.session_state.user_email = ""
if "user_name" not in st.session_state:
    st.session_state.user_name = ""
if "auth_tab" not in st.session_state:
    st.session_state.auth_tab = "login"

# ----------------------------------------------------------------------------
# PREMIUM AUTHENTICATION SCREEN (SHOWN WHEN NOT LOGGED IN)
# ----------------------------------------------------------------------------
if not st.session_state.authenticated:
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stApp {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    background: #0B1120 !important;
}
#MainMenu, footer, header[data-testid="stHeader"] {
    display: none !important;
    visibility: hidden !important;
}
.block-container {
    padding: 2.5rem 1rem !important;
    max-width: 460px !important;
    margin: 0 auto !important;
}

/* Glassmorphism Card Container */
.auth-box {
    background: linear-gradient(180deg, rgba(30, 41, 59, 0.85) 0%, rgba(15, 23, 42, 0.95) 100%);
    border: 1px solid rgba(56, 189, 248, 0.2);
    border-radius: 24px;
    padding: 34px 32px 30px;
    box-shadow: 0 25px 60px -15px rgba(0, 0, 0, 0.7), 0 0 30px -10px rgba(37, 99, 235, 0.25);
    backdrop-filter: blur(16px);
    margin-top: 10px;
}

.auth-header {
    text-align: center;
    margin-bottom: 22px;
}
.auth-icon {
    font-size: 2.6rem;
    display: inline-block;
    filter: drop-shadow(0 4px 12px rgba(37, 99, 235, 0.4));
    margin-bottom: 4px;
}
.auth-brand {
    font-size: 2.1rem;
    font-weight: 800;
    color: #FFFFFF;
    letter-spacing: -.03em;
    margin: 0;
    line-height: 1.1;
}
.auth-sub {
    font-size: .8rem;
    color: #38BDF8;
    font-weight: 600;
    letter-spacing: .02em;
    margin-top: 6px;
}
.auth-credit {
    font-size: .74rem;
    color: #94A3B8;
    margin-top: 4px;
}

/* Auth Tabs */
.auth-tab-row {
    display: flex;
    gap: 8px;
    margin-bottom: 20px;
}

/* Input Fields */
.stTextInput > div > div > input {
    background: #0F172A !important;
    border: 1px solid #334155 !important;
    border-radius: 12px !important;
    color: #F8FAFC !important;
    font-size: .92rem !important;
    padding: 10px 14px !important;
}
.stTextInput > div > div > input:focus {
    border-color: #38BDF8 !important;
    box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.2) !important;
}
.stTextInput label p {
    color: #94A3B8 !important;
    font-weight: 600 !important;
    font-size: .82rem !important;
}

/* Submit Button */
.stButton > button {
    background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
    color: #FFFFFF !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    font-size: .96rem !important;
    padding: 11px 20px !important;
    width: 100% !important;
    box-shadow: 0 8px 20px -4px rgba(37, 99, 235, 0.5) !important;
    transition: all .2s ease !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 12px 25px -4px rgba(37, 99, 235, 0.6) !important;
}

.auth-hint {
    background: rgba(15, 23, 42, 0.6);
    border: 1px dashed #334155;
    border-radius: 10px;
    padding: 10px 14px;
    font-size: .78rem;
    color: #94A3B8;
    margin-top: 18px;
    text-align: center;
}
.auth-hint b {
    color: #38BDF8;
}
</style>
""", unsafe_allow_html=True)

    st.markdown("""
    <div class="auth-header">
      <div class="auth-icon">📈</div>
      <h1 class="auth-brand">MarketOpt</h1>
      <div class="auth-sub">Intelligent Marketing Budget Optimization</div>
      <div class="auth-credit">Made by Lithish Balaji</div>
    </div>
    """, unsafe_allow_html=True)

    c_tab1, c_tab2 = st.columns(2)
    with c_tab1:
        if st.button("🔑 Log In", key="tab_login", use_container_width=True):
            st.session_state.auth_tab = "login"
            st.rerun()
    with c_tab2:
        if st.button("✨ Sign Up", key="tab_signup", use_container_width=True):
            st.session_state.auth_tab = "signup"
            st.rerun()

    if st.session_state.auth_tab == "login":
        st.markdown("<p style='color:#E2E8F0; font-size:.95rem; font-weight:700; margin:14px 0 8px;'>Sign in to your account</p>", unsafe_allow_html=True)
        with st.form("form_login"):
            email = st.text_input("Username or Email", placeholder="owner or you@example.com")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            submitted = st.form_submit_button("Sign In to MarketOpt ➔", use_container_width=True)

        if submitted:
            if not email or not password:
                st.error("Please enter both username/email and password.")
            # Master Owner Bypass
            elif email.strip().lower() == "owner" and password == "Owner@123":
                st.session_state.authenticated = True
                st.session_state.user_email = "owner@marketopt.internal"
                st.session_state.user_name = "Owner (Lithish Balaji)"
                st.rerun()
            elif supabase is None:
                st.error("Supabase client not configured in secrets.")
            else:
                try:
                    res = supabase.auth.sign_in_with_password({"email": email.strip(), "password": password})
                    if res.user:
                        st.session_state.authenticated = True
                        st.session_state.user_email = res.user.email
                        st.session_state.user_name = res.user.user_metadata.get("full_name") or res.user.email.split("@")[0]
                        st.rerun()
                except Exception as err:
                    st.error(f"Login failed: {err}")

        st.markdown("""
        <div class="auth-hint">
          💡 <b>Admin Access:</b> Username <code>owner</code> | Password <code>Owner@123</code>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("<p style='color:#E2E8F0; font-size:.95rem; font-weight:700; margin:14px 0 8px;'>Create a new account</p>", unsafe_allow_html=True)
        with st.form("form_signup"):
            full_name = st.text_input("Full Name", placeholder="e.g. Lithish Balaji")
            email = st.text_input("Email Address", placeholder="you@example.com")
            password = st.text_input("Password", type="password", placeholder="At least 6 characters")
            confirm_pw = st.text_input("Confirm Password", type="password", placeholder="Repeat your password")
            submitted = st.form_submit_button("Create Free Account ➔", use_container_width=True)

        if submitted:
            if not all([full_name, email, password, confirm_pw]):
                st.error("Please fill in all fields.")
            elif password != confirm_pw:
                st.error("Passwords do not match.")
            elif len(password) < 6:
                st.error("Password must be at least 6 characters.")
            elif supabase is None:
                st.error("Supabase client not configured in secrets.")
            else:
                try:
                    res = supabase.auth.sign_up({
                        "email": email.strip(),
                        "password": password,
                        "options": {"data": {"full_name": full_name.strip()}}
                    })
                    if res.user and res.session:
                        st.session_state.authenticated = True
                        st.session_state.user_email = res.user.email
                        st.session_state.user_name = full_name.strip()
                        st.rerun()
                    else:
                        st.success("Account created! Please check your email for confirmation if enabled, then log in.")
                        st.session_state.auth_tab = "login"
                        st.rerun()
                except Exception as err:
                    st.error(f"Sign up failed: {err}")

    st.stop()


# ----------------------------------------------------------------------------
# 1. DATA & BENCHMARKS
# ----------------------------------------------------------------------------
CHANNELS = ["Google Ads", "YouTube", "Instagram", "Facebook", "Newspaper", "Influencer"]
CATEGORIES = [
    "Clothing & Fashion", "Electronics", "FMCG (Food & Beverages)", "Real Estate",
    "Restaurant", "Software / SaaS", "Tourism & Travel"
]
TIERS = ["Budget", "Mid-Range", "Premium"]
STYLES = ["Digital-First", "Traditional-Mixed"]

# Baseline distribution (sums to 100)
BASE = {
    "Clothing & Fashion":      [12, 12, 38, 14, 4, 20],
    "Electronics":             [28, 24, 14, 14, 4, 16],
    "FMCG (Food & Beverages)": [14, 16, 20, 22, 14, 14],
    "Real Estate":             [30, 10, 12, 24, 20, 4],
    "Restaurant":              [14, 6, 34, 20, 6, 20],
    "Software / SaaS":         [36, 18, 10, 14, 2, 20],
    "Tourism & Travel":        [22, 18, 24, 16, 6, 14],
}

STYLE_MULT = {
    "Digital-First":     [1.00, 1.00, 1.00, 1.00, 0.15, 1.00],
    "Traditional-Mixed": [1.00, 1.00, 0.85, 1.10, 2.50, 0.80],
}

TIER_MULT = {
    "Budget":    [1.05, 1.00, 1.00, 1.00, 0.60, 1.10],
    "Mid-Range": [1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    "Premium":   [1.00, 1.10, 1.10, 1.00, 1.20, 1.10],
}

# Average CPC benchmarks in INR
CHANNEL_METRICS = {
    "Google Ads":  {"cpc": 28.0, "icon": "🔍", "desc": "High-intent search traffic"},
    "YouTube":     {"cpc": 12.0, "icon": "▶️", "desc": "Engaged video stream clicks"},
    "Instagram":   {"cpc": 14.0, "icon": "📸", "desc": "Reels & visual story clicks"},
    "Facebook":    {"cpc": 11.0, "icon": "👥", "desc": "Feed link & community clicks"},
    "Influencer":  {"cpc": 22.0, "icon": "⭐", "desc": "Creator referral conversions"},
    "Newspaper":   {"cpc": None, "icon": "📰", "desc": "Offline print & traditional circulation"},
}

CAT_CONV = {
    "Clothing & Fashion":      0.025,
    "Electronics":             0.015,
    "FMCG (Food & Beverages)": 0.035,
    "Real Estate":             0.005,
    "Restaurant":              0.045,
    "Software / SaaS":         0.018,
    "Tourism & Travel":        0.010,
}

TIER_MODIFIERS = {
    "Budget":    1.15,
    "Mid-Range": 1.00,
    "Premium":   0.85,
}

SCENARIO_MULTS = {
    "Cautious":   0.65,
    "Expected":   1.00,
    "Optimistic": 1.45,
}

# ----------------------------------------------------------------------------
# 2. MACHINE LEARNING MODEL
# ----------------------------------------------------------------------------
@st.cache_resource
def build_model():
    rng = np.random.default_rng(42)
    rows = []
    for c in CATEGORIES:
        for t in TIERS:
            for s in STYLES:
                v = np.array(BASE[c], float) * np.array(TIER_MULT[t]) * np.array(STYLE_MULT[s])
                v = v / v.sum() * 100
                for _ in range(12):
                    n = np.clip(v + rng.normal(0, 0.5, len(v)), 0.2, None)
                    n = n / n.sum() * 100
                    rows.append([c, t, s, *n])
    df = pd.DataFrame(rows, columns=["category", "tier", "style", *CHANNELS])
    X = pd.get_dummies(df[["category", "tier", "style"]]).astype(float)
    model = KNeighborsRegressor(n_neighbors=8).fit(X, df[CHANNELS])
    return model, list(X.columns)


def predict_split(category, tier, style):
    model, cols = build_model()
    row = pd.DataFrame([{c: 0.0 for c in cols}])
    for key in (f"category_{category}", f"tier_{tier}", f"style_{style}"):
        if key in row.columns:
            row.loc[0, key] = 1.0
    p = model.predict(row)[0]
    return p / p.sum() * 100


# ----------------------------------------------------------------------------
# 3. FORMATTING HELPERS & THEME
# ----------------------------------------------------------------------------
COLORS = {
    "Google Ads": "#2563EB",
    "YouTube": "#DC2626",
    "Instagram": "#DB2777",
    "Facebook": "#0E7490",
    "Newspaper": "#78716C",
    "Influencer": "#D97706"
}
INK, MUTED, LINE, PAPER = "#0F172A", "#64748B", "#E2E8F0", "#F8FAFC"


def inr(x):
    x = int(round(x))
    s = str(abs(x))
    if len(s) > 3:
        head, tail, parts = s[:-3], s[-3:], []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        s = ",".join(parts + [tail])
    return ("-" if x < 0 else "") + "₹" + s


def short(x):
    if x >= 1e7:
        return f"₹{x / 1e7:.2f} Cr"
    if x >= 1e5:
        return f"₹{x / 1e5:.2f} L"
    if x >= 1e3:
        return f"₹{x / 1e3:.1f} K"
    return f"₹{x:.0f}"


def count(x):
    if x >= 1e7:
        return f"{x / 1e7:.2f} Cr"
    if x >= 1e5:
        return f"{x / 1e5:.2f} L"
    if x >= 1e3:
        return f"{x / 1e3:.1f} K"
    return f"{x:.0f}"


# ----------------------------------------------------------------------------
# 4. STYLING (MODERN DASHBOARD UI & SLIDING SIDEBAR)
# ----------------------------------------------------------------------------
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stApp {{
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}}
.stApp {{ background: {PAPER}; }}
#MainMenu, footer {{ visibility: hidden; height: 0; }}
header[data-testid="stHeader"] {{ background: transparent !important; }}
.block-container {{ padding: 1.2rem 2.2rem 3rem; max-width: 1320px; }}

/* High-contrast dashboard text */
.sec {{ font-size: 1.25rem; font-weight: 800; color: #0F172A !important; margin: 26px 0 14px; letter-spacing: -.01em; }}
.hero * {{ color: #FFFFFF; }}
.hero .lbl {{ color: #94A3B8 !important; }}
.rev-box * {{ color: #FFFFFF; }}
.rev-label {{ color: #94A3B8 !important; }}
.kpi .k {{ font-size: .78rem; color: #64748B !important; font-weight: 600; text-transform: uppercase; letter-spacing: .03em; }}
.kpi .v {{ font-size: 1.7rem; font-weight: 800; color: #0F172A !important; letter-spacing: -.02em; margin-top: 4px; }}
.kpi .s {{ font-size: .8rem; color: #64748B !important; margin-top: 3px; }}
.ch .n {{ font-weight: 700; color: #0F172A !important; font-size: 1rem; }}
.ch .p {{ font-size: 1.85rem; font-weight: 800; letter-spacing: -.03em; color: #0F172A !important; margin: 4px 0 0; }}
.ch .a {{ font-size: .95rem; color: #475569 !important; font-weight: 600; }}
.ch .r {{ font-size: .8rem; color: #64748B !important; margin-top: 6px; line-height: 1.45; }}

/* Sidebar Appearance */
[data-testid="stSidebar"] {{
    display: block !important;
    visibility: visible !important;
    background: #0B1120 !important;
    border-right: 1px solid #1E293B !important;
}}
[data-testid="stSidebar"] * {{ color: #F1F5F9; }}
[data-testid="stSidebar"] label p {{ font-weight: 600; font-size: .84rem; color: #94A3B8 !important; }}
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-baseweb="input"] > div,
[data-testid="stSidebar"] [data-baseweb="base-input"] {{ background: #1E293B; border-color: #334155; border-radius: 8px; }}
[data-testid="stSidebar"] input {{ color: #FFFFFF; font-weight: 600; }}

/* SLIDING SIDEBAR CONTROLS: Arrow mark to open & close */
/* Arrow button on output screen to slide input panel open */
[data-testid="collapsedControl"] {{
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    top: 14px !important;
    left: 14px !important;
    background: #0B1120 !important;
    border: 1px solid #334155 !important;
    border-radius: 10px !important;
    color: #FFFFFF !important;
    box-shadow: 0 4px 14px rgba(0,0,0,0.3) !important;
    z-index: 999999 !important;
    padding: 4px !important;
    cursor: pointer !important;
}}
[data-testid="collapsedControl"]:hover {{
    background: #1E293B !important;
    border-color: #38BDF8 !important;
}}
[data-testid="collapsedControl"] svg {{
    fill: #FFFFFF !important;
    stroke: #FFFFFF !important;
    width: 22px !important;
    height: 22px !important;
}}

/* Arrow button inside sidebar to slide input panel closed */
[data-testid="stSidebarCollapseButton"] {{
    display: flex !important;
    visibility: visible !important;
}}
[data-testid="stSidebarCollapseButton"] button {{
    color: #94A3B8 !important;
    background: #1E293B !important;
    border-radius: 8px !important;
    border: 1px solid #334155 !important;
}}
[data-testid="stSidebarCollapseButton"] button:hover {{
    color: #FFFFFF !important;
    background: #334155 !important;
    border-color: #38BDF8 !important;
}}
[data-testid="stSidebarCollapseButton"] svg {{
    fill: #E2E8F0 !important;
}}

/* Sidebar Brand Header */
.brand {{ font-weight: 800; font-size: 1.35rem; letter-spacing: -.02em; color: #FFFFFF; line-height: 1.2; margin-bottom: 2px; }}
.brand-credits {{ font-size: .76rem; color: #38BDF8; font-weight: 600; margin-bottom: 6px; letter-spacing: .01em; }}
.brand-sub {{ font-size: .74rem; color: #94A3B8; margin-bottom: 1.4rem; line-height: 1.4; border-bottom: 1px solid #1E293B; padding-bottom: 12px; }}

/* Top Navbar for Logged In User */
.top-navbar {{
    display: flex;
    justify-content: flex-end;
    align-items: center;
    gap: 12px;
    margin-bottom: 14px;
}}
.user-pill {{
    background: #0B1120;
    border: 1px solid #1E293B;
    padding: 6px 14px;
    border-radius: 999px;
    font-size: .84rem;
    font-weight: 600;
    color: #F8FAFC;
    box-shadow: 0 2px 6px rgba(0,0,0,0.06);
}}

.hero {{ background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%); border-radius: 20px;
        padding: 26px 30px; color: #fff; margin-bottom: 22px; box-shadow: 0 10px 25px -5px rgba(0,0,0,0.1); }}
.hero-top {{ display: flex; justify-content: space-between; align-items: flex-end; flex-wrap: wrap; gap: 14px; }}
.hero h1 {{ font-size: 2.6rem; font-weight: 800; letter-spacing: -.03em; margin: 0; line-height: 1.1; color: #fff; }}
.hero .lbl {{ font-size: .86rem; color: #94A3B8; margin-bottom: 4px; font-weight: 500; }}
.chips span {{ display: inline-block; background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.16);
              padding: 5px 14px; border-radius: 999px; font-size: .82rem; font-weight: 600; margin-left: 6px; color: #E2E8F0; }}
.splitbar {{ display: flex; height: 44px; border-radius: 10px; overflow: hidden; margin-top: 22px; gap: 2px; }}
.splitbar div {{ display: flex; align-items: center; justify-content: center; font-size: .78rem;
                font-weight: 700; color: #fff; min-width: 2px; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 16px; margin-top: 14px; font-size: .82rem; color: #CBD5E1; }}
.legend i {{ display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: 6px; }}

.kpi {{ background: #fff; border: 1px solid {LINE}; border-radius: 16px; padding: 18px 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.03); }}
.ch {{ background: #fff; border: 1px solid {LINE}; border-radius: 16px; padding: 18px 20px; border-top: 4px solid; height: 100%; box-shadow: 0 2px 8px rgba(0,0,0,0.02); }}

/* REVENUE IMPACT CARD */
.rev-box {{ background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%); border-radius: 18px; padding: 26px 30px; color: #fff; margin-bottom: 22px; box-shadow: 0 10px 25px -5px rgba(0,0,0,0.12); border: 1px solid #334155; }}
.rev-header {{ display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px; margin-bottom: 20px; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 14px; }}
.rev-header h3 {{ margin: 0; font-size: 1.3rem; font-weight: 800; color: #fff; }}
.rev-badge {{ background: #10B981; color: #fff; font-weight: 700; font-size: .92rem; padding: 6px 16px; border-radius: 999px; }}
.rev-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 18px; }}
.rev-item {{ background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); padding: 16px 20px; border-radius: 14px; }}
.rev-val {{ font-size: 1.85rem; font-weight: 800; letter-spacing: -.02em; margin-top: 6px; }}

/* CALCULATION BREAKDOWN PANEL */
.calc-panel {{ background: #FFFFFF; border: 1px solid {LINE}; border-radius: 18px; padding: 24px 28px; box-shadow: 0 4px 16px rgba(0,0,0,0.03); margin-bottom: 24px; }}
.calc-banner {{ display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px; margin-bottom: 18px; padding-bottom: 14px; border-bottom: 1px solid {LINE}; }}
.calc-title {{ font-size: 1.25rem; font-weight: 800; color: {INK}; display: flex; align-items: center; gap: 10px; }}
.calc-step {{ background: #F8FAFC; border-left: 4px solid #2563EB; padding: 14px 18px; border-radius: 0 12px 12px 0; margin: 12px 0; font-size: .92rem; color: #1E293B; line-height: 1.6; border: 1px solid #E2E8F0; border-left: 4px solid #2563EB; }}
.calc-step b {{ color: {INK}; }}
.calc-math-box {{ background: #FFFFFF; border: 1px solid #CBD5E1; padding: 8px 14px; border-radius: 8px; font-family: 'JetBrains Mono', 'Courier New', monospace; font-size: .88rem; margin-top: 6px; display: inline-block; }}
.stDownloadButton button {{ background: {INK}; color: #fff; border-radius: 10px; border: 0; font-weight: 600; padding: .6rem 1.4rem; }}
.stDownloadButton button:hover {{ background: #1E293B; color: #fff; }}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# 5. TOP NAVBAR (USER BADGE & LOGOUT)
# ----------------------------------------------------------------------------
c_spacer, c_user, c_out = st.columns([7.2, 2.8, 1.2])

with c_user:
    st.markdown(f"""
    <div style="text-align: right; font-size: .86rem; color: #0F172A; font-weight: 700; padding-top: 6px;">
      👤 <span>{st.session_state.user_name}</span>
    </div>
    """, unsafe_allow_html=True)

with c_out:
    if st.button("🚪 Logout", key="btn_logout", use_container_width=True):
        if supabase:
            try:
                supabase.auth.sign_out()
            except Exception:
                pass
        st.session_state.authenticated = False
        st.session_state.user_email = ""
        st.session_state.user_name = ""
        st.session_state.auth_tab = "login"
        st.rerun()


# ----------------------------------------------------------------------------
# 6. SIDEBAR: INPUT CONTROLS (SLIDES IN AND OUT VIA ARROW BUTTONS)
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown('''
    <div class="brand">MarketOpt</div>
    <div class="brand-credits">Made by Lithish Balaji</div>
    <div class="brand-sub">Intelligent Marketing Budget Optimization and ROI Prediction System</div>
    ''', unsafe_allow_html=True)

    category = st.selectbox("1. Business Category", CATEGORIES, index=0)
    budget = st.number_input("2. Total Marketing Budget (₹)", min_value=10000, max_value=100000000, value=500000, step=25000)
    tier = st.selectbox("3. Budget Category", TIERS, index=1)
    style = st.radio("4. Media Type", STYLES, index=0)

    st.markdown("---")
    st.markdown("<b style='color:#F1F5F9;'>Revenue Impact Inputs</b>", unsafe_allow_html=True)
    rev_before = st.number_input(
        "Revenue Before Marketing (₹)",
        min_value=0,
        max_value=1000000000,
        value=1500000,
        step=50000,
        help="Enter current monthly revenue before executing this marketing plan"
    )
    rev_after = st.number_input(
        "Revenue After Marketing (₹)",
        min_value=0,
        max_value=1000000000,
        value=2750000,
        step=50000,
        help="Enter projected or realized monthly revenue after marketing execution"
    )


# ----------------------------------------------------------------------------
# 7. CALCULATIONS
# ----------------------------------------------------------------------------
# Budget shares prediction based on dataset & KNN
pct = predict_split(category, tier, style)
amounts_raw = np.round(pct / 100 * budget).astype(int)
amounts_raw[np.argmax(amounts_raw)] += int(budget - amounts_raw.sum())  # Keep exact sum
amounts = dict(zip(CHANNELS, amounts_raw))

# Effective conversion rate base for the business category and tier
effective_cvr = CAT_CONV[category] * TIER_MODIFIERS[tier]

# Per-channel calculations (Digital channels use average CPC & 3 Customer Increase scenarios; Newspaper is offline reach)
channel_data = []
for i, ch in enumerate(CHANNELS):
    ch_spend = amounts[ch]
    cpc = CHANNEL_METRICS[ch]["cpc"]

    if cpc is not None and cpc > 0:
        clicks = int(round(ch_spend / cpc))
        cautious_cust = clicks * effective_cvr * SCENARIO_MULTS["Cautious"]
        expected_cust = clicks * effective_cvr * SCENARIO_MULTS["Expected"]
        optimistic_cust = clicks * effective_cvr * SCENARIO_MULTS["Optimistic"]
    else:
        clicks = None
        cautious_cust = None
        expected_cust = None
        optimistic_cust = None

    channel_data.append({
        "Channel": ch,
        "Share (%)": round(pct[i], 1),
        "Spend": ch_spend,
        "CPC": cpc,
        "Clicks": clicks,
        "Cautious Customers": round(cautious_cust, 1) if cautious_cust is not None else None,
        "Expected Customers": round(expected_cust, 1) if expected_cust is not None else None,
        "Optimistic Customers": round(optimistic_cust, 1) if optimistic_cust is not None else None,
    })

df_results = pd.DataFrame(channel_data)

# Aggregated Totals across digital platforms
tot_spend = df_results["Spend"].sum()
tot_clicks = int(df_results["Clicks"].dropna().sum())
tot_cust_cautious = df_results["Cautious Customers"].dropna().sum()
tot_cust_expected = df_results["Expected Customers"].dropna().sum()
tot_cust_optimistic = df_results["Optimistic Customers"].dropna().sum()

# Revenue Impact Outputs calculated strictly from the two user inputs & total marketing budget
rev_increased = rev_after - rev_before
rev_increase_pct = (rev_increased / rev_before * 100) if rev_before > 0 else 0
computed_roas = (rev_increased / budget) if budget > 0 else 0

top_channel = df_results.sort_values("Share (%)", ascending=False).iloc[0]


# ----------------------------------------------------------------------------
# 8. HERO HEADER & TOP CHANNEL DISPLAY
# ----------------------------------------------------------------------------
bar = "".join(
    f'<div style="width:{p}%;background:{COLORS[c]}">{(f"{p:.0f}%" if p >= 7 else "")}</div>'
    for c, p in zip(CHANNELS, pct)
)
legend = "".join(f'<span><i style="background:{COLORS[c]}"></i>{c}</span>' for c in CHANNELS)

st.markdown(f"""
<div class="hero">
  <div class="hero-top">
    <div>
      <div class="lbl">MarketOpt Optimized Plan for Budget of</div>
      <h1>{inr(budget)}</h1>
    </div>
    <div class="chips">
      <span>🏢 {category}</span>
      <span>🏷️ {tier}</span>
      <span>⚡ {style}</span>
    </div>
  </div>
  <div class="splitbar">{bar}</div>
  <div class="legend">{legend}</div>
</div>
""", unsafe_allow_html=True)

# Top Channel & Summary KPI Cards
k1, k2, k3, k4 = st.columns(4)
k1.markdown(f'<div class="kpi"><div class="k">Top Channel</div><div class="v">{top_channel.Channel}</div><div class="s">{top_channel["Share (%)"]}% share ({inr(top_channel["Spend"])})</div></div>', unsafe_allow_html=True)
k2.markdown(f'<div class="kpi"><div class="k">Total Paid Clicks</div><div class="v">{count(tot_clicks)}</div><div class="s">Across all digital platforms</div></div>', unsafe_allow_html=True)
k3.markdown(f'<div class="kpi"><div class="k">Expected New Customers</div><div class="v">+{count(tot_cust_expected)}</div><div class="s">Conversion rate: {(effective_cvr*100):.2f}%</div></div>', unsafe_allow_html=True)
k4.markdown(f'<div class="kpi"><div class="k">Customer Growth Range</div><div class="v">+{count(tot_cust_cautious)} to +{count(tot_cust_optimistic)}</div><div class="s">Cautious vs Optimistic</div></div>', unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# 9. REVENUE IMPACT ANALYSIS AREA
# ----------------------------------------------------------------------------
st.markdown('<div class="sec">Revenue Impact Analysis</div>', unsafe_allow_html=True)
st.markdown(f"""
<div class="rev-box">
  <div class="rev-header">
    <div>
      <div style="font-size: .8rem; color: #94A3B8; text-transform: uppercase; font-weight: 600;">Performance & Return on Investment</div>
      <h3>Revenue Comparison & ROAS Output</h3>
    </div>
    <div class="rev-badge">{"+" if rev_increase_pct >= 0 else ""}{rev_increase_pct:.1f}% Revenue Increase</div>
  </div>
  <div class="rev-grid">
    <div class="rev-item">
      <div class="rev-label">Revenue Before Marketing (Input)</div>
      <div class="rev-val" style="color: #E2E8F0;">{inr(rev_before)}</div>
    </div>
    <div class="rev-item">
      <div class="rev-label">Revenue After Marketing (Input)</div>
      <div class="rev-val" style="color: #38BDF8;">{inr(rev_after)}</div>
    </div>
    <div class="rev-item">
      <div class="rev-label">Revenue Increased (Output)</div>
      <div class="rev-val" style="color: {'#4ADE80' if rev_increased >= 0 else '#F87171'};">
        {"+" if rev_increased >= 0 else ""}{inr(rev_increased)}
      </div>
    </div>
    <div class="rev-item">
      <div class="rev-label">Calculated ROAS (Output)</div>
      <div class="rev-val" style="color: #FBBF24;">{computed_roas:.2f}x</div>
      <div style="font-size: .75rem; color: #94A3B8; margin-top: 4px;">Incremental Rev ÷ Total Budget</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# 10. SECTION-BY-SECTION BUDGET ALLOCATION CARDS
# ----------------------------------------------------------------------------
st.markdown('<div class="sec">Allocated Budget & Predicted Customer Increase Per Section</div>', unsafe_allow_html=True)

cols = st.columns(3)
for i, row in df_results.iterrows():
    c = row["Channel"]
    with cols[i % 3]:
        if row["CPC"] is not None:
            metric_snippet = f"""<b>Avg CPC:</b> ₹{row['CPC']:.1f} &nbsp;|&nbsp; <b>Clicks:</b> {count(row['Clicks'])}<br>
            <span style="color:#059669; font-weight:700;">Expected Increase: +{count(row['Expected Customers'])} customers</span>"""
        else:
            metric_snippet = """<b>Media Type:</b> Offline Print & Circulation<br>
            <span style="color:#64748B; font-weight:600;">Brand Awareness & Local Trust</span>"""

        st.markdown(f"""
        <div class="ch" style="border-top-color:{COLORS[c]}">
          <div class="n">{CHANNEL_METRICS[c]["icon"]} {c}</div>
          <div class="p">{row['Share (%)']}%</div>
          <div class="a">{inr(row['Spend'])}</div>
          <div class="r" style="margin-top: 8px;">
            {metric_snippet}
          </div>
        </div>
        """, unsafe_allow_html=True)
        st.write("")


# ----------------------------------------------------------------------------
# 11. CALCULATION BREAKDOWN PANEL
# ----------------------------------------------------------------------------
st.markdown('<div class="sec">Section Calculation Breakdown & Formula Inspector</div>', unsafe_allow_html=True)

selected_channel = st.selectbox(
    "Select or click a section to reveal detailed calculations & formulas:",
    CHANNELS,
    index=0
)

ch_info = df_results[df_results["Channel"] == selected_channel].iloc[0]
ch_cpc = ch_info["CPC"]
ch_spend = ch_info["Spend"]
ch_clicks = ch_info["Clicks"]

if ch_cpc is not None:
    st.markdown(f"""
    <div class="calc-panel">
      <div class="calc-banner">
        <h4 style="margin:0; font-size:1.15rem;">
          {CHANNEL_METRICS[selected_channel]['icon']} Calculation Breakdown for: <span style="color:{COLORS[selected_channel]}; font-weight:800;">{selected_channel}</span>
        </h4>
        <span style="background:{COLORS[selected_channel]}15; color:{COLORS[selected_channel]}; font-weight:700; padding:4px 12px; border-radius:999px; font-size:.82rem;">
          {ch_info['Share (%)']}% of Total Budget
        </span>
      </div>
      <p style="color:#64748B; font-size:.88rem; margin-bottom:14px;">Here is how MarketOpt computes the clicks and predicted customer increase across all three scenario categories:</p>
      
      <div class="calc-step">
        <b>Step 1: Budget Allocation</b><br>
        Allocated Spend = Total Budget ({inr(budget)}) × {ch_info['Share (%)']}% = <b>{inr(ch_spend)}</b>
      </div>

      <div class="calc-step">
        <b>Step 2: Paid Clicks Generated via Average CPC</b><br>
        Estimated Clicks = Allocated Spend ({inr(ch_spend)}) ÷ Avg CPC (₹{ch_cpc:.2f}) = <b>{ch_clicks:,} Clicks</b>
      </div>

      <div class="calc-step">
        <b>Step 3: Conversion Rate Base for {category} ({tier})</b><br>
        Base Conversion Rate = {(effective_cvr * 100):.2f}%
      </div>

      <div class="calc-step">
        <b>Step 4: Predicted Increase in Customers Across Three Categories</b><br>
        • <b style="color:#D97706;">Cautious (+0.65x):</b> {ch_clicks:,} clicks × {(effective_cvr * 100):.2f}% × 0.65 = <b style="color:#D97706;">+{ch_info['Cautious Customers']} Customers</b><br>
        • <b style="color:#2563EB;">Expected (1.00x):</b> {ch_clicks:,} clicks × {(effective_cvr * 100):.2f}% × 1.00 = <b style="color:#2563EB;">+{ch_info['Expected Customers']} Customers</b><br>
        • <b style="color:#059669;">Optimistic (+1.45x):</b> {ch_clicks:,} clicks × {(effective_cvr * 100):.2f}% × 1.45 = <b style="color:#059669;">+{ch_info['Optimistic Customers']} Customers</b>
      </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class="calc-panel">
      <div class="calc-banner">
        <h4 style="margin:0; font-size:1.15rem;">
          {CHANNEL_METRICS[selected_channel]['icon']} Offline Allocation for: <span style="color:{COLORS[selected_channel]}; font-weight:800;">{selected_channel}</span>
        </h4>
        <span style="background:{COLORS[selected_channel]}15; color:{COLORS[selected_channel]}; font-weight:700; padding:4px 12px; border-radius:999px; font-size:.82rem;">
          {ch_info['Share (%)']}% of Total Budget
        </span>
      </div>
      <p style="color:#64748B; font-size:.88rem; margin-bottom:14px;">Newspaper is an offline print medium used for institutional brand awareness and local credibility:</p>
      
      <div class="calc-step">
        <b>Step 1: Budget Allocation</b><br>
        Allocated Spend = Total Budget ({inr(budget)}) × {ch_info['Share (%)']}% = <b>{inr(ch_spend)}</b>
      </div>

      <div class="calc-step">
        <b>Step 2: Medium Characteristics</b><br>
        Newspaper advertising focuses on offline circulation, weekend inserts, and high-trust local community exposure rather than digital clicks or CPC tracking.
      </div>
    </div>
    """, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# 12. STREAMLINED BREAKDOWN TABLE & CSV DOWNLOAD
# ----------------------------------------------------------------------------
st.markdown('<div class="sec">Complete Section Breakdown Table</div>', unsafe_allow_html=True)

tbl = df_results.copy()
tbl["Spend"] = tbl["Spend"].map(inr)
tbl["CPC"] = tbl["CPC"].map(lambda x: f"₹{x:.2f}" if pd.notnull(x) else "-")
tbl["Clicks"] = tbl["Clicks"].map(lambda x: count(x) if pd.notnull(x) else "-")
tbl["Cautious (+0.65x)"] = tbl["Cautious Customers"].map(lambda x: count(x) if pd.notnull(x) else "-")
tbl["Expected (1.00x)"] = tbl["Expected Customers"].map(lambda x: count(x) if pd.notnull(x) else "-")
tbl["Optimistic (+1.45x)"] = tbl["Optimistic Customers"].map(lambda x: count(x) if pd.notnull(x) else "-")

display_cols = [
    "Channel", "Share (%)", "Spend", "CPC", "Clicks",
    "Cautious (+0.65x)", "Expected (1.00x)", "Optimistic (+1.45x)"
]

st.dataframe(tbl[display_cols], hide_index=True, width="stretch")

st.download_button(
    "📥 Download Complete Optimization Plan (CSV)",
    tbl[display_cols].to_csv(index=False).encode(),
    file_name="MarketOpt_Allocation_Plan.csv",
    mime="text/csv"
)
