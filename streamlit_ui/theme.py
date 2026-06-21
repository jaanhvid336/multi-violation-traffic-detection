import streamlit as st

# --- Colour Palette: Slate & Emerald ---
NAVY = "#2D3B48"      # Deep Slate Sidebar
NAVY_DEEP = "#1B242D" # Darker Slate
BLUE = "#059669"      # Emerald Accent
INK = "#2D3B48"       # Dark Text

# Per-page accent colours
DASHBOARD = "#059669"   # emerald
LIVE = "#024462"        # bright emerald
IMAGE = "#94065E"       # deep emerald
ANALYTICS = "#065F46"   # dark forest
LOG = "#D97706"         # amber (caution)
HEATMAP = "#DC2626"     # red (alert)

CSS = f"""
<style>
:root {{ --navy:{NAVY}; --blue:{BLUE}; --ink:{INK}; }}

.stApp {{ background:#F8FAFC; }}

/* Sidebar - Slate Theme */
section[data-testid="stSidebar"] {{
    background:linear-gradient(180deg,{NAVY},{NAVY_DEEP});
}}
section[data-testid="stSidebar"] * {{ color:#E2E8F0; }}
section[data-testid="stSidebar"] a {{
    border-radius:8px;
    transition:background .2s ease, transform .2s ease;
}}
section[data-testid="stSidebar"] a:hover {{
    background:rgba(255,255,255,.08);
    transform:translateX(3px);
}}

/* Headings */
h1,h2,h3 {{ color:{NAVY}; font-weight:700; letter-spacing:.2px; }}

/* Page hero */
.tv-hero {{
    padding:20px 26px; border-radius:14px; color:#fff;
    margin-bottom:22px; box-shadow:0 8px 24px rgba(45,59,72,.15);
    animation:tvFade .5s ease both;
}}
.tv-brand {{
    display:inline-block; font-weight:800; letter-spacing:1.5px;
    font-size:.82rem; text-transform:uppercase;
    background:rgba(255,255,255,.15); padding:4px 12px; border-radius:20px;
    margin-bottom:10px;
}}
.tv-hero h1 {{ color:#fff; margin:0; font-size:1.7rem; font-weight:800; }}
.tv-hero p {{ margin:.35rem 0 0; opacity:.9; font-size:.95rem; }}

/* Metric cards */
.tv-cards {{ display:flex; gap:16px; flex-wrap:wrap; margin:4px 0 8px; }}
.tv-card {{
    flex:1; min-width:160px; background:#fff; border-radius:14px;
    padding:18px 20px; border:1px solid #E2E8F0;
    box-shadow:0 4px 14px rgba(45,59,72,.05);
    animation:tvRise .5s ease both;
    transition:transform .2s ease, box-shadow .2s ease;
}}
.tv-card:hover {{ transform:translateY(-3px); box-shadow:0 10px 22px rgba(45,59,72,.1); }}
.tv-card .label {{ color:#64748B; font-size:.8rem; text-transform:uppercase; letter-spacing:.6px; }}
.tv-card .value {{ color:{NAVY}; font-size:2rem; font-weight:800; line-height:1.1; margin-top:6px; }}
.tv-card .accent {{ height:4px; width:42px; border-radius:4px; margin-top:12px; }}

/* Buttons */
.stButton > button {{
    border-radius:10px; font-weight:600; border:none;
    transition:transform .15s ease, box-shadow .15s ease;
}}
.stButton > button:hover {{ transform:translateY(-1px); box-shadow:0 6px 16px rgba(5,150,105,.2); }}

/* Dataframe container */
[data-testid="stDataFrame"] {{ border-radius:12px; overflow:hidden; border:1px solid #E2E8F0; }}

@keyframes tvFade {{ from{{opacity:0}} to{{opacity:1}} }}
@keyframes tvRise {{ from{{opacity:0; transform:translateY(8px)}} to{{opacity:1; transform:translateY(0)}} }}
</style>
"""

def inject():
    st.markdown(CSS, unsafe_allow_html=True)

def hero(title: str, subtitle: str = "", accent: str = BLUE):
    sub = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(
        f'<div class="tv-hero" style="background:linear-gradient(100deg,{NAVY_DEEP},{accent});">'
        f'<span class="tv-brand">TrafficVision AI</span>'
        f'<h1>{title}</h1>{sub}</div>',
        unsafe_allow_html=True,
    )

def metric_cards(cards):
    html = ['<div class="tv-cards">']
    for label, value, color in cards:
        html.append(
            f'<div class="tv-card"><div class="label">{label}</div>'
            f'<div class="value">{value}</div>'
            f'<div class="accent" style="background:{color}"></div></div>'
        )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)