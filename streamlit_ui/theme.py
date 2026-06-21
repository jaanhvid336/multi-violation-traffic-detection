"""Global theme: CSS injection + small reusable UI pieces (no emojis)."""

import streamlit as st

NAVY = "#0B2D5C"
NAVY_DEEP = "#071C3B"
BLUE = "#2563EB"
INK = "#1E293B"

# Per-page accent colours (the "colour scheme" for each section).
DASHBOARD = "#2563EB"   # blue
LIVE = "#0EA5E9"        # cyan
IMAGE = "#0D9488"       # teal
ANALYTICS = "#7C3AED"   # purple
LOG = "#F97316"         # orange
HEATMAP = "#EF4444"     # red

CSS = """
<style>
:root { --navy:#0B2D5C; --blue:#2563EB; --ink:#1E293B; }

.stApp { background:#F4F7FB; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background:linear-gradient(180deg,#0B2D5C,#071C3B);
}
section[data-testid="stSidebar"] * { color:#E5ECF6; }
section[data-testid="stSidebar"] a {
    border-radius:8px;
    transition:background .2s ease, transform .2s ease;
}
section[data-testid="stSidebar"] a:hover {
    background:rgba(255,255,255,.08);
    transform:translateX(3px);
}

/* Headings */
h1,h2,h3 { color:#163B7A; font-weight:700; letter-spacing:.2px; }

/* Page hero */
.tv-hero {
    padding:20px 26px; border-radius:14px; color:#fff;
    margin-bottom:22px; box-shadow:0 8px 24px rgba(11,45,92,.18);
    animation:tvFade .5s ease both;
}
.tv-brand {
    display:inline-block; font-weight:800; letter-spacing:1.5px;
    font-size:.82rem; text-transform:uppercase;
    background:rgba(255,255,255,.18); padding:4px 12px; border-radius:20px;
    margin-bottom:10px;
}
.tv-hero h1 { color:#fff; margin:0; font-size:1.7rem; font-weight:800; }
.tv-hero p  { margin:.35rem 0 0; opacity:.92; font-size:.95rem; }

/* Brand wordmark in the sidebar */
.tv-sidebrand { font-weight:800; font-size:1.25rem; letter-spacing:.4px;
    background:linear-gradient(90deg,#7DD3FC,#FFFFFF);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; }

/* Metric cards */
.tv-cards { display:flex; gap:16px; flex-wrap:wrap; margin:4px 0 8px; }
.tv-card {
    flex:1; min-width:160px; background:#fff; border-radius:14px;
    padding:18px 20px; border:1px solid #E6EDF6;
    box-shadow:0 4px 14px rgba(16,42,77,.06);
    animation:tvRise .5s ease both;
    transition:transform .2s ease, box-shadow .2s ease;
}
.tv-card:hover { transform:translateY(-3px); box-shadow:0 10px 22px rgba(16,42,77,.12); }
.tv-card .label { color:#64748B; font-size:.8rem; text-transform:uppercase; letter-spacing:.6px; }
.tv-card .value { color:#0B2D5C; font-size:2rem; font-weight:800; line-height:1.1; margin-top:6px; }
.tv-card .accent { height:4px; width:42px; border-radius:4px; margin-top:12px; }

/* Buttons */
.stButton > button {
    border-radius:10px; font-weight:600; border:none;
    transition:transform .15s ease, box-shadow .15s ease;
}
.stButton > button:hover { transform:translateY(-1px); box-shadow:0 6px 16px rgba(37,99,235,.25); }

/* Dataframe container */
[data-testid="stDataFrame"] { border-radius:12px; overflow:hidden; border:1px solid #E6EDF6; }

@keyframes tvFade { from{opacity:0} to{opacity:1} }
@keyframes tvRise { from{opacity:0; transform:translateY(8px)} to{opacity:1; transform:translateY(0)} }
</style>
"""


def inject():
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str = "", accent: str = BLUE):
    """Page header: bold TrafficVision AI brand + page title, tinted by the
    page's accent colour."""
    sub = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(
        f'<div class="tv-hero" style="background:linear-gradient(100deg,{NAVY_DEEP},{accent});">'
        f'<span class="tv-brand">TrafficVision AI</span>'
        f'<h1>{title}</h1>{sub}</div>',
        unsafe_allow_html=True,
    )


def metric_cards(cards):
    """cards: list of (label, value, accent_color)."""
    html = ['<div class="tv-cards">']
    for label, value, color in cards:
        html.append(
            f'<div class="tv-card"><div class="label">{label}</div>'
            f'<div class="value">{value}</div>'
            f'<div class="accent" style="background:{color}"></div></div>'
        )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)
