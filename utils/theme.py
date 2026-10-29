# utils/theme.py
import streamlit as st

# --- Brand (your palette) ---
INDIGO = "#28536B"
MATCHA = "#8FB78F"
PISTACHIO = "#CBE6C7"
AZURE = "#B4CCCF"

# --- Surfaces and ink ---
BG = "#F7FAF5"
CARD = "#FFFFFF"
INK = "#28536B"
MUTED = "#6B8A94"
GRID = "#E1E6E0"

# --- Chart series: fixed order (Indigo, Matcha, Plum, Gold, Azure, Coral, Violet, Crimson) ---
SERIES = ["#2166A3", "#359C5C", "#9C4B74", "#D9A441",
          "#17A3B8", "#D9643C", "#6A4FB0", "#B33B3B"]

# --- Sequential ramp (light -> dark, Indigo family) ---
SEQUENTIAL = ["#E4EEF7", "#BFDAEF", "#8FB9DE", "#4F93C7",
              "#2166A3", "#28536B", "#163A4C"]

# --- Diverging poles + neutral midpoint ---
DIVERGING = {"low": "#2166A3", "mid": "#EDEBE6", "high": "#D9643C"}

# --- Risk status: reserved, always shown with a text label ---
RISK = {
    "Low":      {"color": "#0CA30C", "mark": "●"},
    "Monitor":  {"color": "#FAB219", "mark": "▲"},
    "Elevated": {"color": "#EC835A", "mark": "◆"},
    "High":     {"color": "#D03B3B", "mark": "■"},
}


def risk_badge(level: str) -> str:
    """HTML badge: coloured mark + text label (never colour alone)."""
    r = RISK[level]
    return (
        f'<span style="display:inline-flex;align-items:center;gap:6px;'
        f'padding:2px 10px;border:1px solid {r["color"]};border-radius:999px;'
        f'font-weight:600;font-size:0.85rem;color:{INK};">'
        f'<span style="color:{r["color"]}">{r["mark"]}</span>{level}</span>'
    )


def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"],
        [data-testid="stHeader"] {{ background-color: {BG} !important; }}

        .stApp h1, .stApp h2, .stApp h3, .stApp h4 {{
            color: {INK} !important; font-weight: 700; letter-spacing: -0.01em;
        }}
        .stApp h1 {{ border-bottom: 3px solid {MATCHA}; padding-bottom: 0.4rem; }}

        .stApp [data-testid="stMarkdownContainer"] p,
        .stApp [data-testid="stMarkdownContainer"] li,
        .stApp [data-testid="stCaptionContainer"],
        .stApp label {{ color: #1F3A4A !important; }}

        [data-testid="stSidebar"] {{
            background-color: #E6F1E3 !important;
            border-right: 1px solid {AZURE};
        }}
        [data-testid="stSidebarNav"] a span {{ color: {INK} !important; }}
        [data-testid="stSidebarNav"] a[aria-current="page"] {{ background-color: {PISTACHIO} !important; }}

        [data-testid="stMetric"] {{
            background-color: {CARD} !important;
            border: 1px solid {AZURE};
            border-left: 4px solid {INDIGO};
            border-radius: 6px;
            padding: 0.8rem 1rem;
        }}
        [data-testid="stMetricLabel"] p {{ color: {MUTED} !important; }}
        [data-testid="stMetricValue"] {{ color: {INK} !important; }}

        [data-testid="stDataFrame"] {{ border: 1px solid {AZURE}; border-radius: 6px; }}
        .stDeployButton, [data-testid="stAppDeployButton"] {{ display: none; }}
        .block-container {{ padding-top: 2rem; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def style_fig(fig, height: int = 380):
    """Apply the shared chart look to a Plotly figure."""
    fig.update_layout(
        height=height,
        colorway=SERIES,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="system-ui, -apple-system, 'Segoe UI', sans-serif", color=INK, size=13),
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        title=dict(font=dict(size=15, color=INK)),
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, tickfont=dict(color=MUTED))
    return fig