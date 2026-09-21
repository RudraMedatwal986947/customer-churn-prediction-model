"""
Shared UI Components & Styling Module
Provides modern CSS injection, KPI cards, badges, and Plotly theme harmonization.
Strictly adheres to project styling standards without emojis.
"""

import streamlit as st
import plotly.graph_objects as go

def apply_custom_css():
    """Injects high-end, modern design styling into the Streamlit app."""
    css = """
    <style>
    /* Google Fonts Import */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    /* Top padding reduction for sleek look */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 1280px;
    }

    /* Sleek Header & Titles */
    .page-title-container {
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid rgba(148, 163, 184, 0.2);
    }

    .page-category-badge {
        display: inline-block;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #2563EB;
        background-color: #EFF6FF;
        border: 1px solid #BFDBFE;
        padding: 0.2rem 0.65rem;
        border-radius: 9999px;
        margin-bottom: 0.5rem;
    }

    .page-main-title {
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        color: #0F172A;
        margin: 0 0 0.4rem 0;
        line-height: 1.2;
    }

    .page-subtitle {
        font-size: 0.975rem;
        color: #64748B;
        margin: 0;
        font-weight: 400;
        line-height: 1.5;
    }

    /* KPI Metric Cards */
    .kpi-card-wrapper {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.25rem 1rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05), 0 1px 2px 0 rgba(0, 0, 0, 0.03);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        position: relative;
        overflow: hidden;
        margin-bottom: 0.5rem;
    }

    .kpi-card-wrapper:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08), 0 4px 6px -2px rgba(0, 0, 0, 0.04);
    }

    .kpi-card-accent {
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
    }

    .kpi-label {
        font-size: 0.8rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.4rem;
    }

    .kpi-value {
        font-size: 1.85rem;
        font-weight: 800;
        color: #0F172A;
        line-height: 1.1;
        letter-spacing: -0.03em;
        margin-bottom: 0.35rem;
    }

    .kpi-caption {
        font-size: 0.8rem;
        color: #94A3B8;
        font-weight: 500;
    }

    /* Action Card / Feature Box */
    .feature-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 1.5rem;
        height: 100%;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }

    .feature-card-header {
        font-size: 1.15rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.5rem;
    }

    .feature-card-body {
        font-size: 0.875rem;
        color: #475569;
        line-height: 1.55;
        margin-bottom: 1rem;
    }

    .feature-card-footer {
        font-size: 0.8rem;
        font-weight: 600;
        color: #2563EB;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    /* Badges */
    .badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    .badge-blue   { background: #EFF6FF; color: #1D4ED8; border: 1px solid #BFDBFE; }
    .badge-green  { background: #ECFDF5; color: #047857; border: 1px solid #A7F3D0; }
    .badge-yellow { background: #FFFBEB; color: #B45309; border: 1px solid #FDE68A; }
    .badge-red    { background: #FEF2F2; color: #B91C1C; border: 1px solid #FECACA; }
    .badge-slate  { background: #F8FAFC; color: #334155; border: 1px solid #E2E8F0; }

    /* Customer Profile Summary Card */
    .profile-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #E2E8F0;
        padding-bottom: 4px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 42px;
        border-radius: 8px 8px 0 0;
        font-size: 0.9rem;
        font-weight: 600;
        padding: 0 16px;
        color: #64748B;
        border: none;
    }

    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF !important;
        color: #2563EB !important;
        border-bottom: 3px solid #2563EB !important;
    }

    /* Button Polish */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.9rem;
        padding: 0.5rem 1.25rem;
        transition: all 0.15s ease;
    }

    /* Subtle divider */
    hr {
        margin: 1.5rem 0 !important;
        border: 0 !important;
        border-top: 1px solid #E2E8F0 !important;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def render_page_header(title: str, subtitle: str, category: str = "Enterprise Analytics"):
    """Renders a clean, polished page header with standard design tokens."""
    html = f"""
    <div class="page-title-container">
        <div class="page-category-badge">{category}</div>
        <h1 class="page-main-title">{title}</h1>
        <p class="page-subtitle">{subtitle}</p>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_kpi(label: str, value: str, caption: str = "", accent_color: str = "#2563EB"):
    """Renders an individual KPI card with an accent colored top border."""
    html = f"""
    <div class="kpi-card-wrapper">
        <div class="kpi-card-accent" style="background-color: {accent_color};"></div>
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-caption">{caption}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_feature_card(title: str, description: str, tag: str, tag_color: str = "#2563EB"):
    """Renders an executive overview feature card for the main portal."""
    html = f"""
    <div class="feature-card">
        <div>
            <div class="feature-card-header">{title}</div>
            <div class="feature-card-body">{description}</div>
        </div>
        <div class="feature-card-footer" style="color: {tag_color};">{tag}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def apply_plotly_theme(fig: go.Figure, height: int = 400) -> go.Figure:
    """Harmonizes a Plotly figure with modern minimal design conventions."""
    fig.update_layout(
        template="plotly_white",
        font=dict(
            family="'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
            size=12,
            color="#334155"
        ),
        title_font=dict(size=14, color="#0F172A", family="'Inter', sans-serif"),
        height=height,
        margin=dict(l=20, r=20, t=50, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(
            bgcolor="#0F172A",
            font_size=12,
            font_family="'Inter', sans-serif",
            font_color="#FFFFFF"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            title_text=""
        )
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(226, 232, 240, 0.8)",
        linecolor="#CBD5E1",
        tickfont=dict(size=11, color="#64748B")
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(226, 232, 240, 0.8)",
        linecolor="#CBD5E1",
        tickfont=dict(size=11, color="#64748B")
    )
    return fig
