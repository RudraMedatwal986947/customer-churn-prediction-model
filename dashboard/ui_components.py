"""
Shared UI Components & Styling Module
Provides dynamic Light/Dark theme toggling, modern CSS injection, KPI cards,
and Plotly theme harmonization. Strictly adheres to project styling standards
without emojis and without em dashes.
"""

import streamlit as st
import plotly.graph_objects as go

def get_current_theme() -> str:
    """Returns the current active theme, defaulting to light mode."""
    if "theme_mode" not in st.session_state:
        st.session_state["theme_mode"] = "light"
    return st.session_state["theme_mode"]


def _on_theme_radio_change():
    """Callback triggered immediately when user clicks the theme toggle."""
    selected = st.session_state.get("theme_mode_radio_selector", "Light")
    st.session_state["theme_mode"] = selected.lower()


def render_theme_toggle():
    """Renders a clean theme selector in the sidebar without emojis or em dashes."""
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Interface Theme**")
    current_theme = get_current_theme()

    # Sync radio selector state with session theme
    target_val = "Dark" if current_theme == "dark" else "Light"
    if "theme_mode_radio_selector" not in st.session_state:
        st.session_state["theme_mode_radio_selector"] = target_val
    elif st.session_state["theme_mode_radio_selector"] != target_val:
        st.session_state["theme_mode_radio_selector"] = target_val

    st.sidebar.radio(
        "Select Theme",
        options=["Light", "Dark"],
        horizontal=True,
        label_visibility="collapsed",
        key="theme_mode_radio_selector",
        on_change=_on_theme_radio_change,
    )


def apply_custom_css():
    """Injects dynamic, theme-responsive design styling into the Streamlit app."""
    theme = get_current_theme()
    is_dark = (theme == "dark")

    if is_dark:
        # Dark Theme Palette (Deep Slate & Surface Slate)
        bg_root = "#0F172A"
        bg_card = "#1E293B"
        bg_sidebar = "#0B0F19"
        text_primary = "#F8FAFC"
        text_secondary = "#CBD5E1"
        text_muted = "#94A3B8"
        border_color = "#334155"
        badge_bg = "#1E3A8A"
        badge_text = "#93C5FD"
        badge_border = "#3B82F6"
        tab_active_bg = "#1E293B"
        tab_active_color = "#60A5FA"
        info_box_bg = "#1E293B"
        info_box_border = "#3B82F6"
        info_box_text = "#93C5FD"
    else:
        # Light Theme Palette (Clean White & Slate 50)
        bg_root = "#F8FAFC"
        bg_card = "#FFFFFF"
        bg_sidebar = "#FFFFFF"
        text_primary = "#0F172A"
        text_secondary = "#475569"
        text_muted = "#64748B"
        border_color = "#E2E8F0"
        badge_bg = "#EFF6FF"
        badge_text = "#1D4ED8"
        badge_border = "#BFDBFE"
        tab_active_bg = "#FFFFFF"
        tab_active_color = "#2563EB"
        info_box_bg = "#EFF6FF"
        info_box_border = "#BFDBFE"
        info_box_text = "#1E40AF"

    css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }}

    /* Root Canvas */
    .stApp {{
        background-color: {bg_root} !important;
        color: {text_primary} !important;
    }}

    /* Sidebar */
    [data-testid="stSidebar"] {{
        background-color: {bg_sidebar} !important;
        border-right: 1px solid {border_color} !important;
    }}
    [data-testid="stSidebar"] * {{
        color: {text_secondary} !important;
    }}
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {{
        color: {text_primary} !important;
    }}

    /* Content block padding */
    .block-container {{
        padding-top: 2rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 1280px;
    }}

    /* Typography Overrides */
    .stApp h1, .stApp h2, .stApp h3, .stApp h4 {{
        color: {text_primary} !important;
    }}
    .stApp p, .stApp li {{
        color: {text_secondary} !important;
    }}

    /* Header Container */
    .page-title-container {{
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid {border_color};
    }}

    .page-category-badge {{
        display: inline-block;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: {badge_text};
        background-color: {badge_bg};
        border: 1px solid {badge_border};
        padding: 0.2rem 0.65rem;
        border-radius: 9999px;
        margin-bottom: 0.5rem;
    }}

    .page-main-title {{
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        color: {text_primary} !important;
        margin: 0 0 0.4rem 0;
        line-height: 1.2;
    }}

    .page-subtitle {{
        font-size: 0.975rem;
        color: {text_muted} !important;
        margin: 0;
        font-weight: 400;
        line-height: 1.5;
    }}

    /* KPI Metric Cards */
    .kpi-card-wrapper {{
        background: {bg_card};
        border: 1px solid {border_color};
        border-radius: 12px;
        padding: 1.25rem 1rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.08);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        position: relative;
        overflow: hidden;
        margin-bottom: 0.5rem;
    }}

    .kpi-card-wrapper:hover {{
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.15);
    }}

    .kpi-card-accent {{
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
    }}

    .kpi-label {{
        font-size: 0.8rem;
        font-weight: 600;
        color: {text_muted};
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.4rem;
    }}

    .kpi-value {{
        font-size: 1.85rem;
        font-weight: 800;
        color: {text_primary};
        line-height: 1.1;
        letter-spacing: -0.03em;
        margin-bottom: 0.35rem;
    }}

    .kpi-caption {{
        font-size: 0.8rem;
        color: {text_muted};
        font-weight: 500;
    }}

    /* Action / Feature Box */
    .feature-card {{
        background: {bg_card};
        border: 1px solid {border_color};
        border-radius: 14px;
        padding: 1.5rem;
        height: 100%;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }}

    .feature-card-header {{
        font-size: 1.15rem;
        font-weight: 700;
        color: {text_primary};
        margin-bottom: 0.5rem;
    }}

    .feature-card-body {{
        font-size: 0.875rem;
        color: {text_secondary};
        line-height: 1.55;
        margin-bottom: 1rem;
    }}

    .feature-card-footer {{
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }}

    /* Badges */
    .badge {{
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }}
    .badge-blue   {{ background: {"#1E3A8A" if is_dark else "#EFF6FF"}; color: {"#93C5FD" if is_dark else "#1D4ED8"}; border: 1px solid {"#3B82F6" if is_dark else "#BFDBFE"}; }}
    .badge-green  {{ background: {"#064E3B" if is_dark else "#ECFDF5"}; color: {"#A7F3D0" if is_dark else "#047857"}; border: 1px solid {"#059669" if is_dark else "#A7F3D0"}; }}
    .badge-yellow {{ background: {"#78350F" if is_dark else "#FFFBEB"}; color: {"#FDE68A" if is_dark else "#B45309"}; border: 1px solid {"#D97706" if is_dark else "#FDE68A"}; }}
    .badge-red    {{ background: {"#7F1D1D" if is_dark else "#FEF2F2"}; color: {"#FECACA" if is_dark else "#B91C1C"}; border: 1px solid {"#DC2626" if is_dark else "#FECACA"}; }}
    .badge-slate  {{ background: {"#334155" if is_dark else "#F8FAFC"}; color: {"#E2E8F0" if is_dark else "#334155"}; border: 1px solid {"#475569" if is_dark else "#E2E8F0"}; }}

    /* Profile Box */
    .profile-card {{
        background: {bg_card};
        border: 1px solid {border_color};
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
    }}

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        border-bottom: 1px solid {border_color};
        padding-bottom: 4px;
    }}

    .stTabs [data-baseweb="tab"] {{
        height: 42px;
        border-radius: 8px 8px 0 0;
        font-size: 0.9rem;
        font-weight: 600;
        padding: 0 16px;
        color: {text_muted} !important;
        border: none;
    }}

    .stTabs [aria-selected="true"] {{
        background-color: {tab_active_bg} !important;
        color: {tab_active_color} !important;
        border-bottom: 3px solid {tab_active_color} !important;
    }}

    /* Expanders */
    [data-testid="stExpander"] {{
        background-color: {bg_card} !important;
        border: 1px solid {border_color} !important;
        border-radius: 8px !important;
    }}
    [data-testid="stExpander"] details summary {{
        color: {text_primary} !important;
    }}

    /* Buttons */
    .stButton > button {{
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.9rem;
        padding: 0.5rem 1.25rem;
        transition: all 0.15s ease;
    }}

    /* Horizontal Rules */
    hr {{
        margin: 1.5rem 0 !important;
        border: 0 !important;
        border-top: 1px solid {border_color} !important;
    }}
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
    """Harmonizes a Plotly figure with the active Light or Dark theme."""
    theme = get_current_theme()
    is_dark = (theme == "dark")

    if is_dark:
        template_name = "plotly_dark"
        font_color = "#CBD5E1"
        title_color = "#F8FAFC"
        grid_color = "rgba(148, 163, 184, 0.15)"
        line_color = "#475569"
        hover_bg = "#1E293B"
        hover_text = "#F8FAFC"
    else:
        template_name = "plotly_white"
        font_color = "#334155"
        title_color = "#0F172A"
        grid_color = "rgba(226, 232, 240, 0.8)"
        line_color = "#CBD5E1"
        hover_bg = "#0F172A"
        hover_text = "#FFFFFF"

    fig.update_layout(
        template=template_name,
        font=dict(
            family="'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
            size=12,
            color=font_color
        ),
        title_font=dict(size=14, color=title_color, family="'Inter', sans-serif"),
        height=height,
        margin=dict(l=20, r=20, t=50, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(
            bgcolor=hover_bg,
            font_size=12,
            font_family="'Inter', sans-serif",
            font_color=hover_text
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
        gridcolor=grid_color,
        linecolor=line_color,
        tickfont=dict(size=11, color=font_color)
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=grid_color,
        linecolor=line_color,
        tickfont=dict(size=11, color=font_color)
    )
    return fig
