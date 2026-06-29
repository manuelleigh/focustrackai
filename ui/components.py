import json
import urllib.request
import streamlit as st
from streamlit_lottie import st_lottie

@st.cache_data
def load_lottieurl(url: str):
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=3) as response:
            return json.loads(response.read())
    except Exception:
        return None

def render_live_indicator(is_running: bool):
    if is_running:
        st.markdown(
            '<div class="status-container"><div class="live-indicator"></div> Monitoreo Activo</div>', 
            unsafe_allow_html=True
        )

def render_empty_state():
    # URL de animacion de Lottie (Scanning/AI)
    lottie_url = "https://lottie.host/79051fb2-fc19-4cb3-9ff4-a957b44747eb/B1G35V3G53.json"
    
    lottie_json = load_lottieurl(lottie_url)
    if lottie_json:
        st_lottie(lottie_json, height=200, key="empty_state")
    else:
        st.info("El frame analizado aparecerá aquí al iniciar el monitoreo.")

def render_gauge_score(score: float):
    color = "#10B981" if score >= 75 else "#F59E0B" if score >= 45 else "#EF4444"
    st.markdown(f"""
        <div style="text-align: center; padding: 1rem;">
            <div style="font-size: 3.5rem; font-weight: 700; color: {color}; line-height: 1;">
                {score:.1f}
            </div>
            <div style="color: var(--text-muted); font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px; margin-top: 5px;">
                Score Principal
            </div>
        </div>
    """, unsafe_allow_html=True)
