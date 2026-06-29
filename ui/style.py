import streamlit as st

def inject_custom_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        :root {
            --primary: #4F46E5;
            --background: #111827;
            --card-bg: #1F2937;
            --text-main: #F9FAFB;
            --text-muted: #9CA3AF;
        }
        
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif !important;
        }

        /* Contenedores (Cards) */
        div[data-testid="stMetric"], div.stDataFrame, div[data-testid="stExpander"], .custom-card {
            background-color: var(--card-bg) !important;
            padding: 1rem !important;
            border-radius: 0.75rem !important;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
            border: 1px solid rgba(255, 255, 255, 0.05) !important;
        }

        /* Tabs styling */
        div.stTabs [data-baseweb="tab"] {
            padding: 0.5rem 1rem !important;
            border-radius: 0.5rem !important;
            margin-right: 0.5rem !important;
            background: transparent !important;
            color: var(--text-muted) !important;
            transition: all 0.3s ease !important;
        }
        div.stTabs [aria-selected="true"] {
            background-color: var(--primary) !important;
            color: white !important;
            box-shadow: 0 4px 14px 0 rgba(79, 70, 229, 0.39) !important;
        }

        /* Botones primarios */
        button[kind="primary"] {
            background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%) !important;
            border: none !important;
            color: white !important;
            box-shadow: 0 4px 14px 0 rgba(79, 70, 229, 0.39) !important;
            border-radius: 0.5rem !important;
            font-weight: 600 !important;
            transition: transform 0.2s !important;
        }
        button[kind="primary"]:hover {
            transform: translateY(-2px) !important;
        }

        /* Indicador en vivo (Pulsating Red Dot) */
        .live-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            background-color: #ef4444;
            border-radius: 50%;
            margin-right: 8px;
            animation: pulse-red 2s infinite;
        }

        @keyframes pulse-red {
            0% {
                transform: scale(0.95);
                box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7);
            }
            70% {
                transform: scale(1);
                box-shadow: 0 0 0 6px rgba(239, 68, 68, 0);
            }
            100% {
                transform: scale(0.95);
                box-shadow: 0 0 0 0 rgba(239, 68, 68, 0);
            }
        }
        
        .status-container {
            display: flex;
            align-items: center;
            font-weight: 600;
            color: var(--text-main);
            padding: 0.5rem 1rem;
            background: rgba(239, 68, 68, 0.1);
            border-radius: 20px;
            width: fit-content;
            margin-bottom: 10px;
            border: 1px solid rgba(239, 68, 68, 0.2);
        }
        </style>
    """, unsafe_allow_html=True)
