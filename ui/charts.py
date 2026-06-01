import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

def render_score_chart(history: pd.DataFrame):
    if history.empty:
        return
        
    st.subheader("Score en el tiempo")
    score_data = history.copy().dropna(subset=["timestamp"])
    if score_data.empty:
        return
        
    # Plotly Line/Area chart for score
    fig = px.area(
        score_data.tail(120), 
        x="timestamp", 
        y="productivity_score",
        template="plotly_dark",
        color_discrete_sequence=["#4F46E5"]
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title=None,
        yaxis_title="Score",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(range=[0, 100])
    )
    st.plotly_chart(fig, width=" stretch\)

def render_app_usage_chart(history: pd.DataFrame, refresh_seconds: float):
    if history.empty:
        return
        
    st.subheader("Tiempo estimado por aplicación")
    history = history.copy().dropna(subset=["timestamp"])
    
    time_by_app = (
        history.tail(120)
        .groupby("active_app")
        .size()
        .sort_values(ascending=False)
        .head(8)
        .mul(refresh_seconds)
        .reset_index(name="segundos")
    )
    
    if time_by_app.empty:
        return

    # Plotly Donut chart for app usage
    fig = px.pie(
        time_by_app, 
        values="segundos", 
        names="active_app", 
        hole=0.6,
        template="plotly_dark",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    fig.update_traces(textinfo='percent+label', textposition='inside')
    
    st.plotly_chart(fig, width=" stretch\)
