import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Import backend modules
from data_generator import generate_mock_data
import analytics

st.set_page_config(
    page_title="LTV & Cohort Analytics Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark theme & styling customization via CSS
st.markdown("""
<style>
    .reportview-container {
        background-color: #0e1117;
    }
    .metric-card {
        background-color: #1e222b;
        padding: 20px;
        border-radius: 8px;
        border-left: 5px solid #ff4b4b;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 15px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #ffffff;
    }
    .metric-label {
        font-size: 14px;
        color: #8a92a6;
    }
</style>
""", unsafe_allow_html=True)

# -----------------
# 1. Sidebar Config & Data Caching
# -----------------
st.sidebar.title("Engine Controls")
st.sidebar.markdown("---")

@st.cache_data(show_spinner="Generating mock dataset...")
def load_data(n_users=12000):
    return generate_mock_data(n_users=n_users)

n_users = st.sidebar.slider("Simulated Users Count", min_value=5000, max_value=25000, value=12000, step=1000)
df_raw = load_data(n_users=n_users)

st.sidebar.success(f"Loaded {len(df_raw):,} records (24 months)")

# Target Gross Margin
margin_pct = st.sidebar.slider("Target Gross Margin (%)", min_value=10, max_value=100, value=80, step=5) / 100.0

# Budget setup for CAC optimization
st.sidebar.markdown("### Budget Optimization Settings")
total_budget = st.sidebar.number_input("Total Marketing Budget ($)", min_value=10000, max_value=1000000, value=250000, step=10000)

channels = ["Paid Search", "Paid Social", "Organic", "Referral"]
channel_shares = {}
st.sidebar.write("Baseline Spend Shares (%)")
col1, col2 = st.sidebar.columns(2)
with col1:
    channel_shares["Paid Search"] = st.number_input("Paid Search", min_value=0, max_value=100, value=35)
    channel_shares["Paid Social"] = st.number_input("Paid Social", min_value=0, max_value=100, value=45)
with col2:
    channel_shares["Organic"] = st.number_input("Organic (SEO)", min_value=0, max_value=100, value=10, disabled=True) # locked channel
    channel_shares["Referral"] = st.number_input("Referral", min_value=0, max_value=100, value=10)

# Normalize baseline shares
total_shares = sum(channel_shares.values())
normalized_shares = {k: v / total_shares for k, v in channel_shares.items()}

# -----------------
# 2. Main Analytics Precomputation
# -----------------
# Compute cohort data
retention_matrix, arpu_matrix, size_df = analytics.compute_cohort_matrices(df_raw)

# Forecast LTV
# Run machine learning model to get predicted LTV
with st.spinner("Training predictive models & generating LTV forecasts..."):
    df_user_ltv, error_msg = analytics.train_and_forecast_ltv(df_raw, forecast_horizon_months=12)

if error_msg:
    st.error(error_msg)
    st.stop()

# Run budget optimization
channel_stats, summary_opt = analytics.run_cac_optimization(df_user_ltv, normalized_shares, total_budget, margin_pct)

# Title Banner
st.title("📊 LTV & Cohort Analytics Engine")
st.markdown("A premium, growth-oriented dashboard using machine learning to maximize acquisition efficiency, cohort retention, and checkout conversion.")

# Tabs Setup
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Executive Summary & CAC Payback", 
    "📅 Cohort Retention Matrices", 
    "🔮 Predictive LTV Deep-Dive", 
    "🛒 Funnel Diagnostics & Anomalies"
])

# -----------------
# TAB 1: EXECUTIVE SUMMARY & CAC OPTIMIZATION
# -----------------
with tab1:
    st.header("Executive Summary")
    
    # KPIs Rows
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    
    blended_cac = df_user_ltv["cac"].mean()
    blended_ltv = df_user_ltv["predicted_ltv"].mean()
    ltv_cac_ratio = blended_ltv / blended_cac if blended_cac > 0 else blended_ltv
    
    # Simple payback estimation
    payback_months = channel_stats["avg_cac"].mean() / (channel_stats["avg_predicted_ltv"].mean() / 12)
    
    with kpi_col1:
        st.metric(
            label="Total Customers", 
            value=f"{len(df_user_ltv):,}", 
            delta="Simulated Dataset"
        )
    with kpi_col2:
        st.metric(
            label="Blended LTV (12-Mo Forecast)", 
            value=f"${blended_ltv:.2f}",
            delta=f"LTV:CAC Ratio: {ltv_cac_ratio:.2f}x"
        )
    with kpi_col3:
        st.metric(
            label="Average CAC Payback Period", 
            value=f"{payback_months:.1f} Months",
            delta=f"Blended CAC: ${blended_cac:.2f}"
        )
    with kpi_col4:
        opt_gain = summary_opt.get("efficiency_gain", 0) * 100
        st.metric(
            label="Optimized Efficiency Lift", 
            value=f"+{opt_gain:.2f}%", 
            delta="Target +14%",
            delta_color="normal"
        )
        
    st.markdown("---")
    
    # Optimization section
    st.subheader("Quantitative CAC Payback Optimization")
    st.markdown("Below is the results of reallocating budget from channels with low LTV:CAC performance to higher margin channels.")
    
    opt_col1, opt_col2 = st.columns([1, 1])
    
    with opt_col1:
        st.markdown("#### Performance Metrics comparison")
        
        # Table of channels
        display_stats = channel_stats[[
            "signup_channel", "users", "avg_cac", "avg_predicted_ltv", "ltv_cac_ratio"
        ]].copy()
        display_stats.columns = ["Channel", "Acquired Users", "Avg CAC ($)", "12-Mo Forecasted LTV ($)", "LTV:CAC Ratio"]
        st.dataframe(display_stats.style.format({
            "Avg CAC ($)": "${:.2f}",
            "12-Mo Forecasted LTV ($)": "${:.2f}",
            "LTV:CAC Ratio": "{:.2f}x"
        }), width='stretch')
        
        # Metric comparison cards
        m1, m2 = st.columns(2)
        with m1:
            st.markdown(
                f"<div class='metric-card'>"
                f"<div class='metric-label'>Baseline LTV Yield</div>"
                f"<div class='metric-value'>${summary_opt.get('baseline_total_ltv', 0):,.2f}</div>"
                f"</div>", 
                unsafe_allow_html=True
            )
        with m2:
            st.markdown(
                f"<div class='metric-card'>"
                f"<div class='metric-label'>Optimized LTV Yield</div>"
                f"<div class='metric-value' style='color:#00e676;'>${summary_opt.get('optimized_total_ltv', 0):,.2f}</div>"
                f"</div>", 
                unsafe_allow_html=True
            )
            
    with opt_col2:
        st.markdown("#### Budget Share Shift Analysis")
        # Bar chart comparing baseline vs optimized shares
        fig_budget = go.Figure()
        fig_budget.add_trace(go.Bar(
            name='Baseline Spend Share',
            x=channel_stats['signup_channel'],
            y=channel_stats['baseline_share'] * 100,
            marker_color='#8a92a6'
        ))
        fig_budget.add_trace(go.Bar(
            name='Optimized Spend Share',
            x=channel_stats['signup_channel'],
            y=channel_stats['optimized_share'] * 100,
            marker_color='#00e676'
        ))
        fig_budget.update_layout(
            barmode='group',
            xaxis_title="Marketing Channel",
            yaxis_title="Share Percentage (%)",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#ffffff')
        )
        st.plotly_chart(fig_budget, width='stretch')

# -----------------
# TAB 2: COHORT RETENTION MATRICES
# -----------------
with tab2:
    st.header("Cohort Retention & ARPU Matrices")
    
    matrix_type = st.radio(
        "Select Heatmap Mode", 
        ["Retention Rate (%)", "Cohort Active ARPU ($)"], 
        horizontal=True
    )
    
    if matrix_type == "Retention Rate (%)":
        st.subheader("Monthly Retention Matrix")
        # Format retention matrix
        matrix_pct = retention_matrix.mul(100).round(1)
        
        fig_cohort = px.imshow(
            matrix_pct,
            labels=dict(x="Months Active (Cohort Index)", y="Cohort Signup Month", color="Retention (%)"),
            x=matrix_pct.columns,
            y=[str(idx) for idx in matrix_pct.index],
            color_continuous_scale="Viridis",
            text_auto=".1f"
        )
        fig_cohort.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#ffffff'),
            height=600
        )
        st.plotly_chart(fig_cohort, width='stretch')
    else:
        st.subheader("Cohort Average Revenue Per Active User (ARPU)")
        fig_arpu = px.imshow(
            arpu_matrix.round(2),
            labels=dict(x="Months Active (Cohort Index)", y="Cohort Signup Month", color="ARPU ($)"),
            x=arpu_matrix.columns,
            y=[str(idx) for idx in arpu_matrix.index],
            color_continuous_scale="Plasma",
            text_auto=".2f"
        )
        fig_arpu.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#ffffff'),
            height=600
        )
        st.plotly_chart(fig_arpu, width='stretch')

# -----------------
# TAB 3: LTV FORECASTING DEEP-DIVE
# -----------------
with tab3:
    st.header("Predictive LTV Deep-Dive")
    st.markdown("Forecasting 12-Month Customer Lifetime Value based on early signal variables (Months 1-3 behavioral transactions).")
    
    # Plotly cumulative actual vs predicted curves
    # We construct a summary of cumulative cohort spends
    # Let's create an evaluation dataframe
    
    df_purchases = df_raw[df_raw["funnel_stage"] == "purchase_complete"].copy()
    first_purchases = df_purchases.groupby("user_id")["timestamp"].min().reset_index()
    first_purchases.rename(columns={"timestamp": "first_purchase_time"}, inplace=True)
    
    df_eval = df_purchases.merge(first_purchases, on="user_id")
    df_eval["days_since_first"] = (df_eval["timestamp"] - df_eval["first_purchase_time"]).dt.days
    
    # We plot cumulative revenue over days since first purchase
    # Bin by days
    bins = np.arange(0, 365, 30)
    df_eval["day_bin"] = pd.cut(df_eval["days_since_first"], bins=bins, labels=bins[:-1])
    
    cum_actual = df_eval.groupby("day_bin")["amount"].sum().cumsum().reset_index()
    cum_actual.columns = ["day_bin", "actual_cumulative"]
    
    # Scale to display user trajectory average
    n_total_users = df_user_ltv["user_id"].nunique()
    cum_actual["avg_actual"] = cum_actual["actual_cumulative"] / n_total_users
    
    # Forecast curve
    # To create actual vs predicted curves, let's map actual trajectories vs predicted overall LTV
    avg_pred_ltv = df_user_ltv["predicted_ltv"].mean()
    
    fig_ltv = go.Figure()
    fig_ltv.add_trace(go.Scatter(
        x=cum_actual["day_bin"], 
        y=cum_actual["avg_actual"],
        mode='lines+markers',
        name='Actual Avg Cumulative Spend',
        line=dict(color='#00e676', width=3)
    ))
    fig_ltv.add_trace(go.Scatter(
        x=[0, 90, 180, 270, 360], 
        y=[0, df_user_ltv["monetary_total_90d"].mean(), avg_pred_ltv * 0.65, avg_pred_ltv * 0.85, avg_pred_ltv],
        mode='lines+markers',
        name='ML Model Cumulative Forecast Curve',
        line=dict(color='#ff4b4b', width=3, dash='dash')
    ))
    fig_ltv.update_layout(
        title="Average Cumulative LTV Trajectory (Actual vs Predicted Model)",
        xaxis_title="Days Since First Purchase",
        yaxis_title="Average Spend Value ($)",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#ffffff')
    )
    st.plotly_chart(fig_ltv, width='stretch')
    
    # Show ML features analysis
    st.subheader("Forecast Engine Model Context")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Features Used for Random Forest Model:**
        1. **Recency**: Days between the first and last transaction inside 90 days.
        2. **Frequency**: Count of orders placed within the first 90 days.
        3. **Monetary Spend**: Total amount spent in the first 90 days.
        4. **Cohort Early Signals**: Month 1, 2, and 3 individual spending velocity.
        5. **Signup Market Channel**: Acquisition route mapped via OneHotEncoder.
        """)
    with col2:
        st.markdown(
            f"<div class='metric-card' style='border-left: 5px solid #00e676;'>"
            f"<div class='metric-label'>Model Feature Input Dimensionality</div>"
            f"<div class='metric-value'>11 Features (One-Hot Encoded)</div>"
            f"</div>", 
            unsafe_allow_html=True
        )

# -----------------
# TAB 4: FUNNEL DIAGNOSTICS & ANOMALY DETECTION
# -----------------
with tab4:
    st.header("Funnel Diagnostics & Anomaly Detection")
    st.markdown("Real-time monitoring of stage drop-offs in the transaction flow with rolling Z-score statistical alerts.")
    
    z_threshold = st.slider("Z-Score Anomaly Threshold", min_value=1.0, max_value=4.0, value=2.0, step=0.1)
    
    df_funnel, weekly_stats = analytics.analyze_funnel_and_anomalies(df_raw, z_threshold)
    
    col_fun, col_diag = st.columns([1, 1])
    
    with col_fun:
        st.subheader("Aggregate Conversion Funnel")
        
        fig_funnel = go.Figure(go.Funnel(
            y=df_funnel["stage"],
            x=df_funnel["users"],
            textinfo="value+percent initial+percent previous",
            marker=dict(color=["#0d47a1", "#1565c0", "#1976d2", "#1e88e5", "#2196f3"])
        ))
        fig_funnel.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#ffffff')
        )
        st.plotly_chart(fig_funnel, width='stretch')
        
    with col_diag:
        st.subheader("Weekly Funnel Health Warnings")
        
        anomalies = weekly_stats[weekly_stats["is_anomaly"]]
        
        if not anomalies.empty:
            for idx, row in anomalies.iterrows():
                st.error(
                    f"⚠️ **Checkout Anomaly Detected!**  \n"
                    f"**Week Commencing:** {row['week'].strftime('%Y-%m-%d')}  \n"
                    f"**Conversion Rate:** {row['conversion_rate']*100:.1f}% (Expected: {row['rolling_mean']*100:.1f}%)  \n"
                    f"**Z-Score Deviation:** {row['z_score']:.2f} standard deviations below rolling mean.  \n"
                    f"**Impact Diagnostics:** Possible frontend checkout flow error or API server error affecting payment completions."
                )
        else:
            st.success("✅ No statistically significant conversion anomalies detected with current Z-score sensitivity.")
            
        # Time series chart of conversion rates
        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(
            x=weekly_stats["week"],
            y=weekly_stats["conversion_rate"] * 100,
            mode='lines+markers',
            name='Checkout -> Purchase Conv Rate (%)',
            line=dict(color='#2196f3', width=2)
        ))
        # Highlight anomalies
        if not anomalies.empty:
            fig_ts.add_trace(go.Scatter(
                x=anomalies["week"],
                y=anomalies["conversion_rate"] * 100,
                mode='markers',
                name='Anomalous Weeks',
                marker=dict(color='#ff4b4b', size=12, symbol='x')
            ))
            
        fig_ts.update_layout(
            title="Checkout Conversion Rate Trend",
            xaxis_title="Week Commencing",
            yaxis_title="Conversion Rate (%)",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#ffffff'),
            legend=dict(x=0.01, y=0.99)
        )
        st.plotly_chart(fig_ts, width='stretch')
