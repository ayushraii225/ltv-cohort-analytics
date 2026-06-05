import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def compute_cohort_matrices(df):
    """
    Computes cohort retention percentage matrix and cohort ARPU matrix.
    Assumes df contains transaction/purchase events or we filter for purchase_complete.
    Cohort is defined by the first purchase month of each user.
    """
    try:
        # Filter for purchases
        df_purchases = df[df["funnel_stage"] == "purchase_complete"].copy()
        if df_purchases.empty:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
            
        # Get first purchase month for each user
        df_purchases["month"] = df_purchases["timestamp"].dt.to_period("M")
        df_user_first_purchase = df_purchases.groupby("user_id")["month"].min().reset_index()
        df_user_first_purchase.rename(columns={"month": "cohort_month"}, inplace=True)
        
        # Merge back
        df_m = df_purchases.merge(df_user_first_purchase, on="user_id")
        
        # Calculate cohort index (months since first purchase)
        df_m["cohort_index"] = (df_m["month"].dt.year - df_m["cohort_month"].dt.year) * 12 + \
                               (df_m["month"].dt.month - df_m["cohort_month"].dt.month)
                               
        # Retention Matrix: Unique users per cohort_month and cohort_index
        cohort_group = df_m.groupby(["cohort_month", "cohort_index"])
        cohort_data = cohort_group.agg(
            unique_users=("user_id", "nunique"),
            total_revenue=("amount", "sum")
        ).reset_index()
        
        # Get cohort sizes (index = 0)
        cohort_sizes = cohort_data[cohort_data["cohort_index"] == 0][["cohort_month", "unique_users"]]
        cohort_sizes.rename(columns={"unique_users": "cohort_size"}, inplace=True)
        
        cohort_data = cohort_data.merge(cohort_sizes, on="cohort_month")
        cohort_data["retention_rate"] = cohort_data["unique_users"] / cohort_data["cohort_size"]
        cohort_data["arpu"] = cohort_data["total_revenue"] / cohort_data["unique_users"]
        
        # Pivot matrices
        retention_matrix = cohort_data.pivot(index="cohort_month", columns="cohort_index", values="retention_rate")
        arpu_matrix = cohort_data.pivot(index="cohort_month", columns="cohort_index", values="arpu")
        size_df = cohort_sizes.set_index("cohort_month")
        
        return retention_matrix, arpu_matrix, size_df
    except Exception as e:
        print(f"Error computing cohort matrices: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

def train_and_forecast_ltv(df, forecast_horizon_months=12):
    """
    Builds a vectorized Machine Learning model to forecast individual 12/24-month LTV
    based on early transactional behavior (first 3 months).
    """
    try:
        df_purchases = df[df["funnel_stage"] == "purchase_complete"].copy()
        if df_purchases.empty:
            return None, "No purchase transactions available."
            
        # Get first purchase date
        first_purchases = df_purchases.groupby("user_id")["timestamp"].min().reset_index()
        first_purchases.rename(columns={"timestamp": "first_purchase_time"}, inplace=True)
        
        df_m = df_purchases.merge(first_purchases, on="user_id")
        df_m["days_since_first"] = (df_m["timestamp"] - df_m["first_purchase_time"]).dt.days
        
        # Define early window (first 90 days / ~3 months)
        df_early = df_m[df_m["days_since_first"] <= 90]
        
        # Feature Engineering at user level
        user_features = df_early.groupby("user_id").agg(
            frequency=("amount", "count"),
            monetary_avg=("amount", "mean"),
            monetary_total_90d=("amount", "sum"),
            recency=("days_since_first", "max"),
            signup_channel=("signup_market_channel", "first"),
            cac=("cac", "first"),
            tenure_total=("timestamp", lambda x: (df["timestamp"].max() - x.min()).days)
        ).reset_index()
        
        # Let's add monthly spend features in early window
        df_m["month_index"] = df_m["days_since_first"] // 30
        for m in range(3):
            df_m_sp = df_m[df_m["month_index"] == m].groupby("user_id")["amount"].sum().reset_index()
            df_m_sp.rename(columns={"amount": f"spend_month_{m+1}"}, inplace=True)
            user_features = user_features.merge(df_m_sp, on="user_id", how="left").fillna({f"spend_month_{m+1}": 0.0})
            
        # Define target variable: Spend up to 12 months (365 days)
        # We only train on users who have been signed up for at least the forecast horizon
        target_days = forecast_horizon_months * 30
        trainable_mask = user_features["tenure_total"] >= target_days
        
        if trainable_mask.sum() < 50:
            # Fallback if we don't have enough mature cohorts: use a smaller window or train on available
            trainable_mask = user_features["tenure_total"] >= 90
            
        df_target = df_m[df_m["days_since_first"] <= target_days].groupby("user_id")["amount"].sum().reset_index()
        df_target.rename(columns={"amount": "target_ltv"}, inplace=True)
        
        user_features = user_features.merge(df_target, on="user_id", how="left").fillna({"target_ltv": 0.0})
        
        # Machine learning pipeline
        feature_cols = [
            "frequency", "monetary_avg", "monetary_total_90d", "recency", 
            "spend_month_1", "spend_month_2", "spend_month_3", "signup_channel"
        ]
        
        X = user_features[feature_cols]
        y = user_features["target_ltv"]
        
        # Pipelines for numeric vs categoric
        numeric_features = ["frequency", "monetary_avg", "monetary_total_90d", "recency", "spend_month_1", "spend_month_2", "spend_month_3"]
        categorical_features = ["signup_channel"]
        
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_features),
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
            ])
            
        model_pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(n_estimators=50, random_state=42, max_depth=6))
        ])
        
        # Train model
        train_idx = user_features[trainable_mask].index
        if len(train_idx) > 10:
            model_pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
            predictions = model_pipeline.predict(X)
        else:
            # Simple heuristic backup if ML fails due to low count
            predictions = user_features["monetary_total_90d"] * (forecast_horizon_months / 3)
            
        user_features["predicted_ltv"] = predictions
        
        return user_features, None
    except Exception as e:
        import traceback
        return None, f"Error in LTV forecasting: {str(e)}\n{traceback.format_exc()}"

def run_cac_optimization(df_user_ltv, channel_spend_shares, total_budget, margin_pct=0.80):
    """
    Implements a CAC Payback Optimization Framework.
    - Computes LTV/CAC ratios per channel
    - Reallocates budget from low LTV/CAC to high LTV/CAC channels
    - Models the new theoretical performance showing an efficiency gain (target ~14%)
    """
    try:
        # Group by channel to find baseline performance
        channel_stats = df_user_ltv.groupby("signup_channel").agg(
            users=("user_id", "count"),
            avg_cac=("cac", "mean"),
            avg_predicted_ltv=("predicted_ltv", "mean")
        ).reset_index()
        
        # Handle organic channel CAC (cac=0) to prevent division by zero
        channel_stats["ltv_cac_ratio"] = np.where(
            channel_stats["avg_cac"] > 0,
            channel_stats["avg_predicted_ltv"] / channel_stats["avg_cac"],
            channel_stats["avg_predicted_ltv"] / 1.0 # baseline scale for organic
        )
        
        # Baseline total spends
        channel_stats["baseline_share"] = channel_stats["signup_channel"].map(channel_spend_shares)
        # Default to equal if not fully specified
        channel_stats["baseline_share"] = channel_stats["baseline_share"].fillna(1 / len(channel_stats))
        
        # Normalize baseline shares to sum to 1.0
        channel_stats["baseline_share"] /= channel_stats["baseline_share"].sum()
        
        # Calculate baseline stats
        channel_stats["baseline_budget"] = channel_stats["baseline_share"] * total_budget
        channel_stats["baseline_acquisitions"] = np.where(
            channel_stats["avg_cac"] > 0,
            channel_stats["baseline_budget"] / channel_stats["avg_cac"],
            channel_stats["baseline_budget"] / 1.0 # standard scale
        )
        
        channel_stats["baseline_total_ltv"] = channel_stats["baseline_acquisitions"] * channel_stats["avg_predicted_ltv"]
        
        # Optimization: Heuristic allocation based on LTV/CAC ratio strength
        # Shift budget: we penalize low ROI channels and reward high ROI channels.
        scores = channel_stats["ltv_cac_ratio"].values
        # Organic gets budget from organic/word-of-mouth which is hard to scale,
        # so let's limit Organic share shift, focusing mostly on paid channels.
        is_paid = channel_stats["signup_channel"] != "Organic"
        
        # Softmax style allocation over paid channels
        paid_scores = scores[is_paid]
        exp_scores = np.exp(paid_scores - np.max(paid_scores)) # numeric stability
        optimized_paid_shares = exp_scores / exp_scores.sum()
        
        # Set up final shares
        opt_shares = channel_stats["baseline_share"].copy()
        # Keep Organic at baseline, distribute rest among paid
        total_paid_share = channel_stats.loc[is_paid, "baseline_share"].sum()
        
        opt_shares[is_paid] = optimized_paid_shares * total_paid_share
        
        # Smooth between baseline and optimized based on optimizer strength (e.g. 0.8)
        alpha = 0.70 # shift strength
        channel_stats["optimized_share"] = (1 - alpha) * channel_stats["baseline_share"] + alpha * opt_shares
        channel_stats["optimized_share"] /= channel_stats["optimized_share"].sum()
        
        # Optimized values
        channel_stats["optimized_budget"] = channel_stats["optimized_share"] * total_budget
        channel_stats["optimized_acquisitions"] = np.where(
            channel_stats["avg_cac"] > 0,
            channel_stats["optimized_budget"] / channel_stats["avg_cac"],
            channel_stats["optimized_budget"] / 1.0
        )
        channel_stats["optimized_total_ltv"] = channel_stats["optimized_acquisitions"] * channel_stats["avg_predicted_ltv"]
        
        # Efficiency improvements
        baseline_total_acq = channel_stats["baseline_acquisitions"].sum()
        optimized_total_acq = channel_stats["optimized_acquisitions"].sum()
        
        baseline_total_ltv = channel_stats["baseline_total_ltv"].sum()
        optimized_total_ltv = channel_stats["optimized_total_ltv"].sum()
        
        baseline_blended_ltv_cac = baseline_total_ltv / total_budget
        optimized_blended_ltv_cac = optimized_total_ltv / total_budget
        
        efficiency_gain = (optimized_total_ltv - baseline_total_ltv) / baseline_total_ltv
        
        return channel_stats, {
            "baseline_total_ltv": baseline_total_ltv,
            "optimized_total_ltv": optimized_total_ltv,
            "baseline_blended_ltv_cac": baseline_blended_ltv_cac,
            "optimized_blended_ltv_cac": optimized_blended_ltv_cac,
            "efficiency_gain": efficiency_gain,
            "baseline_acquisitions": baseline_total_acq,
            "optimized_acquisitions": optimized_total_acq
        }
    except Exception as e:
        print(f"Error in CAC optimization: {e}")
        return pd.DataFrame(), {}

def analyze_funnel_and_anomalies(df, z_threshold=2.0):
    """
    Computes overall conversion funnel metrics and detects time-series anomalies in conversion rates.
    """
    try:
        stages_order = ["home_page", "product_view", "cart_add", "checkout_start", "purchase_complete"]
        
        # 1. Compute aggregate funnel counts
        funnel_counts = []
        for stage in stages_order:
            count = df[df["funnel_stage"] == stage]["user_id"].nunique()
            funnel_counts.append({"stage": stage, "users": count})
            
        df_funnel = pd.DataFrame(funnel_counts)
        df_funnel["drop_off_pct"] = df_funnel["users"].pct_change().fillna(0) * 100
        # Conversion relative to home page
        df_funnel["conversion_pct"] = (df_funnel["users"] / df_funnel.iloc[0]["users"]) * 100
        
        # 2. Time-series anomaly detection on checkout_start -> purchase_complete conversion
        # We look at weekly data
        df["week"] = df["timestamp"].dt.to_period("W").dt.start_time
        
        # Count checkout_start and purchase_complete per week
        weekly_cs = df[df["funnel_stage"] == "checkout_start"].groupby("week")["user_id"].nunique().rename("checkout_starts")
        weekly_pc = df[df["funnel_stage"] == "purchase_complete"].groupby("week")["user_id"].nunique().rename("purchases")
        
        weekly_stats = pd.concat([weekly_cs, weekly_pc], axis=1).fillna(0)
        weekly_stats["conversion_rate"] = np.where(
            weekly_stats["checkout_starts"] > 0,
            weekly_stats["purchases"] / weekly_stats["checkout_starts"],
            0.0
        )
        
        # Detect anomalies using rolling Z-Score
        # Calculate rolling mean and std
        rolling_window = 4
        weekly_stats["rolling_mean"] = weekly_stats["conversion_rate"].rolling(window=rolling_window, min_periods=1).mean()
        weekly_stats["rolling_std"] = weekly_stats["conversion_rate"].rolling(window=rolling_window, min_periods=1).std().fillna(0.01)
        
        weekly_stats["z_score"] = (weekly_stats["conversion_rate"] - weekly_stats["rolling_mean"]) / weekly_stats["rolling_std"]
        # Flag anomalies where conversion drops significantly below the rolling mean
        weekly_stats["is_anomaly"] = (weekly_stats["z_score"] < -z_threshold) & (weekly_stats["checkout_starts"] > 10)
        
        return df_funnel, weekly_stats.reset_index()
    except Exception as e:
        print(f"Error in funnel analysis: {e}")
        return pd.DataFrame(), pd.DataFrame()
