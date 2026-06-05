import numpy as np
import pandas as pd

def generate_mock_data(n_users=10000, seed=42):
    """
    Generates a realistic transactional & funnel event log dataframe.
    - n_users: Number of unique users to simulate
    - Spans 24 months
    - Columns: user_id, timestamp, amount, signup_market_channel, cac, funnel_stage
    - Has conversion drop-offs and a checkout drop-off anomaly in a specific week.
    """
    np.random.seed(seed)
    
    # 1. Generate users
    user_ids = [f"USR_{i:06d}" for i in range(1, n_users + 1)]
    
    # Channel distribution and CAC parameters
    channels = ["Paid Search", "Paid Social", "Organic", "Referral"]
    channel_probs = [0.3, 0.4, 0.2, 0.1]
    channel_cacs = {
        "Paid Search": 45.0,
        "Paid Social": 35.0,
        "Organic": 0.0,
        "Referral": 15.0
    }
    
    user_channels = np.random.choice(channels, size=n_users, p=channel_probs)
    user_cacs = np.array([channel_cacs[ch] for ch in user_channels])
    
    # User signup dates distributed over 24 months (e.g. from 2024-06-01 to 2026-05-31)
    start_date = pd.to_datetime("2024-06-01")
    end_date = pd.to_datetime("2026-05-31")
    total_days = (end_date - start_date).days
    
    # Random signups distributed exponentially (growing business)
    signup_days = np.random.beta(a=2, b=1, size=n_users) * total_days
    signup_dates = start_date + pd.to_timedelta(signup_days, unit="D")
    
    # User DataFrame
    df_users = pd.DataFrame({
        "user_id": user_ids,
        "signup_date": signup_dates,
        "signup_market_channel": user_channels,
        "cac": user_cacs
    })
    
    # 2. Generate Sessions/Funnel Events
    # Funnel stages: home_page -> product_view -> cart_add -> checkout_start -> purchase_complete
    # Base transition probabilities:
    # home_page -> product_view: 80%
    # product_view -> cart_add: 40% (32% cumulative)
    # cart_add -> checkout_start: 50% (16% cumulative)
    # checkout_start -> purchase_complete: 60% (9.6% cumulative)
    
    records = []
    
    # Let's generate multiple sessions per user to simulate retention over 24 months
    # Active users drop over time (retention rate decays)
    # We will simulate events at user-month level or session-level.
    
    all_rows = []
    
    # Vectorized generation of sessions
    # Each user gets a random number of session attempts over time.
    # Users signed up early have more months to have sessions.
    
    # Generate number of sessions per user based on tenure
    user_tenure_days = (end_date - df_users["signup_date"]).dt.days.values
    
    # Average frequency: e.g., 1-5 sessions in the first month, decaying in subsequent months
    # Let's generate session timestamps relative to signup
    # To keep it highly performant, we generate session offsets using exponential decay for active months
    
    sessions_user_id = []
    sessions_timestamp = []
    
    # For each user, draw number of sessions
    # Let's sample session counts
    max_possible_sessions = np.random.poisson(lam=12, size=n_users) + 1
    
    # Vectorized generation of sessions per user
    user_indices = np.repeat(np.arange(n_users), max_possible_sessions)
    
    # Session offsets (days from signup) using exponential distribution to model retention decay
    # We clip the offsets so they don't exceed the user's available tenure
    offsets = np.random.exponential(scale=60, size=len(user_indices)) # average active period of 60 days
    # Let's add some recurrent purchasing behavior for highly retained users
    recurrent_users = np.random.choice([0, 1], size=n_users, p=[0.8, 0.2])
    recurrent_offsets = np.random.exponential(scale=300, size=len(user_indices))
    
    is_recurrent = recurrent_users[user_indices]
    final_offsets = np.where(is_recurrent == 1, recurrent_offsets, offsets)
    
    # Filter offsets that exceed tenure
    tenures = user_tenure_days[user_indices]
    valid_mask = final_offsets <= tenures
    
    user_indices = user_indices[valid_mask]
    final_offsets = final_offsets[valid_mask]
    
    # Create session times
    session_times = df_users["signup_date"].values[user_indices] + pd.to_timedelta(final_offsets, unit="D")
    
    # Now simulate the funnel for these sessions
    n_sessions = len(user_indices)
    
    # Funnel paths: home_page -> product_view -> cart_add -> checkout_start -> purchase_complete
    # We simulate step-by-step progress using uniform random draws.
    # At each stage, if a user fails to convert, their funnel ends.
    
    p_pv = 0.85
    p_ca = 0.45
    p_cs = 0.50
    p_pc = 0.60
    
    # Let's inject an anomaly: During the week of Oct 10 to Oct 20, 2025, there is a technical issue
    # on checkout_start causing a 40% drop-off in completion (p_pc drops by 40%, i.e., from 0.60 to 0.36)
    anomaly_start = pd.to_datetime("2025-10-10")
    anomaly_end = pd.to_datetime("2025-10-20")
    
    # Vectorized conversion decisions
    r_pv = np.random.rand(n_sessions) < p_pv
    r_ca = (np.random.rand(n_sessions) < p_ca) & r_pv
    r_cs = (np.random.rand(n_sessions) < p_cs) & r_ca
    
    # To apply anomaly, check session times
    in_anomaly_window = (session_times >= anomaly_start) & (session_times <= anomaly_end)
    p_pc_adjusted = np.where(in_anomaly_window, p_pc * 0.5, p_pc) # 50% relative drop-off (equivalent to 50% checkout drop-off)
    
    r_pc = (np.random.rand(n_sessions) < p_pc_adjusted) & r_cs
    
    # Create the rows for each session
    # A session always has 'home_page'.
    # If r_pv is true, it has 'product_view'.
    # If r_ca is true, it has 'cart_add'.
    # If r_cs is true, it has 'checkout_start'.
    # If r_pc is true, it has 'purchase_complete'.
    
    # We will build a flat DataFrame of events.
    # To optimize, we generate a block of events.
    stages = ['home_page', 'product_view', 'cart_add', 'checkout_start', 'purchase_complete']
    
    # Create dataframes for each stage and concat
    df_home = pd.DataFrame({
        "user_id": df_users["user_id"].values[user_indices],
        "timestamp": session_times,
        "amount": 0.0,
        "signup_market_channel": df_users["signup_market_channel"].values[user_indices],
        "cac": df_users["cac"].values[user_indices],
        "funnel_stage": "home_page"
    })
    
    df_pv = df_home[r_pv].copy()
    df_pv["funnel_stage"] = "product_view"
    # Delay events slightly within the session
    df_pv["timestamp"] += pd.to_timedelta(np.random.randint(1, 5, size=len(df_pv)), unit="m")
    
    df_ca = df_pv.loc[r_ca[r_pv]].copy()
    df_ca["funnel_stage"] = "cart_add"
    df_ca["timestamp"] += pd.to_timedelta(np.random.randint(1, 5, size=len(df_ca)), unit="m")
    
    df_cs = df_ca.loc[r_cs[r_ca]].copy()
    df_cs["funnel_stage"] = "checkout_start"
    df_cs["timestamp"] += pd.to_timedelta(np.random.randint(1, 5, size=len(df_cs)), unit="m")
    
    df_pc = df_cs.loc[r_pc[r_cs]].copy()
    df_pc["funnel_stage"] = "purchase_complete"
    df_pc["timestamp"] += pd.to_timedelta(np.random.randint(1, 3, size=len(df_pc)), unit="m")
    # Purchase complete has non-zero amount
    # Amount is lognormal to simulate realistic e-commerce ticket size
    df_pc["amount"] = np.random.lognormal(mean=3.8, sigma=0.5, size=len(df_pc))
    df_pc["amount"] = df_pc["amount"].round(2)
    
    df_all = pd.concat([df_home, df_pv, df_ca, df_cs, df_pc], ignore_index=True)
    df_all.sort_values(by="timestamp", inplace=True)
    df_all.reset_index(drop=True, inplace=True)
    
    return df_all
