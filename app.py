"""
🛒 Shopper Spectrum — Streamlit App
Loads dataset directly from Google Drive — no CSV upload needed.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import gdown
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stSidebar"] { background: #1a1a2e; }
    [data-testid="stSidebar"] * { color: #eee !important; }
    .main { background: #f8f9ff; }
    .rec-card {
        background: white;
        border-left: 5px solid #e74c3c;
        border-radius: 8px;
        padding: 12px 18px;
        margin: 8px 0;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
        font-size: 15px;
        font-weight: 500;
        color: #2d2d2d;
    }
    .segment-badge {
        display: inline-block;
        padding: 10px 28px;
        border-radius: 30px;
        font-size: 22px;
        font-weight: 700;
        margin-top: 12px;
        letter-spacing: 1px;
    }
    .High-Value  { background:#fde8e8; color:#c0392b; border:2px solid #e74c3c; }
    .Regular     { background:#e8f4fd; color:#1a6ca8; border:2px solid #3498db; }
    .Occasional  { background:#e8fdf2; color:#1a7a48; border:2px solid #2ecc71; }
    .At-Risk     { background:#fef9e8; color:#9e6c00; border:2px solid #f39c12; }
    .header-strip {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 22px 28px;
        border-radius: 12px;
        margin-bottom: 24px;
    }
</style>
""", unsafe_allow_html=True)

# ── Session State Initialization ─────────────────────────────────────────────
if "rec_input" not in st.session_state:
    st.session_state.rec_input = ""
if "seg_recency" not in st.session_state:
    st.session_state.seg_recency = 30
if "seg_frequency" not in st.session_state:
    st.session_state.seg_frequency = 5
if "seg_monetary" not in st.session_state:
    st.session_state.seg_monetary = 500.0

# ── Download CSV from Google Drive ────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    file_id  = "1x-qk4JQ4z_HITPTa7OD574yBCy1tdIc3"
    url      = f"https://drive.google.com/uc?id={file_id}"
    out_path = "online_retail.csv"
    if not os.path.exists(out_path):
        gdown.download(url, out_path, quiet=True)
    df = pd.read_csv(out_path, encoding="latin-1")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df = df.dropna(subset=["CustomerID"])
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    df["CustomerID"] = df["CustomerID"].astype(int)
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    return df

# ── Build models (cached so runs only once per session) ───────────────────────
@st.cache_resource(show_spinner=False)
def build_models():
    df = load_data()
    # RFM Computation
    snapshot = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = df.groupby("CustomerID").agg(
        Recency   =("InvoiceDate", lambda x: (snapshot - x.max()).days),
        Frequency =("InvoiceNo",   "nunique"),
        Monetary  =("TotalPrice",  "sum"),
    ).reset_index()
    
    # KMeans Clustering
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])
    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    rfm["Cluster"] = km.fit_predict(rfm_scaled)
    
    # Auto-label clusters dynamically based on internal business value logic
    summary = rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean()
    scores  = {c: -r["Recency"] + r["Frequency"] * 50 + r["Monetary"] / 100 
               for c, r in summary.iterrows()}
    ordered  = sorted(scores, key=scores.get, reverse=True)
    rank_map = {ordered[i]: lbl for i, lbl in 
                enumerate(["High-Value", "Regular", "Occasional", "At-Risk"])}
    rfm["Segment"] = rfm["Cluster"].map(rank_map)
    
    # Item-based collaborative filtering (UK subsegment to save computational memory)
    uk = df[df["Country"] == "United Kingdom"]
    prod_matrix = (uk.groupby(["CustomerID", "Description"])["Quantity"]
                     .sum().unstack(fill_value=0))
    sim_arr = cosine_similarity(prod_matrix.T)
    sim_df  = pd.DataFrame(sim_arr, 
                           index=prod_matrix.columns, 
                           columns=prod_matrix.columns)
    return km, scaler, rank_map, sim_df, rfm

# ── Load everything with a single global startup spinner ──────────────────────
with st.spinner("⏳ Downloading dataset & building models — please wait ~30 seconds…"):
    km_model, scaler, rank_map, item_sim_df, rfm_df = build_models()

all_products = item_sim_df.index.tolist()
seg_counts   = rfm_df["Segment"].value_counts()

# ── Constants & Metadata ──────────────────────────────────────────────────────
SEGMENT_INFO = {
    "High-Value": ("🔴", "VIP customer — recent, frequent, high-spending. Offer loyalty rewards and exclusive deals."),
    "Regular":    ("🔵", "Steady purchaser with moderate activity. Great candidate for upselling and cross-sell campaigns."),
    "Occasional": ("🟢", "Buys infrequently but still active. Targeted promotions can increase visit frequency."),
    "At-Risk":    ("🟡", "Has not purchased in a long time. Initiate a win-back campaign with discounts or personal outreach."),
}

# ── Helper Engines ────────────────────────────────────────────────────────────
def get_recommendations(product_name: str, top_n: int = 5):
    name = product_name.upper().strip()
    if name not in item_sim_df.index:
        matches = [p for p in item_sim_df.index if name in p]
        if not matches:
            return None, []
        name = matches[0]
    sims = item_sim_df[name].drop(index=name)
    return name, sims.sort_values(ascending=False).head(top_n).index.tolist()

def predict_segment(recency, frequency, monetary):
    X_sc  = scaler.transform(pd.DataFrame(
        [[recency, frequency, monetary]], 
        columns=["Recency", "Frequency", "Monetary"]
    ))
    return rank_map[km_model.predict(X_sc)[0]]

# ── Sidebar Navigation & Metrics Panel ────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛒 Shopper Spectrum")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 Home", "📦 Recommendation", "👥 Segmentation"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("**Dataset Stats**")
    c1, c2 = st.columns(2)
    c1.metric("Customers", f"{len(rfm_df):,}")
    c2.metric("Clusters",  "4")
    for seg, cnt in seg_counts.items():
        st.markdown(f"{SEGMENT_INFO[seg][0]} **{seg}**: {cnt:,}")
    st.markdown("---")
    st.caption("KMeans · Cosine Similarity · Streamlit")

# ══════════════════════════════════════════════════════════════════════════════
# HOME VIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown("""
    <div class="header-strip">
        <h1 style="margin:0;font-size:32px;">🛒 Shopper Spectrum</h1>
        <p style="margin:4px 0 0;font-size:16px;opacity:0.9;">
            Customer Segmentation &amp; Product Recommendations — E-Commerce Intelligence
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("👥 Customers",   f"{len(rfm_df):,}")
    c2.metric("🔴 High-Value",  seg_counts.get("High-Value",  0))
    c3.metric("🔵 Regular",     seg_counts.get("Regular",     0))
    c4.metric("🟢 Occasional",  seg_counts.get("Occasional",  0))
    c5.metric("🟡 At-Risk",     seg_counts.get("At-Risk",     0))
    
    st.markdown("---")
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("### 📦 Product Recommendation")
        st.markdown("""
        Uses **Item-Based Collaborative Filtering** with cosine similarity 
        to find products frequently bought together based on co-occurrence vectors.
        - Input any custom store SKU or catalog profile description.
        - Surface the top **5 optimal cross-sell targets** dynamically.
        """)
    with col_r:
        st.markdown("### 👥 Customer Segmentation")
        st.markdown("""
        Uses unsupervised **KMeans Clustering on normalized RFM metrics** to map behavioral boundaries instantly.
        - Input manual customer engagement numbers.
        - Evaluate live cohorts against the real database averages.
        """)
    st.markdown("---")
    st.markdown("### 🎯 Segment Definitions")
    for seg, (icon, desc) in SEGMENT_INFO.items():
        with st.expander(f"{icon} {seg}"):
            st.write(desc)

# ══════════════════════════════════════════════════════════════════════════════
# RECOMMENDATION ENGINE VIEW
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📦 Recommendation":
    st.markdown("## 📦 Product Recommender")
    st.markdown("Enter a product name to get the **5 most similar products** based on historical patterns.")
    st.markdown("---")
    
    col_in, col_out = st.columns([1, 1])
    
    with col_in:
        st.markdown("**💡 Try these preset profiles:**")
        ex_products = [
            "WHITE HANGING HEART T-LIGHT HOLDER",
            "REGENCY CAKESTAND 3 TIER",
            "JUMBO BAG RED RETROSPOT",
        ]
        for ex in ex_products:
            if st.button(ex, key=f"btn_{ex}", use_container_width=True):
                st.session_state.rec_input = ex
                st.rerun()

        st.markdown("---")
        
        product_input = st.text_input(
            "Enter Product Name",
            value=st.session_state.rec_input,
            placeholder="e.g. WHITE HANGING HEART T-LIGHT HOLDER",
            help="Case-insensitive. Partial terms are structural matches.",
            key="main_product_input"
        )
        st.session_state.rec_input = product_input

        if product_input:
            suggestions = [p for p in all_products if product_input.upper() in p][:8]
            if suggestions and suggestions != [product_input.upper()]:
                selected = st.selectbox("Matching catalog targets:", ["-- select profile --"] + suggestions)
                if selected != "-- select profile --":
                    st.session_state.rec_input = selected
                    st.rerun()

        recommend_btn = st.button("🔍 Get Recommendations", type="primary", use_container_width=True)

    with col_out:
        if (recommend_btn or st.session_state.rec_input) and st.session_state.rec_input:
            matched, recs = get_recommendations(st.session_state.rec_input)
            if not recs:
                st.error(f"❌ '{st.session_state.rec_input}' not found. Try variations such as 'HEART' or 'BAG'.")
            else:
                st.success("✅ Showing recommendations for:")
                st.markdown(f"**{matched}**")
                st.markdown("#### 🎁 Ranked Recommendation Inferences")
                for i, rec in enumerate(recs, 1):
                    st.markdown(f'<div class="rec-card">#{i} &nbsp; {rec}</div>', 
                                unsafe_allow_html=True)
                st.caption("Derived via Cosine Similarity computations on explicit transactional vectors.")
        elif recommend_btn:
            st.warning("⚠️ Enter a target item label above first.")

# ══════════════════════════════════════════════════════════════════════════════
# SEGMENTATION PANEL VIEW
# ══════════════════════════════════════════════════════════════════════════════
elif page == "👥 Segmentation":
    st.markdown("## 👥 Customer Segmentation Engine")
    st.markdown("Map a target user profile to its performance segment via trained parameters.")
    st.markdown("---")
    
    col_form, col_res = st.columns([1, 1])
    
    with col_form:
        st.markdown("**💡 Try preset customer profiles:**")
        examples = {
            "VIP Customer Profile":       (5,   50, 15000.0),
            "Regular Buyer Profile":      (30,  6,  800.0),
            "Occasional Shopper Profile": (90,  2,  200.0),
            "At-Risk Customer Profile":   (300, 1,  150.0),
        }
        for name, (r, f, m) in examples.items():
            if st.button(f"{name}  (R={r}, F={f}, M=£{m})", key=f"prof_{name}", use_container_width=True):
                st.session_state.seg_recency = r
                st.session_state.seg_frequency = f
                st.session_state.seg_monetary = float(m)
                st.rerun()

        st.markdown("---")
        st.markdown("#### 📊 Metric Parameter Form")
        
        recency = st.number_input("Recency (days since last transaction)", 
                                  min_value=1, max_value=1000, 
                                  value=st.session_state.seg_recency, key="input_rec")
        frequency = st.number_input("Frequency (total lifetime orders)", 
                                    min_value=1, max_value=500, 
                                    value=st.session_state.seg_frequency, key="input_freq")
        monetary = st.number_input("Monetary Value (cumulative spend £)", 
                                   min_value=1.0, max_value=500000.0, 
                                   value=st.session_state.seg_monetary, step=50.0, key="input_mon")
        
        st.session_state.seg_recency = recency
        st.session_state.seg_frequency = frequency
        st.session_state.seg_monetary = monetary

        predict_btn = st.button("🎯 Predict Segment Cohort", type="primary", use_container_width=True)

    with col_res:
        if predict_btn or ("input_rec" in st.session_state):
            segment = predict_segment(st.session_state.seg_recency, st.session_state.seg_frequency, st.session_state.seg_monetary)
            icon, advice = SEGMENT_INFO[segment]
            
            st.markdown("#### 🏷️ Computed Group Allocation")
            st.markdown(
                f'<div class="segment-badge {segment}">{icon} {segment}</div>', 
                unsafe_allow_html=True,
            )
            st.markdown("---")
            st.markdown("#### 📌 Marketing Action Guidance")
            st.info(advice)
            st.markdown("---")
            
            st.markdown("#### 📊 Target Input vs Selected Segment Center Averages")
            seg_avg = rfm_df[rfm_df["Segment"] == segment][
                ["Recency", "Frequency", "Monetary"]].mean()
            
            comp = pd.DataFrame({
                "Current Input":   [float(st.session_state.seg_recency), float(st.session_state.seg_frequency), float(st.session_state.seg_monetary)],
                f"{segment} Avg": [seg_avg["Recency"], seg_avg["Frequency"], seg_avg["Monetary"]],
            }, index=["Recency (Days)", "Frequency (#)", "Monetary (£)"])
            
            st.dataframe(comp.style.format("{:.1f}"), use_container_width=True)
            st.markdown("---")
            
            st.markdown("#### 🗂️ Global Baseline Benchmark Coordinates")
            profile = rfm_df.groupby("Segment")[
                ["Recency", "Frequency", "Monetary"]].mean().round(1)
            st.dataframe(profile, use_container_width=True)