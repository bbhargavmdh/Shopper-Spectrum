"""
🛒 Shopper Spectrum — Streamlit App
Modules:
  1. Product Recommendation (Item-Based Collaborative Filtering)
  2. Customer Segmentation (KMeans Clustering on RFM)

NOTE: Models are built at runtime from online_retail.csv so no pickle files
are needed — works on Streamlit Cloud out of the box.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os

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


# ── Find dataset (works locally and on Streamlit Cloud) ───────────────────────
def find_csv():
    candidates = [
        os.path.join(os.path.dirname(__file__), "online_retail.csv"),
        "online_retail.csv",
        os.path.join(os.path.dirname(__file__), "data", "online_retail.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


# ── Build everything from scratch (cached so it only runs once per session) ───
@st.cache_resource(show_spinner=False)
def build_models():
    csv_path = find_csv()
    if csv_path is None:
        st.error("❌ Could not find online_retail.csv. Please place it in the same folder as app.py.")
        st.stop()

    # ── Load & clean ──────────────────────────────────────────────────────────
    df = pd.read_csv(csv_path, encoding="latin-1")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df = df.dropna(subset=["CustomerID"])
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    df["CustomerID"] = df["CustomerID"].astype(int)
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

    # ── RFM ───────────────────────────────────────────────────────────────────
    snapshot = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = df.groupby("CustomerID").agg(
        Recency   =("InvoiceDate", lambda x: (snapshot - x.max()).days),
        Frequency =("InvoiceNo",   "nunique"),
        Monetary  =("TotalPrice",  "sum"),
    ).reset_index()

    # ── Clustering ────────────────────────────────────────────────────────────
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])
    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    rfm["Cluster"] = km.fit_predict(rfm_scaled)

    # Auto-label clusters by composite RFM score
    summary = rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean()
    scores = {c: -r["Recency"] + r["Frequency"] * 50 + r["Monetary"] / 100
              for c, r in summary.iterrows()}
    ordered = sorted(scores, key=scores.get, reverse=True)
    rank_map = {ordered[i]: lbl for i, lbl in
                enumerate(["High-Value", "Regular", "Occasional", "At-Risk"])}
    rfm["Segment"] = rfm["Cluster"].map(rank_map)

    # ── Item-based collaborative filtering ────────────────────────────────────
    uk = df[df["Country"] == "United Kingdom"]
    prod_matrix = (uk.groupby(["CustomerID", "Description"])["Quantity"]
                     .sum().unstack(fill_value=0))
    item_sim = cosine_similarity(prod_matrix.T)
    sim_df = pd.DataFrame(item_sim,
                          index=prod_matrix.columns,
                          columns=prod_matrix.columns)

    return km, scaler, rank_map, sim_df, rfm


# ── Run model build with a nice spinner ───────────────────────────────────────
with st.spinner("🔧 Building models from dataset — takes ~30 seconds on first load…"):
    km_model, scaler, rank_map, item_sim_df, rfm_df = build_models()

all_products = item_sim_df.index.tolist()

# ── Helpers ───────────────────────────────────────────────────────────────────
SEGMENT_INFO = {
    "High-Value":  ("🔴", "VIP customer — recent, frequent, high-spending. Offer loyalty rewards and exclusive deals."),
    "Regular":     ("🔵", "Steady purchaser with moderate activity. Great candidate for upselling and cross-sell campaigns."),
    "Occasional":  ("🟢", "Buys infrequently but still active. Targeted promotions can increase visit frequency."),
    "At-Risk":     ("🟡", "Has not purchased in a long time. Initiate a win-back campaign with discounts or personal outreach."),
}
COLORS = {"High-Value": "#e74c3c", "Regular": "#3498db",
          "Occasional": "#2ecc71", "At-Risk": "#f39c12"}


def get_recommendations(product_name: str, top_n: int = 5):
    name = product_name.upper().strip()
    if name not in item_sim_df.index:
        matches = [p for p in item_sim_df.index if name in p]
        if not matches:
            return None, []
        name = matches[0]
    sims = item_sim_df[name].drop(index=name)
    top = sims.sort_values(ascending=False).head(top_n).index.tolist()
    return name, top


def predict_segment(recency, frequency, monetary):
    X = np.array([[recency, frequency, monetary]])
    X_sc = scaler.transform(X)
    cluster = km_model.predict(X_sc)[0]
    return rank_map[cluster]


# ── Sidebar ───────────────────────────────────────────────────────────────────
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
    c2.metric("Clusters", "4")
    seg_counts = rfm_df["Segment"].value_counts()
    for seg, cnt in seg_counts.items():
        icon = SEGMENT_INFO[seg][0]
        st.markdown(f"{icon} **{seg}**: {cnt:,}")
    st.markdown("---")
    st.caption("KMeans · Cosine Similarity · Streamlit")


# ══════════════════════════════════════════════════════════════════════════════
# HOME
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
    c1.metric("👥 Customers", f"{len(rfm_df):,}")
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
        to find products frequently bought together.
        - Input any product name
        - Get **5 similar product** recommendations instantly
        """)
    with col_r:
        st.markdown("### 👥 Customer Segmentation")
        st.markdown("""
        Uses **KMeans Clustering on RFM** to classify customers
        into actionable segments.
        - Enter Recency, Frequency, Monetary values
        - Instantly predict the customer segment
        """)

    st.markdown("---")
    st.markdown("### 🎯 Segment Definitions")
    for seg, (icon, desc) in SEGMENT_INFO.items():
        with st.expander(f"{icon} {seg}"):
            st.write(desc)


# ══════════════════════════════════════════════════════════════════════════════
# RECOMMENDATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📦 Recommendation":
    st.markdown("## 📦 Product Recommender")
    st.markdown("Enter a product name to get the **5 most similar products** based on purchase history.")
    st.markdown("---")

    col_in, col_out = st.columns([1, 1])

    with col_in:
        product_input = st.text_input(
            "Enter Product Name",
            placeholder="e.g. WHITE HANGING HEART T-LIGHT HOLDER",
            help="Case-insensitive. Partial names are also matched.",
        )

        # Live suggestions
        if product_input:
            suggestions = [p for p in all_products if product_input.upper() in p][:8]
            if suggestions and suggestions != [product_input.upper()]:
                selected = st.selectbox("Matching products:", ["-- select --"] + suggestions)
                if selected != "-- select --":
                    product_input = selected

        recommend_btn = st.button("🔍 Get Recommendations", use_container_width=True)

        st.markdown("---")
        st.markdown("**💡 Try these:**")
        for ex in [
            "WHITE HANGING HEART T-LIGHT HOLDER",
            "REGENCY CAKESTAND 3 TIER",
            "JUMBO BAG RED RETROSPOT",
        ]:
            if st.button(ex, key=ex, use_container_width=True):
                product_input = ex
                recommend_btn = True

    with col_out:
        if recommend_btn and product_input:
            matched, recs = get_recommendations(product_input)
            if not recs:
                st.error(f"❌ '{product_input}' not found. Try a partial name like 'HEART' or 'CANDLE'.")
            else:
                st.success(f"✅ Recommendations for:")
                st.markdown(f"**{matched}**")
                st.markdown("#### 🎁 Similar Products")
                for i, rec in enumerate(recs, 1):
                    st.markdown(f'<div class="rec-card">#{i} &nbsp; {rec}</div>',
                                unsafe_allow_html=True)
                st.caption("Similarity computed using Cosine Similarity on customer purchase vectors.")
        elif recommend_btn:
            st.warning("⚠️ Please enter a product name first.")


# ══════════════════════════════════════════════════════════════════════════════
# SEGMENTATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "👥 Segmentation":
    st.markdown("## 👥 Customer Segmentation")
    st.markdown("Enter RFM values to predict which segment this customer belongs to.")
    st.markdown("---")

    col_form, col_res = st.columns([1, 1])

    with col_form:
        st.markdown("#### 📊 RFM Inputs")
        recency   = st.number_input("Recency (days since last purchase)",   min_value=1,   max_value=1000,   value=30)
        frequency = st.number_input("Frequency (number of purchases)",      min_value=1,   max_value=500,    value=5)
        monetary  = st.number_input("Monetary (total amount spent £)",      min_value=1.0, max_value=500000.0, value=500.0, step=50.0)
        predict_btn = st.button("🎯 Predict Segment", use_container_width=True)

        st.markdown("---")
        st.markdown("**💡 Example profiles:**")
        examples = {
            "VIP Customer":      (5,   50, 15000),
            "Regular Buyer":     (30,  6,  800),
            "Occasional Shopper":(90,  2,  200),
            "At-Risk Customer":  (300, 1,  150),
        }
        for name, (r, f, m) in examples.items():
            if st.button(f"{name}  (R={r}, F={f}, M=£{m})", key=name, use_container_width=True):
                recency, frequency, monetary = r, f, m
                predict_btn = True

    with col_res:
        if predict_btn:
            segment = predict_segment(recency, frequency, monetary)
            icon, advice = SEGMENT_INFO[segment]

            st.markdown("#### 🏷️ Predicted Segment")
            st.markdown(
                f'<div class="segment-badge {segment}">{icon} {segment}</div>',
                unsafe_allow_html=True,
            )
            st.markdown("---")
            st.markdown("#### 📌 Recommended Action")
            st.info(advice)

            st.markdown("---")
            st.markdown("#### 📊 Your Values vs Segment Averages")
            seg_avg = rfm_df[rfm_df["Segment"] == segment][
                ["Recency", "Frequency", "Monetary"]].mean()
            comp = pd.DataFrame({
                "Your Input":       [recency, frequency, monetary],
                f"{segment} Avg":   [seg_avg["Recency"], seg_avg["Frequency"], seg_avg["Monetary"]],
            }, index=["Recency (days)", "Frequency (#)", "Monetary (£)"])
            st.dataframe(comp.style.format("{:.1f}"), use_container_width=True)

            st.markdown("---")
            st.markdown("#### 🗂️ All Segment Profiles")
            profile = rfm_df.groupby("Segment")[
                ["Recency", "Frequency", "Monetary"]].mean().round(1)
            st.dataframe(profile, use_container_width=True)
