"""
🛒 Shopper Spectrum — Streamlit App
Modules:
  1. Product Recommendation (Item-Based Collaborative Filtering)
  2. Customer Segmentation (KMeans Clustering on RFM)
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Sidebar */
    [data-testid="stSidebar"] { background: #1a1a2e; }
    [data-testid="stSidebar"] * { color: #eee !important; }

    /* Main background */
    .main { background: #f8f9ff; }

    /* Cards */
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

    /* Header strip */
    .header-strip {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 22px 28px;
        border-radius: 12px;
        margin-bottom: 24px;
    }
    .metric-box {
        background: white;
        border-radius: 10px;
        padding: 14px 18px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    }
</style>
""", unsafe_allow_html=True)


# ── Load Models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    base = os.path.join(os.path.dirname(__file__), "models")
    km     = pickle.load(open(f"{base}/kmeans_model.pkl",   "rb"))
    sc     = pickle.load(open(f"{base}/scaler.pkl",         "rb"))
    rm     = pickle.load(open(f"{base}/rank_map.pkl",       "rb"))
    sim_df = pickle.load(open(f"{base}/item_similarity.pkl","rb"))
    rfm    = pd.read_csv(f"{base}/rfm_data.csv")
    return km, sc, rm, sim_df, rfm

km_model, scaler, rank_map, item_sim_df, rfm_df = load_models()
all_products = item_sim_df.index.tolist()


# ── Recommendation helper ─────────────────────────────────────────────────────
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


# ── Segment helper ────────────────────────────────────────────────────────────
SEGMENT_INFO = {
    "High-Value":  ("🔴", "This customer is a VIP! Recent, frequent, and high-spending. Offer loyalty rewards and exclusive deals."),
    "Regular":     ("🔵", "Steady purchaser with moderate activity. Great candidate for upselling and cross-sell campaigns."),
    "Occasional":  ("🟢", "Buys infrequently but is still active. Targeted promotions can increase visit frequency."),
    "At-Risk":     ("🟡", "Has not purchased in a long time. Initiate a win-back campaign with discounts or personal outreach."),
}

def predict_segment(recency, frequency, monetary):
    X = np.array([[recency, frequency, monetary]])
    X_scaled = scaler.transform(X)
    cluster = km_model.predict(X_scaled)[0]
    return rank_map[cluster]


# ── Sidebar navigation ────────────────────────────────────────────────────────
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
    total_customers = len(rfm_df)
    col1, col2 = st.columns(2)
    col1.metric("Customers", f"{total_customers:,}")
    col2.metric("Clusters", "4")
    st.markdown("---")
    st.caption("Built with ❤️ using KMeans & Cosine Similarity")


# ══════════════════════════════════════════════════════════════════════════════
# HOME PAGE
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown("""
    <div class="header-strip">
        <h1 style="margin:0; font-size:32px;">🛒 Shopper Spectrum</h1>
        <p style="margin:4px 0 0; font-size:16px; opacity:0.9;">
            Customer Segmentation &amp; Product Recommendations — E-Commerce Intelligence Platform
        </p>
    </div>
    """, unsafe_allow_html=True)

    # KPIs
    seg_counts = rfm_df['Segment'].value_counts()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("👥 Total Customers", f"{len(rfm_df):,}")
    c2.metric("🔴 High-Value",  seg_counts.get('High-Value', 0))
    c3.metric("🔵 Regular",     seg_counts.get('Regular', 0))
    c4.metric("🟢 Occasional",  seg_counts.get('Occasional', 0))
    c5.metric("🟡 At-Risk",     seg_counts.get('At-Risk', 0))

    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("### 📦 Product Recommendation")
        st.markdown("""
        Uses **Item-Based Collaborative Filtering** with cosine similarity
        to find products frequently bought together.

        - Input any product name
        - Get **5 similar product** recommendations instantly
        - Powered by customer purchase history patterns
        """)
        if st.button("Go to Recommender →", use_container_width=True):
            st.rerun()

    with col_r:
        st.markdown("### 👥 Customer Segmentation")
        st.markdown("""
        Uses **KMeans Clustering on RFM features** to classify customers
        into actionable segments.

        - Enter Recency, Frequency, Monetary values
        - Instantly predict the customer segment
        - Receive targeted marketing strategy advice
        """)
        if st.button("Go to Segmentation →", use_container_width=True):
            st.rerun()

    st.markdown("---")
    st.markdown("### 🎯 Customer Segment Definitions")
    for seg, (icon, desc) in SEGMENT_INFO.items():
        with st.expander(f"{icon} {seg}"):
            st.write(desc)


# ══════════════════════════════════════════════════════════════════════════════
# RECOMMENDATION PAGE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📦 Recommendation":
    st.markdown("## 📦 Product Recommender")
    st.markdown("Enter a product name to discover the **5 most similar products** based on customer purchase history.")

    st.markdown("---")

    col_input, col_result = st.columns([1, 1])

    with col_input:
        # Autocomplete via selectbox
        product_input = st.text_input(
            "Enter Product Name",
            placeholder="e.g. WHITE HANGING HEART T-LIGHT HOLDER",
            help="Type part of a product name. Case-insensitive.",
        )

        # Suggestions live search
        if product_input:
            suggestions = [p for p in all_products if product_input.upper() in p][:8]
            if suggestions and suggestions != [product_input.upper()]:
                selected = st.selectbox("Matching products:", ["-- Type to filter --"] + suggestions)
                if selected != "-- Type to filter --":
                    product_input = selected

        recommend_btn = st.button("🔍 Get Recommendations", use_container_width=True)

        st.markdown("---")
        st.markdown("**💡 Try these examples:**")
        examples = [
            "WHITE HANGING HEART T-LIGHT HOLDER",
            "REGENCY CAKESTAND 3 TIER",
            "JUMBO BAG RED RETROSPOT",
        ]
        for ex in examples:
            if st.button(ex, key=ex, use_container_width=True):
                product_input = ex
                recommend_btn = True

    with col_result:
        if recommend_btn and product_input:
            matched_name, recs = get_recommendations(product_input)

            if not recs:
                st.error(f"❌ Product '{product_input}' not found in our catalogue.")
                st.info("Try typing a partial name — e.g. 'HEART' or 'CANDLE'")
            else:
                st.success(f"✅ Showing recommendations for:")
                st.markdown(f"**{matched_name}**")
                st.markdown("#### 🎁 Recommended Products")
                for i, rec in enumerate(recs, 1):
                    st.markdown(
                        f'<div class="rec-card">#{i} &nbsp; {rec}</div>',
                        unsafe_allow_html=True,
                    )

                st.markdown("---")
                st.caption("Similarity computed using Cosine Similarity on customer purchase vectors.")
        elif recommend_btn and not product_input:
            st.warning("⚠️ Please enter a product name first.")


# ══════════════════════════════════════════════════════════════════════════════
# SEGMENTATION PAGE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "👥 Segmentation":
    st.markdown("## 👥 Customer Segmentation")
    st.markdown("Enter a customer's **RFM values** to predict their segment.")

    st.markdown("---")
    col_form, col_result = st.columns([1, 1])

    with col_form:
        st.markdown("#### 📊 Enter RFM Values")

        recency = st.number_input(
            "Recency (days since last purchase)",
            min_value=1, max_value=1000, value=30,
            help="Lower = more recent customer"
        )
        frequency = st.number_input(
            "Frequency (number of purchases)",
            min_value=1, max_value=500, value=5,
            help="Higher = more frequent buyer"
        )
        monetary = st.number_input(
            "Monetary (total amount spent £)",
            min_value=1.0, max_value=500000.0, value=500.0, step=50.0,
            help="Total lifetime spend in GBP"
        )

        predict_btn = st.button("🎯 Predict Segment", use_container_width=True)

        st.markdown("---")
        st.markdown("**💡 Example profiles:**")
        examples = {
            "VIP Customer": (5, 50, 15000),
            "Regular Buyer": (30, 6, 800),
            "Occasional Shopper": (90, 2, 200),
            "At-Risk Customer": (300, 1, 150),
        }
        for name, (r, f, m) in examples.items():
            if st.button(f"{name} (R={r}, F={f}, M=£{m})", key=name, use_container_width=True):
                recency, frequency, monetary = r, f, m
                predict_btn = True

    with col_result:
        if predict_btn:
            segment = predict_segment(recency, frequency, monetary)
            icon, advice = SEGMENT_INFO[segment]

            st.markdown("#### 🏷️ Predicted Segment")
            st.markdown(
                f'<div class="segment-badge {segment}">{icon} {segment}</div>',
                unsafe_allow_html=True,
            )

            st.markdown("---")
            st.markdown("#### 📌 Business Action")
            st.info(advice)

            # Show where this customer sits vs averages
            st.markdown("---")
            st.markdown("#### 📊 Your Values vs Segment Averages")
            seg_avg = rfm_df[rfm_df['Segment'] == segment][['Recency','Frequency','Monetary']].mean()

            comp = pd.DataFrame({
                "Your Input": [recency, frequency, monetary],
                f"{segment} Avg": [seg_avg['Recency'], seg_avg['Frequency'], seg_avg['Monetary']],
            }, index=["Recency (days)", "Frequency (#)", "Monetary (£)"])

            st.dataframe(comp.style.format("{:.1f}"), use_container_width=True)

            st.markdown("---")
            st.markdown("#### 🗂️ All Segment Profiles")
            profile = rfm_df.groupby('Segment')[['Recency','Frequency','Monetary']].mean().round(1)
            st.dataframe(profile, use_container_width=True)
