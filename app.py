"""
SupplyGuard — AI-Powered Delivery Risk Intelligence Dashboard
===================================================================
Stage 10: Streamlit integration of all prior stages.

Run with:
    streamlit run app.py

Expects this directory layout (see README.md):
    app.py
    src/  (feature_engineering.py, order_explainer.py, rag_engine.py)
    models/  (models.pkl, app_bundle.pkl)
    data/policies/  (*.md policy documents)
    charts/  (pre-generated PNGs from Stages 3, 5, 7, 8)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import pickle
import numpy as np
import pandas as pd
import streamlit as st

from feature_engineering import risk_level
from order_explainer import OrderExplainer
from rag_engine import PolicyRetriever, recommend, answer_question

# ----------------------------------------------------------------------
# Page config & style
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="SupplyGuard — Delivery Risk Intelligence",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

PALETTE = {
    "teal": "#2A9D8F",
    "red": "#E63946",
    "blue": "#457B9D",
    "orange": "#F4A261",
    "yellow": "#E9C46A",
    "dark": "#264653",
}

RISK_COLORS = {
    "Low": PALETTE["teal"],
    "Moderate": PALETTE["yellow"],
    "High": PALETTE["orange"],
    "Very High": PALETTE["red"],
}

st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        border-left: 4px solid #457B9D;
    }
    div[data-testid="stMetricValue"] { font-size: 1.8rem; }
    .risk-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 14px;
        color: white;
        font-weight: 600;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(__file__)


# ----------------------------------------------------------------------
# Cached data / model loading
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading models...")
def load_models():
    with open(os.path.join(BASE_DIR, "models", "models.pkl"), "rb") as f:
        saved = pickle.load(f)
    return saved


@st.cache_resource(show_spinner="Loading app data bundle...")
def load_app_bundle():
    with open(os.path.join(BASE_DIR, "models", "app_bundle.pkl"), "rb") as f:
        bundle = pickle.load(f)
    return bundle


@st.cache_resource(show_spinner="Loading policy knowledge base...")
def load_retriever():
    return PolicyRetriever(policy_dir=os.path.join(BASE_DIR, "data", "policies"))


@st.cache_resource(show_spinner="Building risk explainer...")
def load_explainer(_fe_tree, _state):
    return OrderExplainer.from_state(_fe_tree, _state)


@st.cache_data(show_spinner="Scoring all test-set shipments...")
def build_worklist(_models, _fe_tree, X_test_raw, keys_test, y_test):
    gb_model = _models["Gradient Boosting (XGB-substitute)"][0]
    rf_model = _models["Random Forest"][0]
    Xte = _fe_tree.transform(X_test_raw)
    proba_gb = gb_model.predict_proba(Xte)[:, 1]
    proba_rf = rf_model.predict_proba(Xte)[:, 1]

    # risk_level() is imported from feature_engineering — single shared
    # source of truth for the business-defined risk thresholds (0.35 /
    # 0.55 / 0.75). See feature_engineering.py for the disclosure note
    # that these are interpretability-driven cutoffs, not statistically
    # validated against a cost function.

    worklist = keys_test.copy()
    # Carry the original X_test_raw row position through the sort
    # explicitly, so it can never drift out of sync with a separately
    # recomputed sort order (e.g. from tie-breaking differences between
    # pandas.sort_values and numpy.argsort on probability ties, which do
    # occur in this data — verified during development).
    worklist["_orig_idx"] = np.arange(len(worklist))
    worklist["Shipping Mode"] = X_test_raw["Shipping Mode"].values
    worklist["Order Status"] = X_test_raw["Order Status"].values
    worklist["Market"] = X_test_raw["Market"].values
    worklist["Order Region"] = X_test_raw["Order Region"].values
    worklist["Customer Segment"] = X_test_raw["Customer Segment"].values
    worklist["Actual_Late_Risk"] = y_test.values
    worklist["Predicted_Probability"] = proba_gb.round(4)
    worklist["Predicted_Probability_RF"] = proba_rf.round(4)
    worklist["Risk_Level"] = [risk_level(p) for p in proba_gb]
    worklist = worklist.sort_values("Predicted_Probability", ascending=False).reset_index(drop=True)
    worklist["Priority_Rank"] = np.arange(1, len(worklist) + 1)
    return worklist


models_saved = load_models()
fe_tree = models_saved["fe_tree"]
bundle = load_app_bundle()
X_test_raw = bundle["X_test_raw"]
keys_test = bundle["keys_test"]
y_test = bundle["y_test"]
retriever = load_retriever()
explainer = load_explainer(fe_tree, bundle["explainer_state"])

worklist = build_worklist(models_saved["models"], fe_tree, X_test_raw, keys_test, y_test)
_gb_model = models_saved["models"]["Gradient Boosting (XGB-substitute)"][0]


# ----------------------------------------------------------------------
# Sidebar navigation
# ----------------------------------------------------------------------
st.sidebar.title("📦 SupplyGuard")
st.sidebar.caption("AI-Powered Delivery Risk Intelligence")
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Risk Worklist", "Order Detail & Recommendation",
     "RAG Decision Assistant", "Model Insights", "Capacity Planning"],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Data note**: this dashboard operates on the held-out chronological "
    "test set (27,080 shipments, June 2017 – Jan 2018) as a stand-in for "
    "\"current\" shipments, since the underlying dataset is historical."
)
st.sidebar.markdown(
    "**Model substitution note**: Gradient Boosting here uses scikit-learn's "
    "`HistGradientBoostingClassifier` in place of XGBoost, and explainability "
    "uses permutation importance in place of SHAP — both due to an offline "
    "training environment. See project documentation for details."
)


# ----------------------------------------------------------------------
# PAGE: Overview
# ----------------------------------------------------------------------
if page == "Overview":
    st.title("SupplyGuard — Delivery Risk Intelligence")
    st.markdown(
        "Early-warning decision support for proactive supply chain management. "
        "Predicts late-delivery risk, explains key drivers, and recommends "
        "grounded operational actions."
    )

    col1, col2, col3, col4 = st.columns(4)
    n_total = len(worklist)
    n_late_actual = int(worklist["Actual_Late_Risk"].sum())
    n_very_high = int((worklist["Risk_Level"] == "Very High").sum())
    avg_proba = worklist["Predicted_Probability"].mean()

    col1.metric("Shipments Monitored", f"{n_total:,}")
    col2.metric("Actual Late Rate (test set)", f"{n_late_actual/n_total*100:.1f}%")
    col3.metric("Very High Risk Shipments", f"{n_very_high:,}",
                f"{n_very_high/n_total*100:.1f}% of total")
    col4.metric("Avg Predicted Risk", f"{avg_proba*100:.1f}%")

    st.markdown("---")

    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.subheader("Risk Level Distribution")
        counts = worklist["Risk_Level"].value_counts().reindex(
            ["Low", "Moderate", "High", "Very High"]).fillna(0)
        chart_df = pd.DataFrame({"Risk Level": counts.index, "Count": counts.values})
        st.bar_chart(chart_df.set_index("Risk Level"), color=PALETTE["blue"])

    with col_b:
        st.subheader("Model Performance vs. Business Baseline")
        results_path = os.path.join(BASE_DIR, "stage5_test_results.csv")
        if os.path.exists(results_path):
            results_df = pd.read_csv(results_path, index_col=0)
            st.dataframe(
                results_df[["accuracy", "precision", "recall", "f1", "roc_auc"]].round(3),
                use_container_width=True,
            )
        st.caption(
            "All ML models outperform the simple business-rule baseline "
            "(flag First/Second Class as risky) on every metric — see "
            "Stage 5 report for full analysis."
        )

    st.markdown("---")
    st.subheader("Key Business Finding")
    st.info(
        "**Shipping Mode is the dominant late-delivery risk driver.** "
        "First Class shipments have a ~95% historical late-risk rate vs. "
        "~38% for Standard Class — expedited shipping commitments leave "
        "little operational slack. Financial features (price, discount, "
        "profit) are essentially uncorrelated with late-delivery risk, "
        "meaning risk-based prioritization does not systematically favor "
        "or penalize high-value orders."
    )


# ----------------------------------------------------------------------
# PAGE: Risk Worklist
# ----------------------------------------------------------------------
elif page == "Risk Worklist":
    st.title("Risk-Ranked Shipment Worklist")
    st.markdown(
        "Every monitored shipment, ranked by predicted late-delivery risk. "
        "Use this to prioritize limited operational review capacity."
    )

    col1, col2, col3, col4 = st.columns(4)
    risk_filter = col1.multiselect(
        "Risk Level", ["Very High", "High", "Moderate", "Low"],
        default=["Very High", "High"],
    )
    mode_filter = col2.multiselect(
        "Shipping Mode", sorted(worklist["Shipping Mode"].unique()), default=[]
    )
    market_filter = col3.multiselect(
        "Market", sorted(worklist["Market"].unique()), default=[]
    )
    status_filter = col4.multiselect(
        "Order Status", sorted(worklist["Order Status"].unique()), default=[]
    )

    filtered = worklist.copy()
    if risk_filter:
        filtered = filtered[filtered["Risk_Level"].isin(risk_filter)]
    if mode_filter:
        filtered = filtered[filtered["Shipping Mode"].isin(mode_filter)]
    if market_filter:
        filtered = filtered[filtered["Market"].isin(market_filter)]
    if status_filter:
        filtered = filtered[filtered["Order Status"].isin(status_filter)]

    st.caption(f"Showing {len(filtered):,} of {len(worklist):,} shipments")

    display_cols = ["Priority_Rank", "Order Id", "Predicted_Probability",
                     "Risk_Level", "Shipping Mode", "Order Status", "Market",
                     "Customer Segment"]
    st.dataframe(
        filtered[display_cols].head(500).style.format({"Predicted_Probability": "{:.1%}"}),
        use_container_width=True,
        height=500,
    )
    if len(filtered) > 500:
        st.caption("Showing top 500 of filtered results (by priority rank). "
                   "Narrow filters further to see more specific rows, or export "
                   "the full CSV from Stage 8 deliverables for complete data.")


# ----------------------------------------------------------------------
# PAGE: Order Detail & Recommendation
# ----------------------------------------------------------------------
elif page == "Order Detail & Recommendation":
    st.title("Order Detail & Recommended Action")
    st.markdown(
        "Select a shipment to see its predicted risk, key drivers, and a "
        "policy-grounded recommended action."
    )

    quick_pick = st.radio(
        "Quick select", ["Choose from top-20 highest risk", "Search by Order Id"],
        horizontal=True,
    )

    if quick_pick == "Choose from top-20 highest risk":
        top20 = worklist.head(20)
        options = {
            f"#{r.Priority_Rank} — Order {r['Order Id']} — {r.Predicted_Probability:.1%} ({r.Risk_Level})": r["_orig_idx"]
            for _, r in top20.iterrows()
        }
        choice = st.selectbox("Shipment", list(options.keys()))
        selected_idx = options[choice]
    else:
        order_id_input = st.number_input("Order Id", min_value=0, step=1, value=int(worklist.iloc[0]["Order Id"]))
        matches = worklist[worklist["Order Id"] == order_id_input]
        if len(matches) == 0:
            st.warning("No matching Order Id found in the monitored test-set window.")
            st.stop()
        selected_idx = matches.iloc[0]["_orig_idx"]

    row = X_test_raw.iloc[[selected_idx]]
    result = explainer.explain(row, _gb_model, top_n=5)
    proba = result["predicted_probability"]
    risk_level = result["risk_level"]
    shipping_mode = row["Shipping Mode"].iloc[0]
    order_status = row["Order Status"].iloc[0]
    market = row["Market"].iloc[0]
    actual = y_test.iloc[selected_idx]

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(
            f'<span class="risk-badge" style="background-color:{RISK_COLORS[risk_level]}">'
            f'{risk_level} Risk</span>', unsafe_allow_html=True
        )
        st.metric("Predicted Late-Delivery Probability", f"{proba*100:.1f}%")
        st.write(f"**Shipping Mode:** {shipping_mode}")
        st.write(f"**Order Status:** {order_status}")
        st.write(f"**Market:** {market}")
        st.write(f"**Actual outcome (historical data):** "
                 f"{'Late' if actual == 1 else 'Not Late'}")

    with col2:
        st.subheader("Key Risk Drivers")
        st.caption(
            "Heuristic explanation (permutation importance x training-set "
            "deviation) — a documented, offline-capable substitute for SHAP. "
            "See Stage 7 report for methodology."
        )
        for d in result["top_drivers"]:
            st.markdown(f"- {d}")

    st.markdown("---")
    st.subheader("Recommended Action")
    with st.spinner("Retrieving relevant policy guidance..."):
        driver_terms = [d.split(" —")[0].split(" (")[0] for d in result["top_drivers"]]
        rag_result = recommend(
            risk_level=risk_level, proba=proba, shipping_mode=shipping_mode,
            order_status=order_status, market=market, top_drivers=driver_terms,
            retriever=retriever, top_k=3,
        )
    st.markdown(rag_result["recommendation"])

    with st.expander("Show retrieval details"):
        st.write("**Query used:**", rag_result["query"])
        st.write("**Retrieved policy sections:**")
        for sec in rag_result["retrieved_sections"]:
            st.write(f"- [{sec['similarity']:.2f}] {sec['doc_name']} — {sec['section_title']}")


# ----------------------------------------------------------------------
# PAGE: RAG Decision Assistant
# ----------------------------------------------------------------------
elif page == "RAG Decision Assistant":
    st.title("RAG Decision Assistant")
    st.markdown(
        "Ask questions in plain language — e.g. *\"Why is this shipment high "
        "risk?\"*, *\"What should the operations team consider?\"*, *\"What does "
        "the relevant intervention policy recommend?\"* Answers are grounded "
        "in SupplyGuard's policy knowledge base (see retrieved sources below "
        "each answer) — the system will say so rather than guess if nothing "
        "relevant is found."
    )

    use_shipment_context = st.checkbox(
        "Ground my question in a specific shipment", value=True,
        help="If checked, pick a shipment below and your question will be "
             "answered with that shipment's risk profile in mind. If "
             "unchecked, ask a general policy question instead."
    )

    ctx = {}
    if use_shipment_context:
        top20 = worklist.head(20)
        options = {
            f"#{r.Priority_Rank} — Order {r['Order Id']} — {r.Predicted_Probability:.1%} ({r.Risk_Level})": r["_orig_idx"]
            for _, r in top20.iterrows()
        }
        choice = st.selectbox("Shipment to ground the question in", list(options.keys()))
        selected_idx = options[choice]
        row = X_test_raw.iloc[[selected_idx]]
        result = explainer.explain(row, _gb_model, top_n=5)
        ctx = {
            "risk_level": result["risk_level"],
            "proba": result["predicted_probability"],
            "shipping_mode": row["Shipping Mode"].iloc[0],
            "order_status": row["Order Status"].iloc[0],
            "market": row["Market"].iloc[0],
            "top_drivers": [d.split(" —")[0].split(" (")[0] for d in result["top_drivers"]],
        }
        st.caption(
            f"Grounded in: {result['risk_level']} risk, "
            f"{result['predicted_probability']*100:.1f}% predicted probability, "
            f"{ctx['shipping_mode']} shipping, {ctx['order_status']} status."
        )

    st.markdown("---")
    suggested = ["Why is this shipment high risk?", "What should the operations team consider?",
                 "What does the relevant intervention policy recommend?"]
    quick_q = st.radio("Quick questions", ["(type my own)"] + suggested, horizontal=False)
    question = (st.text_input("Your question", placeholder="Type a question about risk, policy, or recommended action...")
                if quick_q == "(type my own)" else quick_q)

    if st.button("Ask", type="primary") and question:
        with st.spinner("Retrieving relevant policy guidance..."):
            qa_result = answer_question(question, retriever, top_k=3, **ctx)
        st.markdown(qa_result["answer"])
        with st.expander("Show retrieval details"):
            st.write("**Retrieval query used:**", qa_result["query"])
            for sec in qa_result["retrieved_sections"]:
                st.write(f"- [{sec['similarity']:.2f}] {sec['doc_name']} — {sec['section_title']}")


# ----------------------------------------------------------------------
# PAGE: Model Insights
# ----------------------------------------------------------------------
elif page == "Model Insights":
    st.title("Model Insights & Explainability")

    tab1, tab2, tab3 = st.tabs(["Feature Importance", "Model Comparison", "Partial Dependence"])

    with tab1:
        st.subheader("Permutation Importance — Top Risk Drivers")
        img_path = os.path.join(BASE_DIR, "charts", "01_permutation_importance.png")
        if os.path.exists(img_path):
            st.image(img_path, use_container_width=True)
        st.caption(
            "Both Random Forest and Gradient Boosting agree on the same top-5 "
            "drivers (Shipping Mode variants, Order_Hour, Was_Canceled) — "
            "convergence across two different algorithms is good evidence "
            "these are genuine signals, not training artifacts."
        )

    with tab2:
        st.subheader("Model Comparison — Test Set")
        img_path = os.path.join(BASE_DIR, "charts", "03_metrics_comparison.png")
        if os.path.exists(img_path):
            st.image(img_path, use_container_width=True)
        img_path2 = os.path.join(BASE_DIR, "charts", "02_roc_curves.png")
        if os.path.exists(img_path2):
            st.image(img_path2, use_container_width=True)

    with tab3:
        st.subheader("Partial Dependence — Direction of Effects")
        img_path = os.path.join(BASE_DIR, "charts", "02_partial_dependence.png")
        if os.path.exists(img_path):
            st.image(img_path, use_container_width=True)
        st.caption(
            "Shows whether each top feature pushes predicted risk up or down. "
            "First Class shipping sharply increases risk; cancellation and "
            "Standard Class shipping decrease it — all consistent with "
            "business intuition."
        )


# ----------------------------------------------------------------------
# PAGE: Capacity Planning
# ----------------------------------------------------------------------
elif page == "Capacity Planning":
    st.title("Operational Capacity Planning")
    st.markdown(
        "How much late-delivery coverage does a given amount of review "
        "capacity buy? Use this to size your operations team's daily "
        "review workload."
    )

    img_path = os.path.join(BASE_DIR, "charts", "01_cumulative_gains.png")
    if os.path.exists(img_path):
        st.image(img_path, use_container_width=True)

    st.subheader("Capture Rate by Capacity Level")
    cap_path = os.path.join(BASE_DIR, "stage8_capture_table_gb.csv")
    if os.path.exists(cap_path):
        cap_df = pd.read_csv(cap_path)
        st.dataframe(
            cap_df.style.format({"precision_at_k": "{:.1%}", "recall_at_k": "{:.1%}"}),
            use_container_width=True,
        )
        st.info(
            "**Reading this table**: if your team can review the top 20% of "
            "shipments by predicted risk each day, you'll catch roughly a "
            "third of all actual late deliveries at ~97% precision — almost "
            "everything reviewed really is high-risk."
        )
