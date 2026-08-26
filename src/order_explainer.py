"""
SupplyGuard — Stage 6: Per-Order Risk Driver Explanation
==============================================================
Provides `explain_order()`: given a single order-item's raw feature row
and a trained model, returns a ranked, human-readable list of the top
factors pushing that specific order's predicted risk up or down.

METHODOLOGY (SHAP substitute, documented per Stage 6 note): true Shapley
values require the `shap` package, unavailable in this offline sandbox.
Instead we use a defensible heuristic that combines two ingredients we
already computed in stage6_explainability.py:

  1. GLOBAL IMPORTANCE (permutation importance) — how much each feature
     matters to the model overall.
  2. LOCAL DEVIATION — how unusual this order's value is relative to the
     training population for that feature (categorical: is this
     category the "high-risk" one per Stage 3 EDA rates; numeric:
     z-score vs. training mean/std).

For each feature we compute: contribution_proxy = global_importance x
sign(local deviation towards higher risk). We rank by |contribution_proxy|
and report the top drivers with a plain-language direction.

This is a heuristic, not exact Shapley decomposition — it will be
labeled as such wherever it's surfaced (dashboard, RAG layer), and can be
swapped for `shap.TreeExplainer` on a machine with internet access.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pickle
import numpy as np
import pandas as pd
from feature_engineering import split_keys_features_target, risk_level as compute_risk_level

try:
    import config
    MODEL_DIR = config.MODEL_DIR
    DATA_PATH = config.CLEAN_DATA_PATH
except ImportError:
    # config.py is a pipeline-development convenience not shipped with
    # the Streamlit app bundle (which only needs the OrderExplainer
    # class below, not the demo()/CLI path at the bottom of this file
    # that uses these constants).
    config = None
    MODEL_DIR = None
    DATA_PATH = None


def official_split(df):
    dates_sorted = df["order date (DateOrders)"].sort_values().reset_index(drop=True)
    n = len(dates_sorted)
    cut1 = dates_sorted.iloc[int(n * 0.70)]
    cut2 = dates_sorted.iloc[int(n * 0.85)]
    train_mask = df["order date (DateOrders)"] < cut1
    val_mask = (df["order date (DateOrders)"] >= cut1) & (df["order date (DateOrders)"] < cut2)
    test_mask = df["order date (DateOrders)"] >= cut2
    return train_mask, val_mask, test_mask


class OrderExplainer:
    """Fit on training data + a permutation-importance Series; call
    .explain(row) for a per-order driver breakdown."""

    def __init__(self, feature_engineer, perm_importance: pd.Series,
                 X_train_raw: pd.DataFrame, X_train_encoded: pd.DataFrame,
                 y_train: pd.Series):
        self.fe = feature_engineer
        self.perm_importance = perm_importance
        self.X_train_raw = X_train_raw
        self.X_train_encoded = X_train_encoded
        self.y_train = y_train

        # Precompute training means/stds for numeric encoded columns
        self.train_mean = X_train_encoded.mean()
        self.train_std = X_train_encoded.std().replace(0, 1)

        # Precompute DIRECTION of each feature's relationship with the
        # target (correlation sign) so we know whether an above-average
        # value pushes risk up or down — permutation importance alone is
        # an unsigned magnitude and cannot tell us this.
        combined = X_train_encoded.copy()
        combined["_y"] = y_train.values
        self.direction = combined.corr()["_y"].drop("_y").apply(
            lambda c: 1 if c >= 0 else -1
        )

        # Precompute per-category risk rates for key raw categorical
        # columns, so we can phrase explanations in business terms
        # (e.g. "Shipping Mode = First Class has a 95% historical late
        # rate vs 55% average").
        self.category_risk_rates = {}
        for col in ["Shipping Mode", "Order Status", "Market", "Customer Segment",
                    "Order Region", "Category Name"]:
            if col in X_train_raw.columns:
                self.category_risk_rates[col] = (
                    pd.concat([X_train_raw[col], y_train.rename("y")], axis=1)
                    .groupby(col)["y"].mean()
                )
        self.overall_rate = y_train.mean()

    @classmethod
    def from_state(cls, feature_engineer, state: dict):
        """Reconstruct an OrderExplainer from a precomputed state dict
        (see precompute_app_bundle.py) without needing the original
        training dataframes — used by the Streamlit app for fast startup."""
        obj = cls.__new__(cls)
        obj.fe = feature_engineer
        obj.perm_importance = state["perm_importance"]
        obj.train_mean = state["train_mean"]
        obj.train_std = state["train_std"]
        obj.direction = state["direction"]
        obj.category_risk_rates = state["category_risk_rates"]
        obj.overall_rate = state["overall_rate"]
        obj.X_train_raw = None
        obj.X_train_encoded = None
        obj.y_train = None
        return obj

    def explain(self, raw_row: pd.DataFrame, model, top_n: int = 5) -> dict:
        """raw_row: single-row DataFrame in the same raw (pre-encoding)
        format as X_train_raw. Returns dict with predicted probability,
        risk level, and ranked driver explanations."""
        encoded_row = self.fe.transform(raw_row)
        proba = model.predict_proba(encoded_row)[0, 1]
        level = compute_risk_level(proba)

        contributions = []
        common_features = [f for f in self.perm_importance.index if f in encoded_row.columns]
        for feat in common_features:
            global_imp = self.perm_importance[feat]
            if global_imp <= 0:
                continue
            val = encoded_row[feat].iloc[0]
            z = (val - self.train_mean[feat]) / self.train_std[feat]
            direction_sign = self.direction.get(feat, 1)
            proxy = global_imp * direction_sign * z
            contributions.append((feat, proxy, val))

        contributions.sort(key=lambda x: -abs(x[1]))
        top = contributions[:top_n]

        explanations = []
        for feat, proxy, val in top:
            direction = "increases" if proxy > 0 else "decreases"
            explanations.append(self._phrase(feat, val, direction, raw_row))

        return {
            "predicted_probability": round(float(proba), 4),
            "risk_level": level,
            "top_drivers": explanations,
        }

    def _phrase(self, feat: str, val, direction: str, raw_row: pd.DataFrame) -> str:
        # Try to map one-hot columns back to a business-readable category
        # statement, e.g. "Shipping Mode_First Class" -> "Shipping Mode is
        # First Class (historical late rate 95.3% vs 54.9% average)"
        for col, rates in self.category_risk_rates.items():
            prefix = f"{col}_"
            if feat.startswith(prefix) and val == 1:
                cat_value = feat[len(prefix):]
                if cat_value in rates.index:
                    rate = rates[cat_value]
                    return (f"{col} = '{cat_value}' — historical late-risk rate "
                            f"{rate*100:.1f}% vs {self.overall_rate*100:.1f}% average "
                            f"({direction} predicted risk)")
        if feat == "Was_Canceled":
            label = "This order was canceled/suspected-fraud" if val == 1 else "Order was not canceled"
            return f"{label} ({direction} predicted risk)"
        if feat.endswith("_freq"):
            return f"{feat.replace('_freq','')} frequency in training data = {int(val)} ({direction} predicted risk)"
        return f"{feat} = {val:.3g} ({direction} predicted risk)"


def demo():
    """Builds an OrderExplainer and demonstrates it on 3 example orders
    from the test set — one high-risk, one low-risk, one borderline."""
    with open(f"{MODEL_DIR}/models.pkl", "rb") as f:
        saved = pickle.load(f)
    models = saved["models"]
    fe_tree = saved["fe_tree"]

    df = pd.read_csv(DATA_PATH, parse_dates=["order date (DateOrders)"])
    train_mask, val_mask, test_mask = official_split(df)
    keys, X_raw, y = split_keys_features_target(df)

    X_train_raw = X_raw[train_mask].reset_index(drop=True)
    y_train = y[train_mask].reset_index(drop=True)
    Xtr = fe_tree.transform(X_train_raw)

    perm_imp_path = (os.path.join(config.OUTPUTS_DIR, "perm_importance.csv") if config is not None
                      else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "perm_importance.csv"))
    perm_imp = pd.read_csv(perm_imp_path, index_col=0)
    perm_imp_gb = perm_imp["Gradient Boosting (XGB-substitute)"]

    explainer = OrderExplainer(fe_tree, perm_imp_gb, X_train_raw, Xtr, y_train)
    gb_model = models["Gradient Boosting (XGB-substitute)"][0]

    X_test_raw = X_raw[test_mask].reset_index(drop=True)
    test_encoded = fe_tree.transform(X_test_raw)
    test_proba = gb_model.predict_proba(test_encoded)[:, 1]

    # pick 3 illustrative examples: highest risk, lowest risk, near-median
    idx_high = int(np.argmax(test_proba))
    idx_low = int(np.argmin(test_proba))
    idx_mid = int(np.argsort(np.abs(test_proba - np.median(test_proba)))[0])

    print("="*70)
    print("DEMO: Per-Order Explanations (Gradient Boosting)")
    print("="*70)
    for label, idx in [("HIGHEST RISK example", idx_high),
                        ("LOWEST RISK example", idx_low),
                        ("MEDIAN RISK example", idx_mid)]:
        row = X_test_raw.iloc[[idx]]
        result = explainer.explain(row, gb_model, top_n=5)
        print(f"\n--- {label} (test row {idx}) ---")
        print(f"Predicted late-risk probability: {result['predicted_probability']*100:.1f}%")
        print(f"Risk level: {result['risk_level']}")
        print("Top drivers:")
        for d in result["top_drivers"]:
            print(f"   - {d}")


if __name__ == "__main__":
    demo()
