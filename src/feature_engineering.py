"""
SupplyGuard — Stage 4: Feature Engineering Module
======================================================
Defines a reusable FeatureEngineer class using a fit/transform pattern
(scikit-learn style) so that ALL encodings (frequency maps, one-hot
category lists, scaling stats) are learned ONLY from training data and
then applied unchanged to validation/test data. This avoids a subtle but
real leakage path: if you compute category frequencies or scaling
statistics from the FULL dataset (including future/test rows) before
splitting, information about the future distribution leaks into training,
even though no target values are involved.

This module does not read files or make modeling decisions itself — it is
imported and used by Stage 5 (Modeling), where the actual temporal
train/val/test split happens and .fit() is called on the TRAIN partition
only.

============================
FEATURE GROUPS (design decisions from EDA, Stage 3)
============================

DROPPED FEATURES (redundant / not usable as model inputs):
  - 'Days for shipment (scheduled)': 1:1 duplicate of 'Shipping Mode'
     (confirmed via crosstab in Stage 3). We keep 'Shipping Mode' instead
     since it's more interpretable for the business dashboard narrative.
  - ID columns ('Order Id', 'Order Item Id', 'Customer Id', 'Product Card Id',
     'Category Id', 'Department Id'): arbitrary identifiers, not
     predictive; 'Category Id'/'Department Id' are redundant with their
     Name counterparts anyway. Order Id / Order Item Id are retained
     separately as KEY columns (not features) for order-level splitting
     and row tracing, never passed to the model as X.
  - Raw 'Customer Zipcode' / 'Order Zipcode': very high-cardinality
     quasi-identifiers; geography is already captured continuously via
     Latitude/Longitude and categorically via City/State/Country/Region.
     'Has_Order_Zipcode' flag (Stage 2) is kept.
  - 'order date (DateOrders)': kept as a KEY column for temporal
     splitting, not as a raw model feature (already decomposed into
     Order_Year/Month/Day/DayOfWeek/Hour/IsWeekend in Stage 2).

LOW-CARDINALITY CATEGORICALS (<= 25 unique values) -> ONE-HOT ENCODED:
  Type, Customer Country, Customer Segment, Department Name, Market,
  Order Region, Order Status, Shipping Mode, Category Name (50 -> see
  note below)

  NOTE: Category Name has 50 unique values, above the 25 threshold, but
  we one-hot it anyway (rather than frequency-encode) because product
  category is a core, interpretable business dimension that a supply
  chain manager will want to see broken out by name in the dashboard,
  and 50 dummy columns is computationally trivial. This is a deliberate
  exception to the automatic threshold rule, documented here for
  transparency.

HIGH-CARDINALITY CATEGORICALS (> 25 unique values, excluding Category
Name per above) -> FREQUENCY ENCODED (replaced with the count of that
category in the TRAINING data; unseen categories at inference time get 0):
  Customer City, Customer State, Order City, Order Country, Order State,
  Product Name

  Frequency encoding does not use the target variable, so it carries no
  target-leakage risk by construction. We still fit it on train-only data
  to avoid the milder issue of the encoding reflecting a category
  distribution (e.g. which cities are common) that includes future/test
  information not yet available at training time in a real deployment.

NUMERIC FEATURES (used as-is, optionally scaled for linear models):
  Benefit per order, Sales per customer, Order Item Discount,
  Order Item Discount Rate, Order Item Product Price,
  Order Item Profit Ratio, Order Item Quantity, Sales, Latitude, Longitude,
  Order_Year, Order_Month, Order_Day, Order_DayOfWeek, Order_Hour

BINARY FLAGS (used as-is, no encoding needed):
  Was_Canceled, Has_Order_Zipcode, Order_IsWeekend

TARGET:
  Late_delivery_risk

Multicollinearity note (from Stage 3): 'Sales per customer' vs 'Sales'
(r=0.99) and 'Benefit per order' vs 'Order Item Profit Ratio' (r=0.82) are
both kept in the DEFAULT feature set below since tree-based models (our
primary candidates) are not harmed by correlated features. A
`drop_collinear=True` option is provided for use with Logistic Regression
in Stage 5, which drops 'Sales' and 'Order Item Profit Ratio' (keeping the
more business-interpretable 'Sales per customer' and 'Benefit per order').
"""

import pandas as pd
import numpy as np


KEY_COLS = ["Order Id", "Order Item Id", "order date (DateOrders)"]

TARGET_COL = "Late_delivery_risk"

DROP_COLS = [
    "Days for shipment (scheduled)",
    "Customer Id",
    "Product Card Id",
    "Category Id",
    "Department Id",
    "Customer Zipcode",
    "Order Zipcode",
]

# ----------------------------------------------------------------------
# Risk level thresholds (shared across order_explainer.py,
# stage8_risk_prioritization.py, and the Streamlit app — previously
# duplicated in three places; consolidated here as the single source of
# truth, found during the final QA audit).
#
# IMPORTANT — THESE ARE BUSINESS-DEFINED HEURISTIC CUTOFFS, NOT
# STATISTICALLY OPTIMIZED THRESHOLDS. They were chosen as simple, legible
# round numbers (35% / 55% / 75%) for interpretability, not derived from
# a cost-sensitive optimization over the false-positive/false-negative
# tradeoff. An organization with known intervention costs (e.g. cost of
# manually reviewing a shipment vs. cost of an undetected late delivery)
# should re-derive these cutoffs via expected-value analysis against the
# precision-recall curve rather than treat them as fixed or validated.
# ----------------------------------------------------------------------
RISK_THRESHOLDS = {"very_high": 0.75, "high": 0.55, "moderate": 0.35}


def risk_level(p: float) -> str:
    """Maps a predicted probability to a business-facing risk label using
    RISK_THRESHOLDS. See the module-level note above on threshold
    provenance."""
    if p >= RISK_THRESHOLDS["very_high"]:
        return "Very High"
    elif p >= RISK_THRESHOLDS["high"]:
        return "High"
    elif p >= RISK_THRESHOLDS["moderate"]:
        return "Moderate"
    return "Low"

ONEHOT_COLS = [
    "Type",
    "Customer Country",
    "Customer Segment",
    "Department Name",
    "Market",
    "Order Region",
    "Order Status",
    "Shipping Mode",
    "Category Name",
]

FREQ_ENCODE_COLS = [
    "Customer City",
    "Customer State",
    "Order City",
    "Order Country",
    "Order State",
    "Product Name",
]

NUMERIC_COLS = [
    "Benefit per order",
    "Sales per customer",
    "Order Item Discount",
    "Order Item Discount Rate",
    "Order Item Product Price",
    "Order Item Profit Ratio",
    "Order Item Quantity",
    "Sales",
    "Latitude",
    "Longitude",
    "Order_Year",
    "Order_Month",
    "Order_Day",
    "Order_DayOfWeek",
    "Order_Hour",
]

BINARY_COLS = [
    "Was_Canceled",
    "Has_Order_Zipcode",
    "Order_IsWeekend",
]

# Collinear pairs identified in Stage 3 EDA — dropped only when
# drop_collinear=True is passed to FeatureEngineer (intended for the
# Logistic Regression baseline in Stage 5).
COLLINEAR_DROP_FOR_LINEAR = ["Sales", "Order Item Profit Ratio"]


class FeatureEngineer:
    """Fit/transform-style feature engineering pipeline.

    Usage:
        fe = FeatureEngineer(drop_collinear=False)
        fe.fit(train_df)
        X_train = fe.transform(train_df)
        X_val   = fe.transform(val_df)
        X_test  = fe.transform(test_df)

    IMPORTANT: always call .fit() on the TRAINING partition only.
    """

    def __init__(self, drop_collinear: bool = False):
        self.drop_collinear = drop_collinear
        self.freq_maps_ = {}
        self.onehot_categories_ = {}
        self.feature_names_ = None
        self.is_fitted_ = False

    def fit(self, df: pd.DataFrame):
        df = df.copy()

        # Learn frequency maps from training data only
        for col in FREQ_ENCODE_COLS:
            self.freq_maps_[col] = df[col].value_counts().to_dict()

        # Learn the fixed set of one-hot categories from training data only.
        # Any category seen only in val/test at transform time will map to
        # all-zero dummy columns (i.e. treated as "unseen category"), which
        # is the correct, leakage-safe behavior.
        for col in ONEHOT_COLS:
            self.onehot_categories_[col] = sorted(df[col].dropna().unique().tolist())

        # Run a transform once to lock in the final feature column order
        transformed = self._build(df)
        self.feature_names_ = transformed.columns.tolist()
        self.is_fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("FeatureEngineer.fit() must be called before .transform()")
        transformed = self._build(df)
        # Ensure identical column set/order to what was learned at fit time
        # (handles unseen categories -> missing dummy columns -> fill 0)
        transformed = transformed.reindex(columns=self.feature_names_, fill_value=0)
        return transformed

    def _build(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        pieces = []

        # Numeric passthrough
        num_cols = [c for c in NUMERIC_COLS]
        if self.drop_collinear:
            num_cols = [c for c in num_cols if c not in COLLINEAR_DROP_FOR_LINEAR]
        pieces.append(df[num_cols].reset_index(drop=True))

        # Binary passthrough
        pieces.append(df[BINARY_COLS].reset_index(drop=True))

        # One-hot encode using the FIXED category list learned at fit time
        for col in ONEHOT_COLS:
            cats = self.onehot_categories_.get(col)
            if cats is None:
                # fit() not yet called with this exact df (shouldn't happen
                # in normal use, but guard anyway)
                cats = sorted(df[col].dropna().unique().tolist())
            cat_dtype = pd.CategoricalDtype(categories=cats)
            dummies = pd.get_dummies(
                df[col].astype(cat_dtype), prefix=col, dtype=int
            ).reset_index(drop=True)
            pieces.append(dummies)

        # Frequency-encode high-cardinality columns using TRAIN-fit maps
        freq_frame = pd.DataFrame(index=df.index)
        for col in FREQ_ENCODE_COLS:
            fmap = self.freq_maps_.get(col, {})
            freq_frame[f"{col}_freq"] = df[col].map(fmap).fillna(0).astype(int)
        pieces.append(freq_frame.reset_index(drop=True))

        result = pd.concat(pieces, axis=1)
        return result


def split_keys_features_target(df: pd.DataFrame):
    """Utility: split a Stage-2-cleaned dataframe into
    (keys_df, feature_input_df, target_series) BEFORE feature engineering.
    feature_input_df still contains raw categorical/numeric columns ready
    to be passed into FeatureEngineer.fit()/.transform().
    """
    keys = df[KEY_COLS].copy()
    y = df[TARGET_COL].copy()
    X_raw = df.drop(columns=DROP_COLS + KEY_COLS + [TARGET_COL], errors="ignore")
    return keys, X_raw, y
