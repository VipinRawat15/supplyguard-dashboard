# SupplyGuard Dashboard — Local Setup & Run Instructions

This is the Stage 10 deliverable: an interactive Streamlit dashboard
integrating every prior stage (models, explainability, risk ranking,
RAG-based recommendations) into one app.

## Important — Read Before Running

**This app was built and validated in an offline sandbox with no
internet access.** I could not `pip install streamlit` or launch it
there, so it has **not been visually tested end-to-end in a browser**.
It has been syntax-checked (`python -m py_compile app.py` passes) and
every underlying component (models, `FeatureEngineer`, `OrderExplainer`,
`rag_engine`) has been independently tested and validated in earlier
stages — this app is a thin UI layer wiring those already-working pieces
together. Still, please treat first-run debugging as a real possibility,
not a sign something is fundamentally wrong — Streamlit apps commonly
need one or two small fixes (a version mismatch, a path issue) on first
run in a new environment.

## 1. Setup

```bash
# From this folder:
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

**Scikit-learn version matters.** The trained models in `models/models.pkl`
were pickled with **scikit-learn 1.8.0**. `requirements.txt` pins this
exact version. If you see an unpickling error, confirm your installed
version matches (`python -c "import sklearn; print(sklearn.__version__)"`)
or re-run Stage 5's `stage5_modeling.py` locally to regenerate the
pickle with your installed version.

## 2. Run

```bash
streamlit run app.py
```

This should open the dashboard at `http://localhost:8501` in your
browser automatically. If not, open that URL manually.

## 3. Folder Structure (must be preserved)

```
streamlit_app/
├── app.py
├── requirements.txt
├── README.md
├── stage5_test_results.csv
├── stage8_capture_table_gb.csv
├── stage8_capture_table_rf.csv
├── src/
│   ├── feature_engineering.py
│   ├── order_explainer.py
│   └── rag_engine.py
├── models/
│   ├── models.pkl          (trained models + fitted encoders, ~20MB)
│   └── app_bundle.pkl      (precomputed test-set data + explainer stats, ~7MB)
├── data/
│   └── policies/           (6 policy markdown files for the RAG layer)
└── charts/                 (9 pre-generated PNGs from Stages 3/5/7/8)
```

If you move any of these, update the corresponding path in `app.py`
(all paths are relative to `BASE_DIR = os.path.dirname(__file__)`).

## 4. What Each Page Does

| Page | Contents |
|---|---|
| **Overview** | KPI summary, risk distribution, model performance vs. baseline |
| **Risk Worklist** | Filterable, sortable table of all 27,080 test-set shipments, ranked by predicted risk |
| **Order Detail & Recommendation** | Select any shipment → see probability, risk level, key drivers, and a policy-grounded recommended action (the full RAG pipeline, live) |
| **RAG Decision Assistant** | Free-text question box — ask "Why is this shipment high risk?", "What should the operations team consider?", etc., optionally grounded in a selected shipment. Answers are extractive (built only from retrieved policy text) so the system cannot invent a policy that doesn't exist. |
| **Model Insights** | Permutation importance, ROC curves, confusion matrices, partial dependence plots |
| **Capacity Planning** | Cumulative gains chart + capture-rate table for operational capacity decisions |

## 5. Known Limitations (disclosed for academic transparency)

1. **"Current" shipments are actually the historical test-set window**
   (June 2017 – Jan 2018), since there's no live order feed. The
   dashboard is framed around this test set as a stand-in for "shipments
   currently in the pipeline."
2. **Gradient Boosting = scikit-learn's `HistGradientBoostingClassifier`**,
   not real XGBoost (offline sandbox constraint — see Stage 5 report).
   Swappable for `xgboost.XGBClassifier` with a retrain.
3. **Explainability = permutation importance + partial dependence**, not
   SHAP (same offline constraint — see Stage 7 report). The per-order
   "Key Risk Drivers" panel uses a documented heuristic, not exact
   Shapley values.
4. **RAG retrieval = TF-IDF + cosine similarity**, not dense embeddings
   via ChromaDB (same constraint — see Stage 9 report).
5. **RAG generation = extractive template composition**, not a live LLM
   call — this applies to both the automatic "Recommended Action" panel
   and the "RAG Decision Assistant" free-text Q&A page. Both are
   deliberately extractive (they only assemble text that's actually
   present in the retrieved policy sections), which is why the system
   cannot hallucinate a policy that doesn't exist — if nothing relevant
   is retrieved, it says so rather than guessing. To enable real
   LLM-generated answers instead: `pip install anthropic`, set
   `ANTHROPIC_API_KEY`, and set `USE_LLM = True` at the top of
   `src/rag_engine.py`. No other code changes are needed — both
   `recommend()` and `answer_question()` already have a working
   (currently inactive) LLM call path wired to the same retrieved
   context.

## 6. If Something Breaks

- **Blank page / import error on startup**: confirm you're running
  `streamlit run app.py` from *inside* this folder, not a parent
  directory (relative paths depend on this).
- **`ModuleNotFoundError` for `feature_engineering`/`order_explainer`/
  `rag_engine`**: confirm the `src/` folder is present alongside `app.py`
  — the app adds it to `sys.path` at the top.
- **Slow first load**: `@st.cache_resource`/`@st.cache_data` mean the
  first page load does real work (loading a 20MB model file, scoring
  27,080 rows); subsequent navigation should be fast. If it stays slow,
  check the terminal for errors rather than assuming it's just large data.
