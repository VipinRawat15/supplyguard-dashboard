# Deploying SupplyGuard to Streamlit Community Cloud

Your professor's brief asks for a live-hosted app + LinkedIn post. This
project is a Streamlit app, so it maps directly to the recommended path:
**Python Data App → Streamlit → Streamlit Community Cloud.**

This guide is written specifically for this repo's structure — not a
generic tutorial — and assumes you're starting from the
`streamlit_app/` folder you already have.

Total time: ~15-20 minutes, no cost.

---

## Before you start — quick sanity check

This app has been extensively logic-tested but **never visually run in
a real browser** (it was built in an offline sandbox with no internet
access — see `README.md`, Section "Important — Read Before Running").
**Strongly recommended: run it locally first** (`streamlit run app.py`
per the README) before deploying, so you're debugging on your own
machine with full error visibility rather than discovering a problem for
the first time on a live public URL. Budget for a possible small fix —
this is normal for a first Streamlit run in a new environment, not a
sign of a broken project.

---

## Step 1 — Create a GitHub repository

1. Go to [github.com/new](https://github.com/new).
2. Name it something like `supplyguard-dashboard`.
3. **Public** — your professor wants this discoverable/tagged for
   employers, and Streamlit Community Cloud's free tier only allows
   **one private app** anyway, so public is the practical choice here.
4. Don't initialize with a README (you already have one) — create it
   empty.

## Step 2 — Push this project to the repo

**Important**: push the *contents* of `streamlit_app/` as the repo
root — not `streamlit_app/` as a subfolder — so `app.py` and
`requirements.txt` sit at the top level. This keeps Streamlit Cloud's
default configuration simple (no custom "main file path" needed beyond
`app.py`).

```bash
cd streamlit_app
git init
git add .
git commit -m "Initial commit: SupplyGuard dashboard"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/supplyguard-dashboard.git
git push -u origin main
```

A `.gitignore` is already included (excludes `__pycache__`, local
`.streamlit/secrets.toml`, virtual envs).

**Sanity check before pushing**: this repo intentionally contains no API
keys or secrets — verified during the project's final QA audit
(see `PROJECT_REPORT.md`, Section C). Safe to make public as-is.

## Step 3 — Deploy on Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in
   with your GitHub account (this authorizes Streamlit to read your
   repos).
2. Click **"New app"** → **"Use existing repo"**.
3. Fill in:
   - **Repository**: `YOUR-USERNAME/supplyguard-dashboard`
   - **Branch**: `main`
   - **Main file path**: `app.py`
4. Click **"Deploy!"**

Streamlit Cloud will now `pip install` everything in `requirements.txt`
and launch the app. First build typically takes 2-5 minutes (this app's
dependencies — pandas, numpy, scikit-learn, streamlit — are all common
and fast to install; no heavy ML frameworks needed since we use
scikit-learn's built-in gradient boosting rather than a separate
XGBoost package).

You'll get a URL like `https://supplyguard-dashboard.streamlit.app`.

## Step 4 — Verify it actually works

Since this was never browser-tested before deployment, **actually click
through it** rather than assuming it works because it built
successfully:

- [ ] **Overview** page loads with KPI numbers (not blank/error)
- [ ] **Risk Worklist** page: filters respond, table populates
- [ ] **Order Detail & Recommendation**: pick a shipment, confirm
      probability/risk level/drivers/recommendation all render
- [ ] **RAG Decision Assistant**: type a question (or use a quick-pick),
      click "Ask", confirm an answer with sources appears
- [ ] **Model Insights** and **Capacity Planning**: charts display

If something breaks, check the app's logs (visible to you as the repo
owner, via the "Manage app" menu in the bottom-right of your running
app) — they'll show the actual Python traceback, which is far more
useful for fixing it than guessing.

## Step 5 (optional) — Enable live LLM recommendations

By default the app uses an extractive template composer, not a live LLM
call (see `README.md` Section 5). To switch on real Claude-generated
recommendations:

1. Get an Anthropic API key from [console.anthropic.com](https://console.anthropic.com).
2. In your Streamlit Cloud app settings → **"Secrets"**, add:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   ```
   (Root-level TOML entries are automatically exposed as environment
   variables to your app, which is exactly how the `anthropic` Python
   SDK expects to find the key by default — no code changes needed for
   the key itself.)
3. Add `anthropic` to `requirements.txt`.
4. In `src/rag_engine.py`, change `USE_LLM = False` to `USE_LLM = True`.
5. Commit and push — Streamlit Cloud auto-redeploys on every push to
   `main`.

**Not required** — the app works fully without this step.

## Notes on the free tier (so nothing surprises you)

- **Apps sleep after ~12 hours of no traffic** and need one click to
  wake up on the next visit (shows a "waking up" screen for ~30-60
  seconds). Mention this if you demo it live — the first load after
  sleep will look slow, that's expected, not broken.
- **~1GB memory limit.** This app should comfortably fit — it does
  inference on pre-trained models (20MB), not training, and the test-set
  bundle is 6.6MB.
- **No custom domain** on the free tier — you'll share the
  `*.streamlit.app` URL as-is, which is completely normal for this kind
  of academic/portfolio project.

---

## Step 6 — LinkedIn post

Your professor mentioned they'll tag employers, so keep it professional
and lead with the business problem, not just the tech stack. A rough
structure that tends to work well for this kind of post:

1. One line on the business problem (late deliveries hurt operations).
2. One line on what you built (early-warning ML system + explainability
   + RAG-based recommendations + live dashboard).
3. 2-3 concrete numbers (e.g., "catches ~35% of late deliveries by
   reviewing just the top 20% of shipments, at 97% precision").
4. The live link.
5. Tag your professor/university if appropriate, per their instructions.

Happy to draft the actual post text with you if you want — just ask,
and let me know if you want it technical (for a data-science audience)
or business-framed (for ops/supply-chain hiring managers).
