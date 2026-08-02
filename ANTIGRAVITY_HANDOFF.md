# Drishti — Handoff Notes (for continuation in Google Antigravity)

Last updated: 2026-08-02
Repo: https://github.com/Madhan310301/Drishti (collaborator: uchihadgoku)
Local working copy: D:\New folder\Drishty

WHAT THIS FILE IS
-----------------
Read this BEFORE touching the code. It captures what is actually true in this
repo right now, including things the older docs (README / team manual) get
wrong. The README has been corrected, but the manual PDF is stale.

PROJECT IN ONE PARAGRAPH
------------------------
Drishti is a Predictive Command Console for the Karnataka Police Datathon 2026.
It predicts crime hotspots (DBSCAN + Isolation Forest) over a supervised risk
proxy, explains them with SHAP, and turns the risk surface into optimal patrol
deployments using a PuLP integer program. There is NO Streamlit app — the
dashboard is a custom HTML/JS console (app/index.html) served by the FastAPI
backend.

THE BIG THING THAT WAS WRONG (now fixed)
----------------------------------------
The README used to say "streamlit run app/app.py". That file was deleted in
commit 1aad344 ("feat: fully-static dashboard (no backend) for static site
deploy"). Following those instructions would fail. Corrections made:
  * README install steps now launch only the FastAPI backend, which also
    serves the dashboard at http://127.0.0.1:8000/ .
  * README stack + architecture diagram updated (HTML/Leaflet/vis-network,
    not Streamlit).
  * One stale test (test_streamlit_app_imports_and_exposes_solve_patrol) that
    did `import app.app` was replaced with
    test_dashboard_backend_exposes_patrol_optimizer.
Full test suite is now GREEN: 22 passed.

HOW TO RUN (single command)
---------------------------
    python -m venv venv && venv\Scripts\activate
    pip install -r requirements.txt
    # Optional — only for the live DB-backed API: create a .env (see below)
    python -m uvicorn backend.api.main:app --host 127.0.0.1 --port 8000
    # Open http://127.0.0.1:8000/  (dashboard + API in one process)

Fresh-clone note: backend/api/main.py auto-generates the ML artifacts
(hotspots, maps, network graph, SHAP JSON) on startup if missing, so a clean
checkout works with zero manual pipeline steps.

ENVIRONMENT / SECRETS
---------------------
  * .env is gitignored and NOT in the repo. It holds Supabase credentials.
    A fresh Antigravity checkout will NOT have it. Copy it from the team
    (or recreate it) if you need the live DB-backed API. Without it, the
    static dashboard + generated artifacts still work; only the live DB
    connection won't.
  * Do NOT commit .env. It already matches .gitignore, but double-check before
    any git add . in a new tool.

DEPLOY TARGETS (important context)
----------------------------------
Two deploy paths exist; do not confuse them:

1. Onslate static site (*.onslate.in) — STATIC ONLY.
   * Full project zip FAILS (Onslate auto-detects Python via requirements.txt
     / Procfile and refuses it).
   * Deploy Drishti-static-deploy.zip (lives in D:\New folder, NOT in git):
       - contains only app/ (index.html at root, static-data/, static-viz/)
       - Root Path = '.' when configuring Onslate.
   * Onslate sets X-Frame-Options: DENY on every response and MIME-blocks
     external .js/.css (serves them as application/json). That is why the
     dashboard uses INLINED libraries (Leaflet, vis-network, jQuery embedded
     in the HTML) and renders maps via inline srcdoc iframes. If you add a new
     map/chart dependency, vendor it locally and inline it — do not add a CDN
     <script src=...> or an external local .js file.
   * The old drishtiapp.onslate.in backend is DEAD.

2. Zoho Catalyst (backend) — Drishti-catalyst-deploy.zip + catalyst.json.
   * Python runtime, uvicorn entrypoint (see Procfile / catalyst.json).
   * This path runs the live API (needs .env / Supabase).

PROJECT LAYOUT (real)
---------------------
    backend/
      api/        FastAPI app (main.py) + routes.py + schemas.py
                  - / serves app/index.html ; /patrol/optimize ; data/SHAP/maps
      ml/         hotspots.py, anomalies.py, explainability.py,
                  patrol_optimizer.py (PatrolOptimizer.optimize), network_graph.py,
                  hotspot_map.py, pipeline.py
      etl/        config.py (paths + artifact file locations)
      database/   connection.py (init_db) — Supabase/SQLAlchemy
      common/     logger, constants
      models/     data models
      utils/
    app/
      index.html            # the dashboard (custom HTML/JS)
      static-data/*.json    # pre-built data consumed by the dashboard
      static-viz/           # inlined map libs + generated hotspot_map.html,
                            # offender_network.html
    data/
      raw/        karnataka_crime_2022.csv (REAL KSP 2022 totals),
                  karnataka_socio_economic.csv
      processed/  hotspot_centers.csv, grid_with_anomalies.csv
      output/     generated artifacts (gitignored)
    tests/        test_features.py, test_manual_checklists.py
    DATA_CONTRACTS.md       # team file/column-name agreement — DO NOT rename
                            # columns without team agreement
    requirements.txt, Procfile, catalyst.json, README.md, .env(gitignored)

KEY CONTRACTS (read DATA_CONTRACTS.md)
--------------------------------------
  * solve_patrol / PatrolOptimizer.optimize returns:
      {deployed, covered_pct, uncovered_count, total_hotspots, ...}
  * Shared column names are frozen: crime points (latitude, longitude,
    crime_type, severity, district); hotspots (center_lat, center_lon,
    risk_score); socio-economic district_name must match crime `district`.

TESTING
-------
    pytest            # 22 passed (verified 2026-08-02)
  Warnings are benign (PuLP/CBC and Starlette/TestClient deprecations).

COMMIT / PUSH WORKFLOW (team convention)
----------------------------------------
  * The agent (Hermes) may do `git add` + `git commit` locally.
  * PUSHING is done by the human (uchihadgoku) in their own terminal — their
    2FA/credentials live there. The agent does not handle the token/password.
  * gh is logged in (keyring) on this machine, but default is: agent commits,
    human pushes.

KNOWN GAPS / WHERE TO CONTINUE
------------------------------
  * Dashboard is fully static (data baked into app/static-data/*.json). If a
    judge wants a LIVE connected demo, that requires the Catalyst backend +
    .env — wire app/index.html to call /patrol/optimize and the data endpoints
    instead of reading the static JSON.
  * No facial recognition (by design — manual forbids it).
  * Maps are offline/self-contained (no tile server) to survive Onslate's
    X-Frame-Options + MIME blocking. Keep it that way for the static deploy.
  * If continuing in Antigravity: open D:\New folder\Drishty as the project
    root. The agent harness can run `python -m uvicorn ...` and `pytest`
    directly. Use the GitHub MCP (or the existing gh CLI) for PRs — push is
    manual per the workflow above.
