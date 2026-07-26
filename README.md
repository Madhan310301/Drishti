<div align="center">

  <!-- 3D Header Animation / Banner -->
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:09090e,50:0f172a,100:1e1b4b&height=220&section=header&text=DRISHTI%20%E2%9A%A1%EF%B8%8F&fontSize=70&fontColor=00f2fe&animation=fadeIn&fontAlignY=38&desc=PREDICTIVE%20COMMAND%20CONSOLE%20%7C%20KARNATAKA%20POLICE%20DATATHON%202026&descSize=18&descAlignY=62&descAlign=50" width="100%" alt="Drishti Banner"/>

  <p align="center">
    <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=22&pause=1000&color=00F2FE&center=true&vCenter=true&width=700&lines=Beyond+Prediction%3A+Optimizing+Real-World+Patrol+Action;DBSCAN+Hotspot+Clustering+%2B+Isolation+Forest+Surges;SHAP+Explainable+AI+%2B+PuLP+Resource+Simulator;Karnataka+Police+Datathon+2026+Challenge+02" alt="Typing SVG" />
  </p>

  <!-- Badges -->
  <p align="center">
    <a href="https://python.org"><img src="https://img.shields.io/badge/Python-3.11+-00F2FE?style=for-the-badge&logo=python&logoColor=black" alt="Python"/></a>
    <a href="https://streamlit.io"><img src="https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit"/></a>
    <a href="https://scikit-learn.org"><img src="https://img.shields.io/badge/scikit--learn-ML%20Engine-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=black" alt="Scikit-Learn"/></a>
    <a href="https://shap.readthedocs.io"><img src="https://img.shields.io/badge/SHAP-Explainable%20AI-990000?style=for-the-badge" alt="SHAP"/></a>
    <a href="https://coin-or.github.io/pulp/"><img src="https://img.shields.io/badge/PuLP-Patrol%20Optimizer-00599C?style=for-the-badge" alt="PuLP"/></a>
    <a href="https://github.com/Madhan310301/Drishti/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.style=for-the-badge" alt="License"/></a>
  </p>

  <p align="center">
    <b>Karnataka Police Datathon 2026 • Challenge 02 — Drishti (Predictive Command Console)</b>
  </p>

</div>

---

## 🌌 Overview

Most crime analytics tools generate static heatmaps and stop there—leaving command officers with a critical **Real-Time Crime Center (RTCC) gap**: *they predict, but they don't decide.*

**Drishti** bridges this gap. Built for the Karnataka Police Datathon 2026, Drishti is an interactive, explainable, predictive command console that turns crime predictions into **optimized, actionable patrol unit deployments**.

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                  DRISHTI ENGINE FLOW                                    │
│                                                                                         │
│   [Raw Crime Points] ──► [DBSCAN Density Clusters] ──► [Isolation Forest Surge Alerts]  │
│                                                                   │                     │
│   [Real Socio-Econ Stats] ──► [Supervised Risk Proxy] ────────────┤                     │
│                                                                   ▼                     │
│   [Patrol Units (N)] ──► [PuLP Optimization Solver] ──► [SHAP Explainable Risk Score]   │
│                                 │                                                       │
│                                 ▼                                                       │
│                [OPTIMIZED PATROL ROUTES & MAP DEPLOYMENT]                               │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## ⚡ Key Differentiators & Features

| Feature | Description | Technical Implementation |
| :--- | :--- | :--- |
| 🚓 **Patrol Simulator** | *"What-If"* interactive deployment tool. Given $N$ patrol units, computes optimal unit placement maximizing covered crime risk while minimizing response latency. | `PuLP` / `OR-Tools` Integer Linear Programming (Maximal Covering Location Problem) |
| 🧠 **SHAP Explainability** | Every grid risk score ships with a transparent natural language feature attribution breakdown—explaining *why* an area is high risk. | `shap.TreeExplainer` on Gradient Boosted / Random Forest proxy model |
| 📊 **Real Socio-Economic Overlay** | Merges district/ward open government statistics (unemployment, literacy, poverty, liquor density) with crime incident points. | `data.gov.in` + `GeoPandas` spatial aggregation |
| 🌐 **Offender Network Graph** | Interactive visualization of criminal networks, shared MO signatures, and suspect co-offending linkages—without controversial facial recognition. | `Pyvis` + `NetworkX` physics-enabled graph |
| 📍 **Spatial Density & Surges** | Automatic spatial crime cluster identification and spatio-temporal surge anomaly detection. | `DBSCAN` (Haversine metric) + `Isolation Forest` |

---

## 🏗️ System Architecture

```mermaid
graph TD
    A[data.gov.in Socio-Economic Data] -->|Spatial Join| C[Master Feature Matrix]
    B[Synthetic Crime Points] -->|Haversine Density| D[DBSCAN Hotspots]
    B -->|Temporal Velocity| E[Isolation Forest Anomalies]
    
    C --> F[Supervised Proxy Risk Classifier]
    D --> F
    E --> F
    
    F -->|Local Attributions| G[SHAP Risk Explainer]
    D & F --> H[PuLP Patrol Optimizer]
    
    G --> I[Streamlit Command Dashboard]
    H --> I
    J[Suspect Links CSV] -->|NetworkX / Pyvis| K[Interactive Offender Graph]
    K --> I
```

---

## 🛠️ Technology Stack

```
┌─────────────────────────────────────────────────────────────────────────┐
│ CORE STACK                                                              │
│                                                                         │
│   Frontend & Dashboard   : Streamlit · Folium · Plotly Express · Pyvis  │
│   Machine Learning       : Scikit-learn · DBSCAN · Isolation Forest     │
│   Explainable AI         : SHAP (SHapley Additive exPlanations)         │
│   Optimization Engine    : PuLP (CBC Solver) / SciPy / OR-Tools         │
│   Data Engineering       : Pandas · GeoPandas · NumPy · Shapely         │
│   Data Sources           : data.gov.in (Karnataka Open Govt Data)       │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start & Installation

### Prerequisites
* Python 3.10+
* Git

```bash
# 1. Clone the repository
git clone https://github.com/your-username/Drishti.git
cd Drishti

# 2. Create and activate a virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# 3. Install required packages
pip install -r requirements.txt

# 4. Generate synthetic datasets and master grid pipeline
python data/crime_data_generator.py
python data/data_pipeline.py

# 5. Launch the Command Console
streamlit run app/app.py
```

---

## 📂 Project Structure

```
Drishti/
├── data/
│   ├── crime_data_generator.py     # Generates spatio-temporal crime points
│   ├── socio_economic_data.py      # Parses Karnataka data.gov.in statistics
│   ├── data_pipeline.py            # Spatial aggregation & grid cell feature join
│   ├── synthetic_crimes.csv        # Generated crime incidents
│   ├── karnataka_socio_economic.csv# Karnataka district indicators
│   ├── master_grid_features.csv    # Joined spatial matrix
│   ├── offender_nodes.csv          # Criminal network nodes
│   └── offender_edges.csv          # Criminal network co-offending links
├── ml/
│   ├── hotspots.py                 # DBSCAN spatial clustering engine
│   ├── anomalies.py                # Isolation Forest surge anomaly detector
│   ├── explainability.py           # Supervised risk classifier + SHAP explainer
│   └── patrol_optimizer.py         # PuLP patrol deployment optimization solver
├── app/
│   └── app.py                      # Main Streamlit Command Console UI
├── DATA_CONTRACTS.md               # Team API & Data schema contract
├── requirements.txt                # Dependencies
└── README.md
```

---

## 👥 Team Matrix & Roles

| Member | Role | Key Output / Module |
| :--- | :--- | :--- |
| **Madhan** | **Team Leader** | Git Repo Architecture, Integration Testing, Cloud Deployment, Demo Video |
| **Sai Ram** | **Data Engineer** | Synthetic Crime Generator, `data.gov.in` Socio-Econ Overlay, Spatial Join Pipeline |
| **Vijay** | **ML Engineer** | DBSCAN Hotspots (`hotspots.py`), Isolation Forest (`anomalies.py`), SHAP Explainer |
| **Kalyan** | **Optimization Engineer** | PuLP Patrol Simulator Engine (`patrol_optimizer.py`), Coverage Metrics |
| **Jenifa** | **Frontend Developer** | Streamlit Dark Command UI (`app.py`), Folium Maps, SHAP Panels, Pyvis Network Graph |

---

## ⏱️ Submission Deadline

* **Final Hackathon Submission**: `26th July 2026, 08:00 PM IST`

---

<div align="center">
  <sub>Built with ❤️ for Karnataka Police Datathon 2026 by Team Madhan, Sai Ram, Vijay, Kalyan & Jenifa.</sub>
</div>
