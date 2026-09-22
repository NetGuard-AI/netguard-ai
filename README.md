# NetGuard AI

AI-based network attack forecasting and proactive cyber defence system.

## Project architecture

NetGuard AI is organized into seven functional modules grouped into five implementation areas:

- **M1 — Data & Network Telemetry**
- **M2 — Network State & AI World Model**
- **M3 — Attack Forecasting & Forward Simulation**
- **M4 — Attack Stage + Explainable AI**
- **M5 — Risk, Alerts & Security Intelligence**
- **M6 — NetGuard Application**
- **M7 — Cyber Safety & Assistance**

Core dependency:

`M1 → M2 → M3/M4 → M5 → M6`

M7 supports the application and help layer.

## Repository structure

- `m1_data/` — M1 Data & Network Telemetry
- `m2_world_model/` — M2 Network State & AI World Model
- `m3_forecast_xai/` — M3 Attack Forecasting + M4 Explainable AI
- `m4_backend/` — M5 Risk/Alerts + M7 Cyber Safety & Assistance
- `m6_frontend/` — M6 NetGuard Application

## Technology direction

Python 3.10+, pandas, NumPy, scikit-learn, PyTorch/GRU, SHAP, FastAPI, Pydantic, SQLite, JWT/bcrypt, ReportLab, React, Vite, Recharts, and pytest.

## Status

Repository scaffold created. Module implementations, trained models, generated datasets, and deployment configuration will be added incrementally.

## Source of truth

The project SRS defines the module contracts, file names, interfaces, dependencies, and implementation requirements. Cross-module contract changes are coordinated through the integration lead.
