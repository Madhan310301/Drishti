# Drishti

Drishti is an analytical platform integrating census and crime data processing with machine learning models and an interactive dashboard interface.

## Project Structure

```
Drishti/
│
├── app/                  # Frontend application
│   ├── dashboard/       # Dashboard components and views
│   ├── pages/           # Application pages
│   ├── components/      # Reusable UI components
│   ├── services/        # API integration & data services
│   └── utils/           # Frontend utilities
│
├── backend/              # Backend services & logic
│   ├── api/             # API endpoints/routes
│   ├── ml/              # Machine learning models & training
│   ├── database/        # Database configurations & migrations
│   ├── models/          # Data schemas & ORM models
│   ├── etl/             # Data extraction, transformation, loading
│   ├── utils/           # Helper scripts & shared utilities
│   └── config/          # Environment & app configuration
│
├── data/                 # Data directory
│   ├── raw/             # Raw datasets (census, crime)
│   │   ├── census/
│   │   └── crime/
│   ├── processed/       # Cleaned and feature-engineered datasets
│   └── output/          # Exported predictions & generated reports
│
├── docs/                 # Documentation files
├── scripts/              # Automation and maintenance scripts
└── tests/                # Unit and integration test suites
```

## Getting Started

1. Set up your Python virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
