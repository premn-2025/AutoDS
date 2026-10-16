# AutoDS Pipeline

A production-ready autonomous data science pipeline powered by AI agents and comprehensive observability.

## 🚀 Features

- **AI-Powered Agents**: 6 specialized agents (Ingestion, Preprocessing, Model Selection, Training, Evaluation) orchestrated by a central coordinator
- **Google Gemini Integration**: Uses Gemini Flash for intelligent decision-making at each pipeline stage
- **Comprehensive Logging**: Structured logging with metrics tracking throughout the pipeline
- **Production Ready**: Docker containerization, Cloud Run deployment, health checks, and structured logging

## 📁 Project Structure

```
autods-pipeline/
├── app.py                      # Flask API application
├── Dockerfile                  # Container image
├── docker-compose.yml          # Local development
├── cloudbuild.yaml             # Cloud Run deployment
├── requirements.txt            # Python dependencies
├── agents/
│   ├── base_agent.py          # Base agent with LLM + observability
│   ├── orchestrator.py        # Pipeline coordinator
│   ├── data_ingestion_agent.py
│   ├── preprocessing_agent.py
│   ├── model_selection_agent.py
│   ├── training_agent.py
│   └── evaluation_agent.py
├── observability/
│   ├── metrics_collector.py   # Logging-based metrics
│   ├── llm_monitor.py         # LLM usage tracking
│   └── alerting.py            # Incident management
├── config/
│   └── pipeline_config.yaml   # Pipeline configuration
├── scripts/
│   ├── run_demo.py            # Local demo script
│   └── api_demo.py            # API client demo
└── data/                      # Sample datasets
```

## 🛠️ Quick Start

### Prerequisites

- Python 3.11+
- Docker (optional)
- Google Cloud account with Vertex AI enabled OR Gemini API key

### 1. Clone and Setup

```bash
# Clone repository
cd autods-pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy example env file
cp .env.example .env

# Edit with your credentials
# Required:
#   - GEMINI_API_KEY or GOOGLE_CLOUD_PROJECT
```

### 3. Run Demo

```bash
# Create sample data and run classification demo
python scripts/run_demo.py --create-data --demo classification

# Run regression demo
python scripts/run_demo.py --demo regression

# Run with your own data
python scripts/run_demo.py --data path/to/data.csv --target target_column
```

### 4. Run API Server

```bash
# Development
python app.py

# Production (with gunicorn)
gunicorn --bind 0.0.0.0:8080 --workers 2 app:app
```

### 5. Test API

```bash
# Health check
curl http://localhost:8080/health

# Run pipeline
curl -X POST http://localhost:8080/pipeline/run \
  -F "file=@data/iris_sample.csv" \
  -F "target_column=species"

# Check status
curl http://localhost:8080/pipeline/{pipeline_id}/status
```

## 🐳 Docker Deployment

### Local Development

```bash
# Start the application
docker-compose up

# Access API at http://localhost:8080
```

### Cloud Run Deployment

```bash
# Deploy using Cloud Build
gcloud builds submit --config=cloudbuild.yaml

# Or deploy manually
gcloud run deploy autods-pipeline \
  --image gcr.io/PROJECT_ID/autods-pipeline:latest \
  --region us-central1 \
  --allow-unauthenticated
```

## 📊 Observability

The pipeline includes comprehensive logging-based observability:

- **Structured Logging**: JSON-formatted logs for easy parsing
- **Metrics Tracking**: Pipeline runs, model training, LLM usage tracked via logs
- **Alerting**: Incident management with logging-based alerts

### Key Metrics (Logged)

| Metric | Description |
|--------|-------------|
| `autods.pipeline.runs` | Total pipeline executions |
| `autods.pipeline.duration` | Pipeline execution time |
| `autods.llm.latency` | LLM API response time |
| `autods.llm.tokens.total` | Total tokens consumed |
| `autods.llm.cost.total` | Estimated LLM costs |
| `autods.model.cv_score` | Cross-validation scores |
| `autods.data.missing_ratio` | Data quality metric |

## 🤖 Agent Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    OrchestratorAgent                         │
│  - Manages pipeline flow                                     │
│  - AI-driven execution planning                              │
│  - Error handling & recovery                                 │
└─────────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ DataIngestion   │  │ Preprocessing   │  │ ModelSelection  │
│ Agent           │→ │ Agent           │→ │ Agent           │
└─────────────────┘  └─────────────────┘  └─────────────────┘
                              │
         ┌────────────────────┴────────────────────┐
         ▼                                         ▼
┌─────────────────┐                       ┌─────────────────┐
│ Training        │  ─────────────────→   │ Evaluation      │
│ Agent           │                       │ Agent           │
└─────────────────┘                       └─────────────────┘
```

Each agent:
1. Receives input from previous step
2. Calls Gemini for intelligent decisions
3. Executes ML operations
4. Tracks metrics via logging
5. Passes results to next agent

## 📋 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/ready` | GET | Readiness check |
| `/pipeline/run` | POST | Start new pipeline |
| `/pipeline/{id}/status` | GET | Get pipeline status |
| `/pipeline/{id}/result` | GET | Get pipeline results |
| `/pipeline/runs` | GET | List all runs |
| `/metrics` | GET | Get system metrics |
| `/config` | GET | Get configuration |

## 🔧 Configuration

Edit `config/pipeline_config.yaml`:

```yaml
llm:
  model: "gemini-flash-latest"
  temperature: 0.1
  max_tokens: 2048

models:
  max_models_to_train: 5
  cv_folds: 5

logging:
  level: INFO
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

## 📄 License

MIT License - See LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `pytest tests/`
5. Submit a pull request

## 🆘 Support

- Check [issues](link-to-issues) for known problems
- Review application logs for debugging
- Check Cloud Run logs for deployment issues
