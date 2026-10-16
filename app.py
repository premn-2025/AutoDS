"""
AutoDS Pipeline - Flask Application
Production-ready API for the autonomous data science pipeline
"""

import os
import uuid
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from flask import Flask, request, jsonify, g, render_template, send_from_directory
from werkzeug.utils import secure_filename

# Import pipeline components
from agents.orchestrator import OrchestratorAgent
from agents.data_ingestion_agent import DataIngestionAgent
from agents.preprocessing_agent import PreprocessingAgent
from agents.model_selection_agent import ModelSelectionAgent
from agents.training_agent import TrainingAgent
from agents.evaluation_agent import EvaluationAgent
from agents.eda_agent import EDAAgent
from agents.forecasting_agent import ForecastingAgent
from observability.metrics_collector import MetricsCollector
from observability.llm_monitor import LLMMonitor
from observability.alerting import handle_critical_failure


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max
app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_FOLDER', '/tmp/autods_uploads')

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize observability
metrics = MetricsCollector()
llm_monitor = LLMMonitor()

# Pipeline state storage (in production, use Redis or database)
pipeline_runs: Dict[str, Dict[str, Any]] = {}


# Allowed file extensions
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls', 'parquet', 'json'}


def serialize_result(obj):
    """Convert DataFrames, numpy types and model objects into JSON-serializable values."""
    import pandas as pd
    import numpy as np
    
    if isinstance(obj, pd.DataFrame):
        return {
            '_type': 'DataFrame',
            'shape': list(obj.shape),
            'columns': list(obj.columns),
            'head': obj.head(5).to_dict('records')
        }
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: serialize_result(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_result(item) for item in obj]
    elif hasattr(obj, '__dict__'):
        return str(obj)  # For model objects
    else:
        try:
            # Try to convert to native Python type
            return obj.item() if hasattr(obj, 'item') else obj
        except (AttributeError, ValueError):
            return str(obj)


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def create_orchestrator() -> OrchestratorAgent:
    """
    Create and configure the orchestrator with all agents.
    
    Returns:
        Configured OrchestratorAgent
    """
    orchestrator = OrchestratorAgent()
    
    # Register all pipeline agents
    orchestrator.register_agent('ingestion', DataIngestionAgent())
    orchestrator.register_agent('eda', EDAAgent())  # Add EDA agent
    orchestrator.register_agent('preprocessing', PreprocessingAgent())
    orchestrator.register_agent('model_selection', ModelSelectionAgent())
    orchestrator.register_agent('training', TrainingAgent())
    orchestrator.register_agent('evaluation', EvaluationAgent())
    
    return orchestrator


# Request hooks for metrics
@app.before_request
def before_request():
    """Track request start time."""
    g.start_time = datetime.now()
    g.request_id = request.headers.get('X-Request-ID', str(uuid.uuid4()))


@app.after_request
def after_request(response):
    """Track request metrics."""
    # Add request ID to response
    response.headers['X-Request-ID'] = g.get('request_id', 'unknown')
    return response


@app.errorhandler(Exception)
def handle_exception(e):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
    
    return jsonify({
        'error': 'Internal server error',
        'message': str(e),
        'request_id': g.get('request_id')
    }), 500


# API Endpoints

@app.route('/')
def index():
    """Serve the main UI page."""
    return render_template('index.html')


@app.route('/data/<path:filename>')
def serve_data(filename):
    """Serve sample data files."""
    data_folder = os.path.join(os.path.dirname(__file__), 'data')
    return send_from_directory(data_folder, filename)


@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint.
    
    Returns:
        Health status and component availability
    """
    health = {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': os.getenv('APP_VERSION', '1.0.0'),
        'components': {
            'llm_available': True,  # Would check Vertex AI
            'storage_available': os.path.exists(app.config['UPLOAD_FOLDER'])
        }
    }
    
    return jsonify(health), 200


@app.route('/ready', methods=['GET'])
def readiness_check():
    """
    Readiness check for Kubernetes/Cloud Run.
    
    Returns:
        Readiness status
    """
    # Check critical dependencies
    ready = True
    checks = {}
    
    # Check upload folder
    checks['storage'] = os.path.exists(app.config['UPLOAD_FOLDER'])
    
    # Check environment variables
    checks['config'] = bool(
        os.getenv('GOOGLE_CLOUD_PROJECT') or os.getenv('GEMINI_API_KEY')
    )
    
    ready = all(checks.values())
    
    return jsonify({
        'ready': ready,
        'checks': checks
    }), 200 if ready else 503


@app.route('/pipeline/run', methods=['POST'])
def run_pipeline():
    """
    Start a new pipeline run.
    
    Accepts:
        - file: Uploaded data file (CSV, Excel, Parquet, JSON)
        - config: Pipeline configuration (JSON)
        
    Returns:
        Pipeline run ID and initial status
    """
    pipeline_id = str(uuid.uuid4())
    
    try:
        # Validate file upload
        if 'file' not in request.files:
            return jsonify({
                'error': 'No file uploaded',
                'message': 'Please provide a data file'
            }), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'error': 'No file selected',
                'message': 'Please select a file to upload'
            }), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'error': 'Invalid file type',
                'message': f'Allowed extensions: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400
        
        # Save file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{pipeline_id}_{filename}")
        file.save(file_path)
        
        # Parse config
        config = {}
        if 'config' in request.form:
            try:
                config = json.loads(request.form['config'])
            except json.JSONDecodeError:
                return jsonify({
                    'error': 'Invalid config',
                    'message': 'Config must be valid JSON'
                }), 400
        
        # Extract target column
        target_column = config.get('target_column') or request.form.get('target_column')
        
        if not target_column:
            return jsonify({
                'error': 'Missing target column',
                'message': 'Please specify the target column for prediction'
            }), 400
        
        # Initialize pipeline state
        pipeline_runs[pipeline_id] = {
            'id': pipeline_id,
            'status': 'starting',
            'created_at': datetime.now().isoformat(),
            'file_path': file_path,
            'target_column': target_column,
            'config': config,
            'steps': [],
            'result': None,
            'error': None
        }
        
        # Track pipeline start - pass proper config with data_source
        metrics.track_pipeline_start(pipeline_id, {'data_source': file_path, 'source_type': 'api_upload'})
        
        # Run pipeline (synchronously for now - in production, use Celery/Cloud Tasks)
        try:
            orchestrator = create_orchestrator()
            
            # Build pipeline config
            pipeline_config = {
                'pipeline_id': pipeline_id,
                'data_source': file_path,
                'target_column': target_column,
                **config  # Merge any additional user config
            }
            
            result = orchestrator.run_pipeline(pipeline_config)
            
            pipeline_runs[pipeline_id]['status'] = 'completed'
            pipeline_runs[pipeline_id]['result'] = result
            pipeline_runs[pipeline_id]['completed_at'] = datetime.now().isoformat()
            
        except Exception as e:
            import traceback
            logger.error(f"Pipeline {pipeline_id} failed: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            pipeline_runs[pipeline_id]['status'] = 'failed'
            pipeline_runs[pipeline_id]['error'] = str(e)
            pipeline_runs[pipeline_id]['completed_at'] = datetime.now().isoformat()
            
            # Create incident for critical failure
            handle_critical_failure({
                'agent': 'orchestrator',
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'pipeline_id': pipeline_id
            })
        
        return jsonify({
            'pipeline_id': pipeline_id,
            'status': pipeline_runs[pipeline_id]['status'],
            'message': 'Pipeline execution started'
        }), 202
        
    except Exception as e:
        import traceback
        logger.error(f"Failed to start pipeline: {str(e)}")
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        
        return jsonify({
            'error': 'Pipeline start failed',
            'message': str(e),
            'pipeline_id': pipeline_id
        }), 500


@app.route('/pipeline/<pipeline_id>/status', methods=['GET'])
def get_pipeline_status(pipeline_id: str):
    """
    Get status of a pipeline run.
    
    Args:
        pipeline_id: Unique pipeline identifier
        
    Returns:
        Pipeline status and progress
    """
    if pipeline_id not in pipeline_runs:
        return jsonify({
            'error': 'Pipeline not found',
            'pipeline_id': pipeline_id
        }), 404
    
    run = pipeline_runs[pipeline_id]
    
    return jsonify({
        'pipeline_id': pipeline_id,
        'status': run['status'],
        'created_at': run['created_at'],
        'completed_at': run.get('completed_at'),
        'steps': run.get('steps', []),
        'error': run.get('error')
    }), 200


@app.route('/pipeline/<pipeline_id>/result', methods=['GET'])
def get_pipeline_result(pipeline_id: str):
    """
    Get results of a completed pipeline run.
    
    Args:
        pipeline_id: Unique pipeline identifier
        
    Returns:
        Pipeline results including model metrics and recommendations
    """
    if pipeline_id not in pipeline_runs:
        return jsonify({
            'error': 'Pipeline not found',
            'pipeline_id': pipeline_id
        }), 404
    
    run = pipeline_runs[pipeline_id]
    
    if run['status'] != 'completed':
        return jsonify({
            'error': 'Pipeline not completed',
            'status': run['status'],
            'pipeline_id': pipeline_id
        }), 400
    
    serialized_result = serialize_result(run['result'])
    
    # Extract key metrics for easier frontend access
    training_result = serialized_result.get('results', {}).get('training', {})
    evaluation_result = serialized_result.get('results', {}).get('evaluation', {})
    
    # Build flattened model_results for UI
    model_results = {}
    training_metrics = training_result.get('training_metrics', {})
    for model_name, metrics in training_metrics.items():
        if isinstance(metrics, dict) and 'error' not in metrics:
            model_results[model_name] = {
                'cv_score': metrics.get('cv_score_mean', 0),
                'test_score': metrics.get('test_score', 0),
                'training_time': metrics.get('training_time', 0),
                'feature_importance': metrics.get('feature_importance', {}),
                'tuned': metrics.get('tuned', False),
                'best_params': metrics.get('best_params', {})
            }
    
    return jsonify({
        'pipeline_id': pipeline_id,
        'status': 'completed',
        'result': serialized_result,
        # Flattened fields for UI
        'model_results': model_results,
        'best_model': training_result.get('best_model', ''),
        'problem_type': training_result.get('problem_type', 'classification'),
        'evaluation': evaluation_result
    }), 200


@app.route('/pipeline/runs', methods=['GET'])
def list_pipeline_runs():
    """
    List all pipeline runs.
    
    Query params:
        - status: Filter by status
        - limit: Max results (default 50)
        
    Returns:
        List of pipeline runs
    """
    status_filter = request.args.get('status')
    limit = int(request.args.get('limit', 50))
    
    runs = []
    for run_id, run in pipeline_runs.items():
        if status_filter and run['status'] != status_filter:
            continue
        
        runs.append({
            'pipeline_id': run['id'],
            'status': run['status'],
            'created_at': run['created_at'],
            'completed_at': run.get('completed_at')
        })
        
        if len(runs) >= limit:
            break
    
    return jsonify({
        'runs': runs,
        'count': len(runs)
    }), 200


@app.route('/metrics', methods=['GET'])
def get_metrics():
    """
    Get current system metrics.
    
    Returns:
        LLM and pipeline metrics
    """
    try:
        llm_metrics = llm_monitor.get_cost_report()
    except Exception as e:
        logger.warning(f"Failed to get LLM metrics: {e}")
        llm_metrics = {
            'total_requests': 0,
            'total_tokens': 0,
            'estimated_cost_usd': 0.0,
            'error': str(e)
        }
    
    return jsonify({
        'llm': llm_metrics,
        'pipeline': {
            'total_runs': len(pipeline_runs),
            'completed': len([r for r in pipeline_runs.values() if r['status'] == 'completed']),
            'failed': len([r for r in pipeline_runs.values() if r['status'] == 'failed']),
            'running': len([r for r in pipeline_runs.values() if r['status'] == 'running'])
        }
    }), 200


@app.route('/config', methods=['GET'])
def get_config():
    """
    Get current pipeline configuration.
    
    Returns:
        Active configuration settings
    """
    return jsonify({
        'environment': os.getenv('APP_ENV', 'development'),
        'version': os.getenv('APP_VERSION', '1.0.0'),
        'llm_model': os.getenv('LLM_MODEL', 'gemini-flash-latest'),
        'max_file_size_mb': 100,
        'allowed_extensions': list(ALLOWED_EXTENSIONS),
        'observability': {
            'metrics_enabled': True
        }
    }), 200


# Dashboard Endpoints
@app.route('/pipeline/<pipeline_id>/dashboard', methods=['GET'])
def get_dashboard_data(pipeline_id: str):
    """
    Get dashboard data with statistics and chart data for visualization.
    
    Args:
        pipeline_id: Unique pipeline identifier
        
    Returns:
        Comprehensive dashboard data including stats, distributions, correlations
    """
    import pandas as pd
    import numpy as np
    
    if pipeline_id not in pipeline_runs:
        return jsonify({
            'error': 'Pipeline not found',
            'pipeline_id': pipeline_id
        }), 404
    
    run = pipeline_runs[pipeline_id]
    
    # Load the original data file
    file_path = run.get('file_path')
    if not file_path or not os.path.exists(file_path):
        return jsonify({
            'error': 'Data file not found'
        }), 404
    
    try:
        df = read_data_file(file_path)
        target_column = run.get('target_column')
        
        # Basic Statistics
        stats = {
            'rows': len(df),
            'columns': len(df.columns),
            'missing_total': int(df.isnull().sum().sum()),
            'missing_by_column': df.isnull().sum().to_dict(),
            'duplicates': int(df.duplicated().sum()),
            'memory_mb': round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
        }
        
        # Column info
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        columns = {
            'numeric': numeric_cols,
            'categorical': categorical_cols,
            'total': len(df.columns)
        }
        
        # Numeric statistics
        numeric_stats = {}
        for col in numeric_cols:
            numeric_stats[col] = {
                'mean': round(float(df[col].mean()), 4),
                'std': round(float(df[col].std()), 4),
                'min': round(float(df[col].min()), 4),
                'max': round(float(df[col].max()), 4),
                'median': round(float(df[col].median()), 4),
                'q1': round(float(df[col].quantile(0.25)), 4),
                'q3': round(float(df[col].quantile(0.75)), 4)
            }
        
        # Distributions (histogram data for charts)
        distributions = {}
        for col in numeric_cols[:6]:  # Limit to 6 columns
            hist, bin_edges = np.histogram(df[col].dropna(), bins=15)
            distributions[col] = {
                'values': hist.tolist(),
                'bins': [round(float(b), 2) for b in bin_edges.tolist()]
            }
        
        # Correlations
        correlations = {}
        if len(numeric_cols) >= 2:
            corr_matrix = df[numeric_cols].corr()
            correlations = {
                'columns': numeric_cols,
                'matrix': [[round(float(corr_matrix.iloc[i, j]), 3) 
                           for j in range(len(numeric_cols))] 
                          for i in range(len(numeric_cols))]
            }
        
        # Categorical value counts
        categorical_stats = {}
        for col in categorical_cols[:5]:  # Limit to 5
            value_counts = df[col].value_counts().head(10)
            categorical_stats[col] = {
                'labels': value_counts.index.tolist(),
                'values': value_counts.values.tolist()
            }
        
        # Target distribution
        target_distribution = None
        if target_column and target_column in df.columns:
            if df[target_column].dtype == 'object' or df[target_column].nunique() < 20:
                vc = df[target_column].value_counts()
                target_distribution = {
                    'labels': vc.index.tolist(),
                    'values': vc.values.tolist()
                }
        
        # Model results summary (if pipeline completed)
        model_summary = None
        if run['status'] == 'completed' and run.get('result'):
            result = run['result']
            results = result.get('results', {})
            training = results.get('training', {})
            
            if training.get('training_metrics'):
                model_summary = {
                    'models': [],
                    'best_model': training.get('best_model')
                }
                for name, metrics in training['training_metrics'].items():
                    if isinstance(metrics, dict) and 'error' not in metrics:
                        model_summary['models'].append({
                            'name': name,
                            'cv_score': round(metrics.get('cv_score_mean', 0), 4),
                            'test_score': round(metrics.get('test_score', 0), 4)
                        })
        
        return jsonify({
            'pipeline_id': pipeline_id,
            'stats': stats,
            'columns': columns,
            'numeric_stats': numeric_stats,
            'distributions': distributions,
            'correlations': correlations,
            'categorical_stats': categorical_stats,
            'target_distribution': target_distribution,
            'model_summary': model_summary,
            'target_column': target_column
        }), 200
        
    except Exception as e:
        import traceback
        logger.error(f"Dashboard error: {traceback.format_exc()}")
        return jsonify({
            'error': 'Failed to generate dashboard',
            'message': str(e)
        }), 500


@app.route('/pipeline/<pipeline_id>/dashboard/download', methods=['GET'])
def download_dashboard(pipeline_id: str):
    """
    Download dashboard as HTML report.
    
    Args:
        pipeline_id: Unique pipeline identifier
        
    Returns:
        HTML file download
    """
    import pandas as pd
    import numpy as np
    from flask import Response
    
    if pipeline_id not in pipeline_runs:
        return jsonify({'error': 'Pipeline not found'}), 404
    
    run = pipeline_runs[pipeline_id]
    file_path = run.get('file_path')
    
    if not file_path or not os.path.exists(file_path):
        return jsonify({'error': 'Data file not found'}), 404
    
    try:
        df = read_data_file(file_path)
        target_column = run.get('target_column')
        
        # Generate HTML report
        html_content = generate_dashboard_html(df, target_column, run, pipeline_id)
        
        response = Response(
            html_content,
            mimetype='text/html',
            headers={
                'Content-Disposition': f'attachment; filename=dashboard_{pipeline_id}.html'
            }
        )
        return response
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def read_data_file(file_path: str):
    """Load an uploaded dataset based on its file extension."""
    import pandas as pd

    ext = file_path.rsplit('.', 1)[-1].lower()
    if ext in ('xlsx', 'xls'):
        return pd.read_excel(file_path)
    if ext == 'parquet':
        return pd.read_parquet(file_path)
    if ext == 'json':
        return pd.read_json(file_path)
    return pd.read_csv(file_path)


REPORT_CSS = """
:root {
    --bg: #f5f6fa; --surface: #ffffff; --surface-alt: #f0f2f8; --border: #e1e4ee;
    --text: #1d2233; --muted: #5d6478; --accent: #5a67d8; --accent-soft: #e9ebfc;
    --good: #1f9d55; --warn: #c27c0e; --bad: #d64545; --bar: #7c88e6;
}
@media (prefers-color-scheme: dark) {
    :root {
        --bg: #14161f; --surface: #1d2030; --surface-alt: #252a3d; --border: #30364d;
        --text: #e8eaf2; --muted: #9aa1b8; --accent: #8b96f0; --accent-soft: #2a2f4a;
        --good: #48c78e; --warn: #f0b429; --bad: #f07178; --bar: #8b96f0;
    }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text);
       font: 14px/1.55 -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; }
.wrap { max-width: 1100px; margin: 0 auto; padding: 0 16px 48px; }
header.report { background: linear-gradient(135deg, #5a67d8 0%, #764ba2 100%); color: #fff; padding: 32px 0 28px; margin-bottom: 24px; }
header.report h1 { margin: 0 0 6px; font-size: 26px; }
header.report .meta { display: flex; flex-wrap: wrap; gap: 6px 22px; opacity: .92; font-size: 13px; }
header.report .meta b { font-weight: 600; }
h2 { font-size: 17px; margin: 0 0 14px; }
h3 { font-size: 14px; margin: 18px 0 8px; color: var(--muted); text-transform: uppercase; letter-spacing: .04em; }
section.card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 20px 22px; margin-bottom: 20px; break-inside: avoid; }
.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin-bottom: 20px; }
.kpi { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 14px 16px; }
.kpi .v { font-size: 24px; font-weight: 700; }
.kpi .l { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }
.summary { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 2fr); gap: 20px; }
.best { background: var(--accent-soft); border-radius: 10px; padding: 16px; }
.best .name { font-size: 22px; font-weight: 700; color: var(--accent); }
.best .score { font-size: 32px; font-weight: 700; margin-top: 4px; }
.badge { display: inline-block; padding: 2px 9px; border-radius: 999px; font-size: 11px; font-weight: 600; letter-spacing: .03em; }
.badge.good { background: rgba(31,157,85,.15); color: var(--good); }
.badge.warn { background: rgba(194,124,14,.15); color: var(--warn); }
.badge.bad { background: rgba(214,69,69,.15); color: var(--bad); }
.badge.accent { background: var(--accent-soft); color: var(--accent); }
.table-wrap { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; font-size: 13px; font-variant-numeric: tabular-nums; }
th, td { padding: 7px 10px; border-bottom: 1px solid var(--border); text-align: left; white-space: nowrap; }
th { color: var(--muted); font-weight: 600; font-size: 12px; background: var(--surface-alt); }
td.num, th.num { text-align: right; }
tr.is-best td { background: var(--accent-soft); font-weight: 600; }
ul { margin: 6px 0 0; padding-left: 20px; }
li { margin-bottom: 4px; }
.cols { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 20px; }
.bar-row { display: grid; grid-template-columns: minmax(90px, 160px) 1fr 56px; gap: 10px; align-items: center; margin-bottom: 6px; font-size: 13px; }
.bar-row .label { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.bar-track { background: var(--surface-alt); border-radius: 4px; height: 12px; }
.bar-fill { background: var(--bar); border-radius: 4px; height: 12px; }
.bar-row .val { text-align: right; color: var(--muted); font-variant-numeric: tabular-nums; }
.cm td { text-align: center; min-width: 52px; }
.chart { margin: 0 0 16px; }
.chart img { display: block; max-width: 100%; max-height: 520px; width: auto; height: auto; border-radius: 8px; background: #fff; border: 1px solid var(--border); }
.chart figcaption { color: var(--muted); font-size: 12px; margin-top: 6px; }
.muted { color: var(--muted); }
footer { text-align: center; color: var(--muted); font-size: 12px; padding-top: 12px; }
@media (max-width: 720px) { .summary { grid-template-columns: 1fr; } header.report h1 { font-size: 21px; } }
@media print {
    body { background: #fff; color: #000; }
    header.report { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
    section.card, .kpi { border-color: #ccc; }
    .table-wrap { overflow: visible; }
    th, td { white-space: normal; }
}
"""


def generate_dashboard_html(df, target_column, run, pipeline_id):
    """Generate a standalone, self-contained HTML analysis report."""
    import numpy as np
    from html import escape
    from datetime import datetime

    def esc(value) -> str:
        return escape(str(value))

    def fmt(value, digits=4) -> str:
        if value is None:
            return '—'
        try:
            value = float(value)
        except (TypeError, ValueError):
            return esc(value)
        if np.isnan(value):
            return '—'
        if abs(value) >= 1e6:
            return f"{value:,.0f}"
        if abs(value) >= 1000:
            return f"{value:,.2f}"
        return f"{value:.{digits}f}"

    def pct(value) -> str:
        return '—' if value is None else f"{float(value) * 100:.2f}%"

    acronyms = {'Knn': 'KNN', 'Svm': 'SVM', 'Xgboost': 'XGBoost', 'Lightgbm': 'LightGBM', 'Svr': 'SVR'}
    
    def pretty_raw(name: str) -> str:
        words = str(name).replace('_', ' ').title().split()
        return ' '.join(acronyms.get(w, w) for w in words)
    
    def pretty(name: str) -> str:
        return esc(pretty_raw(name))

    def as_list(value):
        if not value:
            return []
        return value if isinstance(value, list) else [value]

    def bullet_list(items) -> str:
        return '<ul>' + ''.join(f'<li>{esc(i)}</li>' for i in items) + '</ul>'

    # Pipeline results
    result = run.get('result') or {}
    results = result.get('results', {}) if isinstance(result, dict) else {}
    training = results.get('training', {}) or {}
    evaluation = results.get('evaluation', {}) or {}
    preprocessing = results.get('preprocessing', {}) or {}
    model_selection = results.get('model_selection', {}) or {}
    eda = results.get('eda', {}) or {}

    train_metrics = {k: v for k, v in (training.get('training_metrics') or {}).items()
                     if isinstance(v, dict) and 'error' not in v}
    eval_metrics = evaluation.get('evaluation_results') or {}
    best_model = training.get('best_model', '')
    problem_type = training.get('problem_type') or model_selection.get('problem_type')
    if not problem_type and target_column in df.columns:
        is_numeric_target = np.issubdtype(df[target_column].dtype, np.number)
        problem_type = 'regression' if is_numeric_target and df[target_column].nunique() >= 20 else 'classification'
    is_regression = problem_type == 'regression'

    # Map encoded class labels back to their original names
    class_names = None
    target_encoder = (preprocessing.get('encoders') or {}).get(target_column)
    if target_encoder is not None and hasattr(target_encoder, 'classes_'):
        class_names = [str(c) for c in target_encoder.classes_]

    # Dataset overview
    rows, cols = df.shape
    missing = int(df.isnull().sum().sum())
    missing_pct = missing / df.size * 100 if df.size else 0
    duplicates = int(df.duplicated().sum())
    numeric_df = df.select_dtypes(include=[np.number])
    quality_score = (eda.get('quality_report') or {}).get('overall_score')
    upload_name = os.path.basename(run.get('file_path', ''))
    dataset_name = upload_name[len(pipeline_id) + 1:] if upload_name.startswith(f"{pipeline_id}_") else upload_name

    kpis = [
        (f"{rows:,}", 'Rows'),
        (f"{cols}", 'Columns'),
        (f"{missing:,} <span class='muted' style='font-size:13px'>({missing_pct:.1f}%)</span>", 'Missing values'),
        (f"{duplicates:,}", 'Duplicate rows'),
    ]
    if quality_score is not None:
        kpis.append((f"{float(quality_score):.0f}%", 'Data quality score'))
    kpi_html = ''.join(f'<div class="kpi"><div class="v">{v}</div><div class="l">{l}</div></div>' for v, l in kpis)

    # Executive summary
    summary_html = ''
    if best_model:
        best_eval = eval_metrics.get(best_model, {})
        best_train = train_metrics.get(best_model, {})
        if is_regression:
            headline_label, headline = 'R² on test set', fmt(best_eval.get('r2', best_train.get('test_score')))
        else:
            headline_label, headline = 'Accuracy on test set', pct(best_eval.get('accuracy', best_train.get('test_score')))

        insights = evaluation.get('insights') or {}
        readiness = insights.get('production_readiness')
        badge_class = {'HIGH': 'good', 'MEDIUM': 'warn', 'LOW': 'bad'}.get(str(readiness).upper(), 'accent')
        readiness_html = (f'<div style="margin-top:10px">Production readiness: '
                          f'<span class="badge {badge_class}">{esc(readiness)}</span></div>') if readiness else ''

        notes = []
        if insights.get('summary'):
            notes.append(f'<p style="margin-top:0">{esc(insights["summary"])}</p>')
        ai_insights = training.get('ai_insights') or {}
        if ai_insights.get('best_model_analysis'):
            notes.append(f'<p>{esc(ai_insights["best_model_analysis"])}</p>')
        eval_best = (evaluation.get('comparison') or {}).get('best_model')
        if eval_best and eval_best != best_model:
            notes.append(f'<p class="muted">Note: the model was selected on cross-validation score. '
                         f'On the held-out test set, <b>{pretty(eval_best)}</b> scored highest by '
                         f'{esc((evaluation.get("comparison") or {}).get("primary_metric", "the primary metric"))}.</p>')
        if insights.get('strengths'):
            notes.append('<h3>Strengths</h3>' + bullet_list(as_list(insights['strengths'])))
        if insights.get('weaknesses'):
            notes.append('<h3>Weaknesses</h3>' + bullet_list(as_list(insights['weaknesses'])))

        summary_html = f"""
        <section class="card">
            <h2>Executive summary</h2>
            <div class="summary">
                <div class="best">
                    <div class="muted">Best model</div>
                    <div class="name">{pretty(best_model)}</div>
                    <div class="score">{headline}</div>
                    <div class="muted">{headline_label}</div>
                    {readiness_html}
                </div>
                <div>{''.join(notes) or '<p class="muted">No additional insights were generated.</p>'}</div>
            </div>
        </section>
        """

    # Model comparison
    model_html = ''
    if train_metrics:
        if is_regression:
            eval_cols = [('r2', 'Test R²', fmt), ('mae', 'MAE', fmt), ('rmse', 'RMSE', fmt), ('mape', 'MAPE %', lambda v: fmt(v, 2))]
            cv_label = 'CV R²'
        else:
            eval_cols = [('accuracy', 'Accuracy', pct), ('precision', 'Precision', pct), ('recall', 'Recall', pct),
                         ('f1', 'F1 score', pct), ('roc_auc', 'ROC AUC', fmt)]
            cv_label = 'CV accuracy'

        header = (f'<th>Model</th><th class="num">{cv_label} (mean ± std)</th>'
                  + ''.join(f'<th class="num">{label}</th>' for _, label, _ in eval_cols)
                  + '<th class="num">Train time</th><th>Tuned</th>')
        body = ''
        order = sorted(train_metrics, key=lambda n: train_metrics[n].get('cv_score_mean', float('-inf')), reverse=True)
        for name in order:
            m = train_metrics[name]
            e = eval_metrics.get(name, {})
            cv_fmt = fmt if is_regression else pct
            cv_std = m.get('cv_score_std')
            cv_cell = cv_fmt(m.get('cv_score_mean')) + (f" ± {cv_fmt(cv_std)}" if cv_std is not None else '')
            if not e:
                e = {'r2' if is_regression else 'accuracy': m.get('test_score')}
            cells = ''.join(f'<td class="num">{f(e.get(key))}</td>' for key, _, f in eval_cols)
            best_badge = ' <span class="badge accent">BEST</span>' if name == best_model else ''
            body += (f'<tr class="{"is-best" if name == best_model else ""}"><td>{pretty(name)}{best_badge}</td>'
                     f'<td class="num">{cv_cell}</td>{cells}'
                     f'<td class="num">{float(m.get("training_time", 0)):.3f}s</td>'
                     f'<td>{"Yes" if m.get("tuned") else "No"}</td></tr>')

        metric_note = ('R² of 1.0 is a perfect fit; 0 means no better than predicting the mean. MAE/RMSE are in the target\'s units.'
                       if is_regression else
                       'Test metrics are weighted averages across classes, computed on the held-out test set.')
        model_html = f"""
        <section class="card">
            <h2>Model comparison</h2>
            <div class="table-wrap"><table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table></div>
            <p class="muted" style="margin-bottom:0">{metric_note}</p>
        </section>
        """

    # Best model details: feature importance + confusion matrix / residuals
    detail_parts = []
    importance = (train_metrics.get(best_model) or {}).get('feature_importance') or {}
    if importance:
        items = sorted(importance.items(), key=lambda kv: abs(kv[1]), reverse=True)[:15]
        top = max(abs(v) for _, v in items) or 1
        bars = ''.join(
            f'<div class="bar-row"><div class="label" title="{esc(k)}">{esc(k)}</div>'
            f'<div class="bar-track"><div class="bar-fill" style="width:{abs(v) / top * 100:.1f}%"></div></div>'
            f'<div class="val">{float(v):.3f}</div></div>'
            for k, v in items)
        detail_parts.append(f'<div><h3>Feature importance</h3>{bars}</div>')

    best_eval = eval_metrics.get(best_model, {})
    cm = best_eval.get('confusion_matrix')
    if cm is not None and not is_regression:
        cm = np.asarray(cm)
        labels = class_names if class_names and len(class_names) == len(cm) else [str(i) for i in range(len(cm))]
        peak = cm.max() or 1
        head = '<th>Actual \\ Predicted</th>' + ''.join(f'<th>{esc(l)}</th>' for l in labels)
        body = ''
        for i, row in enumerate(cm):
            cells = ''.join(
                f'<td style="background: rgba(90,103,216,{0.08 + 0.6 * v / peak:.2f})">{int(v)}</td>' for v in row)
            body += f'<tr><th>{esc(labels[i])}</th>{cells}</tr>'
        detail_parts.append(f'<div><h3>Confusion matrix (test set)</h3><div class="table-wrap">'
                            f'<table class="cm"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div></div>')

        report = best_eval.get('classification_report') or {}
        per_class = [(k, v) for k, v in report.items() if isinstance(v, dict) and k not in ('macro avg', 'weighted avg')]
        if per_class:
            def class_label(key):
                return labels[int(key)] if str(key).isdigit() and int(key) < len(labels) else key
            rows_html = ''.join(
                f'<tr><td>{esc(class_label(k))}</td><td class="num">{pct(v.get("precision"))}</td>'
                f'<td class="num">{pct(v.get("recall"))}</td><td class="num">{pct(v.get("f1-score"))}</td>'
                f'<td class="num">{int(v.get("support", 0))}</td></tr>' for k, v in per_class)
            detail_parts.append(
                '<div><h3>Per-class metrics</h3><div class="table-wrap"><table><thead><tr><th>Class</th>'
                '<th class="num">Precision</th><th class="num">Recall</th><th class="num">F1</th><th class="num">Support</th>'
                f'</tr></thead><tbody>{rows_html}</tbody></table></div></div>')
    elif is_regression and best_eval:
        residual_rows = [('Residual mean', best_eval.get('residual_mean')), ('Residual std', best_eval.get('residual_std')),
                         ('MSE', best_eval.get('mse'))]
        rows_html = ''.join(f'<tr><td>{l}</td><td class="num">{fmt(v)}</td></tr>' for l, v in residual_rows if v is not None)
        if best_eval.get('n_test_samples') is not None:
            rows_html += f'<tr><td>Test samples</td><td class="num">{int(best_eval["n_test_samples"]):,}</td></tr>'
        detail_parts.append(f'<div><h3>Error analysis (test set)</h3><table><tbody>{rows_html}</tbody></table></div>')

    details_html = (f'<section class="card"><h2>Best model details: {pretty(best_model)}</h2>'
                    f'<div class="cols">{"".join(detail_parts)}</div></section>') if detail_parts else ''

    # Recommendations
    recs = []
    for source in (as_list(evaluation.get('recommendations')),
                   as_list((training.get('ai_insights') or {}).get('improvement_suggestions')),
                   as_list((eda.get('insights') or {}).get('preprocessing_recommendations'))):
        for item in source:
            if item and item not in recs:
                recs.append(item)
    findings = as_list((eda.get('insights') or {}).get('key_findings'))
    quality_issues = [f"{col}: {int(n):,} missing ({n / max(len(df), 1) * 100:.1f}%)"
                      for col, n in df.isnull().sum().items() if n > 0]
    if duplicates:
        quality_issues.append(f"{duplicates:,} duplicate rows")
    insight_cols = []
    if findings:
        insight_cols.append('<div><h3>Key findings</h3>' + bullet_list(findings) + '</div>')
    if quality_issues:
        insight_cols.append('<div><h3>Data quality issues</h3>' + bullet_list(quality_issues) + '</div>')
    if recs:
        insight_cols.append('<div><h3>Recommendations</h3>' + bullet_list(recs) + '</div>')
    insights_html = (f'<section class="card"><h2>Insights &amp; recommendations</h2>'
                     f'<div class="cols">{"".join(insight_cols)}</div></section>') if insight_cols else ''

    # Pipeline steps
    strategy = preprocessing.get('strategy') or {}
    step_rows = []
    if model_selection.get('rationale'):
        step_rows.append(('Model selection', model_selection['rationale']))
    if model_selection.get('recommended_models'):
        step_rows.append(('Models evaluated', ', '.join(pretty_raw(m) for m in as_list(model_selection['recommended_models']))))
    mv = strategy.get('missing_value_strategy') or {}
    if mv:
        step_rows.append(('Missing values', ', '.join(f'{k}: {v}' for k, v in mv.items())))
    enc = strategy.get('encoding_strategy') or {}
    if enc.get('columns'):
        step_rows.append(('Encoding', f"{enc.get('method', '')} on {', '.join(map(str, enc['columns']))}"))
    sc = strategy.get('scaling_strategy') or {}
    if sc.get('columns'):
        step_rows.append(('Scaling', f"{sc.get('method', '')} on {len(sc['columns'])} numeric columns"))
    if strategy.get('columns_to_drop'):
        step_rows.append(('Dropped columns', ', '.join(map(str, strategy['columns_to_drop']))))
    if result.get('duration') is not None:
        step_rows.append(('Total pipeline time', f"{float(result['duration']):.2f}s"))
    pipeline_html = ''
    if step_rows:
        rows_html = ''.join(f'<tr><td><b>{esc(k)}</b></td><td style="white-space:normal">{esc(v)}</td></tr>' for k, v in step_rows)
        pipeline_html = f'<section class="card"><h2>Pipeline steps</h2><table><tbody>{rows_html}</tbody></table></section>'

    # Target distribution
    target_html = ''
    if target_column and target_column in df.columns:
        target = df[target_column]
        if is_regression and np.issubdtype(target.dtype, np.number):
            desc = target.describe()
            stats = [('Mean', desc['mean']), ('Std', desc['std']), ('Min', desc['min']), ('25%', desc['25%']),
                     ('Median', desc['50%']), ('75%', desc['75%']), ('Max', desc['max'])]
            content = ('<table><thead><tr>' + ''.join(f'<th class="num">{l}</th>' for l, _ in stats) + '</tr></thead><tbody><tr>'
                       + ''.join(f'<td class="num">{fmt(v, 2)}</td>' for _, v in stats) + '</tr></tbody></table>')
        else:
            counts = target.value_counts()
            top = counts.max() or 1
            content = ''.join(
                f'<div class="bar-row"><div class="label" title="{esc(v)}">{esc(v)}</div>'
                f'<div class="bar-track"><div class="bar-fill" style="width:{c / top * 100:.1f}%"></div></div>'
                f'<div class="val">{c / len(df) * 100:.1f}%</div></div>'
                for v, c in counts.head(20).items())
            if len(counts) > 20:
                content += f'<p class="muted">Showing top 20 of {len(counts)} classes.</p>'
        target_html = f'<section class="card"><h2>Target: {esc(target_column)}</h2><div class="table-wrap">{content}</div></section>'

    # Column information
    outliers = ((eda.get('outliers') or {}).get('outliers_by_column')) or {}
    column_rows = ''
    for col in df.columns:
        series = df[col]
        n_missing = int(series.isna().sum())
        role = ' <span class="badge accent">TARGET</span>' if col == target_column else ''
        out = outliers.get(col, {})
        out_cell = f"{out.get('count', 0)} ({out.get('percentage', 0):.1f}%)" if out else '—'
        column_rows += (f'<tr><td>{esc(col)}{role}</td><td>{esc(series.dtype)}</td>'
                        f'<td class="num">{n_missing:,} ({n_missing / max(len(df), 1) * 100:.1f}%)</td>'
                        f'<td class="num">{series.nunique():,}</td><td class="num">{out_cell}</td></tr>')
    columns_html = f"""
    <section class="card">
        <h2>Column overview</h2>
        <div class="table-wrap"><table>
            <thead><tr><th>Column</th><th>Type</th><th class="num">Missing</th><th class="num">Unique</th><th class="num">Outliers (IQR)</th></tr></thead>
            <tbody>{column_rows}</tbody>
        </table></div>
    </section>
    """

    # Numeric statistics
    numeric_html = ''
    if not numeric_df.empty:
        desc = numeric_df.describe().T
        stat_cols = [('mean', 'Mean'), ('std', 'Std'), ('min', 'Min'), ('25%', '25%'), ('50%', 'Median'), ('75%', '75%'), ('max', 'Max')]
        head = '<th>Column</th>' + ''.join(f'<th class="num">{l}</th>' for _, l in stat_cols)
        body = ''.join(f'<tr><td>{esc(idx)}</td>' + ''.join(f'<td class="num">{fmt(row[k], 3)}</td>' for k, _ in stat_cols) + '</tr>'
                       for idx, row in desc.iterrows())
        numeric_html = (f'<section class="card"><h2>Numeric statistics</h2><div class="table-wrap"><table>'
                        f'<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div></section>')

    # Visualizations from the EDA agent (base64 PNG), with a correlation table fallback
    charts = eda.get('visualizations') or {}
    chart_titles = [('target_distribution', 'Target distribution'), ('distributions', 'Feature distributions'),
                    ('boxplots', 'Box plots (outliers)'), ('correlation_heatmap', 'Correlation heatmap')]
    figures = ''.join(
        f'<figure class="chart"><img alt="{title}" src="data:image/png;base64,{charts[key]}"><figcaption>{title}</figcaption></figure>'
        for key, title in chart_titles if isinstance(charts.get(key), str) and charts.get(key))
    corr_html = ''
    if 'correlation_heatmap' not in charts and len(numeric_df.columns) >= 2:
        corr = numeric_df.corr()
        head = '<th></th>' + ''.join(f'<th class="num">{esc(c)}</th>' for c in corr.columns)
        body = ''.join(
            f'<tr><th>{esc(i)}</th>' + ''.join(
                f'<td class="num" style="background: rgba(90,103,216,{abs(v) * 0.55:.2f})">{v:.2f}</td>' for v in row) + '</tr>'
            for i, row in corr.iterrows())
        corr_html = f'<h3>Correlation matrix</h3><div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'
    charts_html = (f'<section class="card"><h2>Exploratory data analysis</h2>{figures}{corr_html}</section>'
                   if figures or corr_html else '')

    meta = [('Dataset', dataset_name or '—'), ('Target', target_column or '—'),
            ('Problem type', (problem_type or 'unknown').title()), ('Generated', datetime.now().strftime('%Y-%m-%d %H:%M')),
            ('Pipeline ID', pipeline_id)]
    meta_html = ''.join(f'<span><b>{k}:</b> {esc(v)}</span>' for k, v in meta)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AutoDS Report - {esc(dataset_name or pipeline_id)}</title>
    <style>{REPORT_CSS}</style>
</head>
<body>
    <header class="report">
        <div class="wrap" style="padding-bottom:0">
            <h1>AutoDS Analysis Report</h1>
            <div class="meta">{meta_html}</div>
        </div>
    </header>
    <main class="wrap">
        <div class="kpis">{kpi_html}</div>
        {summary_html}
        {model_html}
        {details_html}
        {insights_html}
        {target_html}
        {columns_html}
        {numeric_html}
        {charts_html}
        {pipeline_html}
        <footer>Generated by AutoDS Pipeline</footer>
    </main>
</body>
</html>
"""


# Model Management Endpoints
@app.route('/pipeline/<pipeline_id>/model/download', methods=['GET'])
def download_model(pipeline_id: str):
    """
    Download the best trained model from a completed pipeline.
    
    Args:
        pipeline_id: Unique pipeline identifier
        
    Returns:
        Model file (pickle format)
    """
    import pickle
    from io import BytesIO
    from flask import send_file
    
    if pipeline_id not in pipeline_runs:
        return jsonify({
            'error': 'Pipeline not found',
            'pipeline_id': pipeline_id
        }), 404
    
    run = pipeline_runs[pipeline_id]
    
    if run['status'] != 'completed':
        return jsonify({
            'error': 'Pipeline not completed',
            'status': run['status']
        }), 400
    
    # Get best model
    result = run.get('result', {})
    results = result.get('results', {})
    training_result = results.get('training', {})
    best_model_name = training_result.get('best_model')
    trained_models = training_result.get('trained_models', {})
    
    if not best_model_name or best_model_name not in trained_models:
        return jsonify({
            'error': 'No trained model found',
            'pipeline_id': pipeline_id
        }), 404
    
    model = trained_models[best_model_name]
    
    # Serialize to bytes
    model_buffer = BytesIO()
    pickle.dump(model, model_buffer)
    model_buffer.seek(0)
    
    return send_file(
        model_buffer,
        mimetype='application/octet-stream',
        as_attachment=True,
        download_name=f'{pipeline_id}_{best_model_name}.pkl'
    )


@app.route('/pipeline/<pipeline_id>/predict', methods=['POST'])
def predict(pipeline_id: str):
    """
    Make predictions using a trained model.
    
    Args:
        pipeline_id: Unique pipeline identifier
        
    Request body:
        - data: List of records to predict (JSON)
        - OR file: CSV file with data to predict
        
    Returns:
        Predictions for each record
    """
    import pandas as pd
    import numpy as np
    
    if pipeline_id not in pipeline_runs:
        return jsonify({
            'error': 'Pipeline not found',
            'pipeline_id': pipeline_id
        }), 404
    
    run = pipeline_runs[pipeline_id]
    
    if run['status'] != 'completed':
        return jsonify({
            'error': 'Pipeline not completed',
            'status': run['status']
        }), 400
    
    # Get trained model
    result = run.get('result', {})
    results = result.get('results', {})
    training_result = results.get('training', {})
    best_model_name = training_result.get('best_model')
    trained_models = training_result.get('trained_models', {})
    feature_names = training_result.get('feature_names', [])
    
    if not best_model_name or best_model_name not in trained_models:
        return jsonify({
            'error': 'No trained model found'
        }), 404
    
    model = trained_models[best_model_name]
    
    # Get input data
    try:
        if 'file' in request.files:
            file = request.files['file']
            if file.filename.endswith('.csv'):
                df = pd.read_csv(file)
            elif file.filename.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file)
            else:
                return jsonify({'error': 'Unsupported file format'}), 400
        elif request.is_json:
            data = request.get_json()
            if 'data' in data:
                df = pd.DataFrame(data['data'])
            else:
                df = pd.DataFrame([data])
        else:
            return jsonify({
                'error': 'No input data provided',
                'message': 'Provide JSON data or upload a CSV file'
            }), 400
        
        # Ensure correct columns
        X = df.select_dtypes(include=[np.number])
        
        # Match feature columns
        if feature_names:
            missing_cols = set(feature_names) - set(X.columns)
            extra_cols = set(X.columns) - set(feature_names)
            
            # Add missing columns with zeros
            for col in missing_cols:
                X[col] = 0
            
            # Remove extra columns
            X = X[feature_names]
        
        # Handle NaN values
        X = X.fillna(0)
        
        # Make predictions
        predictions = model.predict(X)
        
        # Get probabilities for classification
        probabilities = None
        if hasattr(model, 'predict_proba'):
            try:
                probabilities = model.predict_proba(X).tolist()
            except Exception:
                pass
        
        # Convert predictions to native Python types
        if isinstance(predictions, np.ndarray):
            predictions = predictions.tolist()
        
        return jsonify({
            'pipeline_id': pipeline_id,
            'model': best_model_name,
            'predictions': predictions,
            'probabilities': probabilities,
            'count': len(predictions)
        }), 200
        
    except Exception as e:
        import traceback
        logger.error(f"Prediction failed: {traceback.format_exc()}")
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e)
        }), 500


@app.route('/pipeline/<pipeline_id>/eda', methods=['GET'])
def get_eda_results(pipeline_id: str):
    """
    Get EDA results and visualizations from a pipeline run.
    
    Args:
        pipeline_id: Unique pipeline identifier
        
    Returns:
        EDA results including charts (base64) and statistics
    """
    if pipeline_id not in pipeline_runs:
        return jsonify({
            'error': 'Pipeline not found',
            'pipeline_id': pipeline_id
        }), 404
    
    run = pipeline_runs[pipeline_id]
    
    if run['status'] not in ['completed', 'running']:
        return jsonify({
            'error': 'Pipeline has no EDA results',
            'status': run['status']
        }), 400
    
    result = run.get('result', {})
    results = result.get('results', {})
    eda_result = results.get('eda', {})
    
    if not eda_result:
        return jsonify({
            'error': 'EDA results not available',
            'message': 'EDA agent may not have run'
        }), 404
    
    eda_result = serialize_result(eda_result)
    
    return jsonify({
        'pipeline_id': pipeline_id,
        'basic_stats': eda_result.get('basic_stats'),
        'quality_report': eda_result.get('quality_report'),
        'correlations': eda_result.get('correlations'),
        'distributions': eda_result.get('distributions'),
        'outliers': eda_result.get('outliers'),
        'visualizations': eda_result.get('visualizations'),
        'insights': eda_result.get('insights')
    }), 200


# Time Series Forecasting Endpoints
@app.route('/forecast/run', methods=['POST'])
def run_forecast():
    """
    Run time series forecasting on uploaded data.
    
    Request:
        - file: CSV file with time series data
        - date_column: Column containing dates (optional)
        - target_column: Column to forecast
        - forecast_periods: Number of periods to forecast (default: 30)
        - frequency: Data frequency (D, W, M, H) (default: D)
        
    Returns:
        Forecast results including predictions and visualization data
    """
    import pandas as pd
    import numpy as np
    
    forecast_id = str(uuid.uuid4())
    
    try:
        # Validate file upload
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Save file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{forecast_id}_{filename}")
        file.save(file_path)
        
        # Load data
        df = read_data_file(file_path)
        
        # Get parameters
        date_column = request.form.get('date_column')
        target_column = request.form.get('target_column')
        forecast_periods = int(request.form.get('forecast_periods', 30))
        frequency = request.form.get('frequency', 'D')
        
        if not target_column:
            # Auto-detect: use last numeric column
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_cols:
                return jsonify({'error': 'No numeric column found for forecasting'}), 400
            target_column = numeric_cols[-1]
        
        # Validate target column exists
        if target_column not in df.columns:
            cols_lower = {c.lower(): c for c in df.columns}
            if target_column.lower() in cols_lower:
                target_column = cols_lower[target_column.lower()]
            else:
                return jsonify({
                    'error': f"Target column '{target_column}' not found",
                    'available': list(df.columns)
                }), 400
        
        # Run forecasting agent
        forecasting_agent = ForecastingAgent()
        
        result = forecasting_agent.execute({
            'data': df,
            'date_column': date_column,
            'target_column': target_column,
            'forecast_periods': forecast_periods,
            'frequency': frequency,
            'models': ['exponential_smoothing', 'linear_trend', 'moving_average']
        })
        
        return jsonify({
            'forecast_id': forecast_id,
            'status': 'completed',
            'target_column': target_column,
            'forecast_periods': forecast_periods,
            'frequency': frequency,
            'analysis': result.get('analysis'),
            'model_results': result.get('model_results'),
            'best_model': result.get('best_model'),
            'forecast': result.get('best_forecast'),
            'visualization': result.get('visualization')
        }), 200
        
    except Exception as e:
        import traceback
        logger.error(f"Forecasting failed: {traceback.format_exc()}")
        return jsonify({
            'error': 'Forecasting failed',
            'message': str(e),
            'forecast_id': forecast_id
        }), 500


# Favicon route to avoid 404/500 errors when browsers request /favicon.ico
@app.route('/favicon.ico')
def favicon():
    # Return 204 No Content (ignore favicon requests)
    return '', 204


# Catch-all route for unknown requests (like Chrome DevTools)
@app.route('/.well-known/<path:filename>')
def catch_well_known(filename):
    """Handle .well-known requests from browsers/tools."""
    return '', 404

@app.route('/<path:unknown_path>')
def catch_all(unknown_path):
    """Handle any other unknown routes gracefully."""
    # Don't log common browser/tool requests to reduce noise
    if unknown_path.startswith(('.well-known', 'apple-touch', 'browserconfig', 'robots.txt')):
        return '', 404
    
    logger.warning(f"Unknown route accessed: /{unknown_path}")
    return jsonify({
        'error': 'Not Found',
        'message': f"The requested URL '/{unknown_path}' was not found on this server.",
        'available_endpoints': [
            '/health', '/config', '/metrics', '/pipeline/run', 
            '/forecast/run'
        ]
    }), 404


# Development server
if __name__ == '__main__':
    port = int(os.getenv('PORT', 8080))
    debug = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
    
    logger.info(f"Starting AutoDS Pipeline API on port {port}")
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug
    )
