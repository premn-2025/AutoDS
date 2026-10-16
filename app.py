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
        df = pd.read_csv(file_path)
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
        df = pd.read_csv(file_path)
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


def generate_dashboard_html(df, target_column, run, pipeline_id):
    """Generate a standalone HTML dashboard report."""
    import numpy as np
    from datetime import datetime
    
    # Basic stats
    rows, cols = df.shape
    missing = int(df.isnull().sum().sum())
    duplicates = int(df.duplicated().sum())
    memory_mb = round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
    
    # Numeric stats
    numeric_df = df.select_dtypes(include=[np.number])
    numeric_stats_html = ""
    if not numeric_df.empty:
        stats_df = numeric_df.describe().round(3).T
        numeric_stats_html = stats_df.to_html(classes='table table-striped table-sm')
    
    # Correlation matrix
    corr_html = ""
    if len(numeric_df.columns) >= 2:
        corr_matrix = numeric_df.corr().round(3)
        corr_html = corr_matrix.to_html(classes='table table-sm text-center')
    
    # Target distribution
    target_dist_html = ""
    if target_column and target_column in df.columns:
        target_counts = df[target_column].value_counts()
        target_dist_html = f"""
        <table class="table table-sm">
            <thead><tr><th>Value</th><th>Count</th><th>Percentage</th></tr></thead>
            <tbody>
            {''.join(f'<tr><td>{val}</td><td>{count}</td><td>{count/len(df)*100:.1f}%</td></tr>' 
                     for val, count in target_counts.items())}
            </tbody>
        </table>
        """
    
    # Model results
    model_html = ""
    if run.get('result'):
        results = run['result'].get('results', {})
        training = results.get('training', {})
        metrics = training.get('training_metrics', {})
        best_model = training.get('best_model', '')
        problem_type = run.get('result', {}).get('problem_type', 'classification')
        is_regression = problem_type == 'regression'
        
        if metrics:
            score_label = "R² Score" if is_regression else "Accuracy"
            model_html = f"""
            <table class="table table-striped table-sm">
                <thead><tr><th>Model</th><th>CV {score_label}</th><th>Test {score_label}</th><th>Training Time</th><th>Status</th></tr></thead>
                <tbody>
            """
            for name, m in metrics.items():
                if isinstance(m, dict) and 'error' not in m:
                    is_best = '⭐ Best' if name == best_model else ''
                    cv_score = m.get('cv_score_mean', 0)
                    test_score = m.get('test_score', 0)
                    if is_regression:
                        cv_display = f"{cv_score:.4f}"
                        test_display = f"{test_score:.4f}"
                    else:
                        cv_display = f"{cv_score*100:.2f}%"
                        test_display = f"{test_score*100:.2f}%"
                    model_html += f"""
                    <tr>
                        <td><strong>{name.replace('_', ' ').title()}</strong></td>
                        <td>{cv_display}</td>
                        <td>{test_display}</td>
                        <td>{m.get('training_time', 0):.3f}s</td>
                        <td style="color: green;">{is_best}</td>
                    </tr>
                    """
            model_html += "</tbody></table>"
    
    # Column info
    column_info_html = """
    <table class="table table-sm">
        <thead><tr><th>Column</th><th>Type</th><th>Non-Null</th><th>Unique</th></tr></thead>
        <tbody>
    """
    for col in df.columns:
        column_info_html += f"""
        <tr>
            <td>{col}</td>
            <td>{df[col].dtype}</td>
            <td>{df[col].notna().sum()}</td>
            <td>{df[col].nunique()}</td>
        </tr>
        """
    column_info_html += "</tbody></table>"
    
    # Full HTML
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AutoDS Dashboard Report - {pipeline_id}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {{ background: #1a1a2e; color: #fff; font-family: 'Segoe UI', sans-serif; }}
        .card {{ background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); margin-bottom: 1.5rem; }}
        .card-header {{ background: rgba(255,255,255,0.05); border-bottom: 1px solid rgba(255,255,255,0.1); font-weight: 600; }}
        .table {{ color: #fff; }}
        .table-striped > tbody > tr:nth-of-type(odd) {{ background: rgba(255,255,255,0.02); }}
        .stat-card {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 12px; padding: 1.5rem; text-align: center; }}
        .stat-value {{ font-size: 2rem; font-weight: bold; }}
        h1, h2, h3 {{ color: #667eea; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; margin-bottom: 2rem; }}
        .header h1 {{ color: #fff; margin: 0; }}
    </style>
</head>
<body>
    <div class="header">
        <div class="container">
            <h1>🤖 AutoDS Pipeline Report</h1>
            <p class="mb-0">Pipeline ID: {pipeline_id} | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    </div>
    
    <div class="container">
        <!-- Summary Stats -->
        <div class="row mb-4">
            <div class="col-md-3">
                <div class="stat-card">
                    <div class="stat-value">{rows:,}</div>
                    <div>Rows</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="stat-card" style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);">
                    <div class="stat-value">{cols}</div>
                    <div>Columns</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="stat-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
                    <div class="stat-value">{missing}</div>
                    <div>Missing Values</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="stat-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
                    <div class="stat-value">{memory_mb}</div>
                    <div>Memory (MB)</div>
                </div>
            </div>
        </div>
        
        <!-- Model Results -->
        {f'''
        <div class="card">
            <div class="card-header">🏆 Model Performance</div>
            <div class="card-body">{model_html}</div>
        </div>
        ''' if model_html else ''}
        
        <!-- Target Distribution -->
        {f'''
        <div class="card">
            <div class="card-header">📊 Target Distribution: {target_column}</div>
            <div class="card-body">{target_dist_html}</div>
        </div>
        ''' if target_dist_html else ''}
        
        <!-- Column Information -->
        <div class="card">
            <div class="card-header">📋 Column Information</div>
            <div class="card-body">{column_info_html}</div>
        </div>
        
        <!-- Numeric Statistics -->
        {f'''
        <div class="card">
            <div class="card-header">📈 Numeric Statistics</div>
            <div class="card-body">{numeric_stats_html}</div>
        </div>
        ''' if numeric_stats_html else ''}
        
        <!-- Correlation Matrix -->
        {f'''
        <div class="card">
            <div class="card-header">🔗 Feature Correlations</div>
            <div class="card-body" style="overflow-x: auto;">{corr_html}</div>
        </div>
        ''' if corr_html else ''}
        
        <div class="text-center text-muted py-4">
            <p>Generated by AutoDS Pipeline | AI-Powered Data Science</p>
        </div>
    </div>
</body>
</html>
    """
    return html


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
        df = pd.read_csv(file_path)
        
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
