"""
Alerting - Automated incident creation and alert management
Uses logging-based alerting (Datadog removed)
"""

import logging
from typing import Any, Dict, Optional
from datetime import datetime


logger = logging.getLogger(__name__)


class IncidentManager:
    """
    Manages automated incident creation via logging.
    Creates context-rich incidents for critical pipeline failures.
    """
    
    def __init__(self):
        """Initialize the incident manager."""
        logger.info("IncidentManager initialized (logging-based)")
    
    def create_incident(
        self,
        title: str,
        severity: str,
        description: str,
        context: Dict[str, Any]
    ) -> Optional[str]:
        """
        Create an incident with full context (logged locally).
        
        Args:
            title: Incident title
            severity: Severity level (SEV-1, SEV-2, SEV-3, SEV-4)
            description: Detailed description
            context: Additional context dict
            
        Returns:
            Incident ID (timestamp-based) if created, None otherwise
        """
        logger.info(f"Incident creation attempt", extra={
            'severity': severity,
            'title': title
        })
        
        # Generate incident ID based on timestamp
        incident_id = f"INC-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Log the incident
        self._log_incident_locally(title, severity, description, context, incident_id)
        
        logger.info(f"Incident created: {incident_id}", extra={
            'incident_id': incident_id,
            'severity': severity
        })
        
        return incident_id
    
    def _log_incident_locally(
        self,
        title: str,
        severity: str,
        description: str,
        context: Dict[str, Any],
        incident_id: str
    ):
        """
        Log incident locally.
        
        Args:
            title: Incident title
            severity: Severity level
            description: Description
            context: Context dict
            incident_id: Generated incident ID
        """
        incident_log = {
            'incident_id': incident_id,
            'timestamp': datetime.now().isoformat(),
            'title': title,
            'severity': severity,
            'description': description,
            'context': context,
            'status': 'logged_locally'
        }
        
        logger.critical(
            f"INCIDENT [{severity}]: {title}",
            extra=incident_log
        )


def handle_critical_failure(error_context: Dict[str, Any]) -> Optional[str]:
    """
    Handle critical pipeline failure by creating an incident.
    
    Args:
        error_context: Dict containing:
            - agent: Agent name where failure occurred
            - error: Error message
            - timestamp: When the error occurred
            - pipeline_id: Pipeline identifier
            - model: Optional model name
            
    Returns:
        Incident ID if created
    """
    agent = error_context.get('agent', 'unknown')
    error = error_context.get('error', 'Unknown error')
    timestamp = error_context.get('timestamp', datetime.now())
    pipeline_id = error_context.get('pipeline_id', 'unknown')
    model = error_context.get('model', 'N/A')
    
    # Determine severity based on agent
    if agent.lower() in ['orchestrator', 'training']:
        severity = 'SEV-2'
    elif agent.lower() in ['ingestion', 'evaluation']:
        severity = 'SEV-2'
    else:
        severity = 'SEV-3'
    
    title = f"AutoDS Pipeline Critical Failure - {agent}"
    
    description = f"""
## Pipeline Failure Detected

**Agent:** {agent}
**Pipeline ID:** {pipeline_id}
**Timestamp:** {timestamp}
**Model:** {model}

### Error Details
```
{error}
```

### Affected Components
- Agent: {agent}
- Pipeline: {pipeline_id}
- Model: {model}

### Recommended Actions
1. Review agent logs
2. Check Vertex AI / Gemini API status
3. Validate input data quality
4. Review recent changes to pipeline configuration
    """
    
    # Log the failure
    logger.error(f"Critical failure in {agent}", extra={
        'agent': agent,
        'pipeline_id': pipeline_id,
        'error': str(error)[:500]
    })
    
    # Create incident
    incident_manager = IncidentManager()
    incident_id = incident_manager.create_incident(
        title=title,
        severity=severity,
        description=description,
        context=error_context
    )
    
    return incident_id


def create_alert_notification(
    alert_type: str,
    message: str,
    context: Dict[str, Any],
    severity: str = 'warning'
):
    """
    Create an alert notification (logged locally).
    
    Args:
        alert_type: Type of alert (e.g., 'high_latency', 'data_quality')
        message: Alert message
        context: Additional context
        severity: Alert severity (info, warning, error, critical)
    """
    log_level = {
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'critical': logging.CRITICAL
    }.get(severity, logging.WARNING)
    
    logger.log(
        log_level,
        f"Alert [{severity}] {alert_type}: {message}",
        extra={
            'alert_type': alert_type,
            'severity': severity,
            'context': context
        }
    )


def check_and_alert_thresholds(metrics: Dict[str, float], thresholds: Dict[str, Dict]):
    """
    Check metrics against thresholds and create alerts.
    
    Args:
        metrics: Dict of metric names to values
        thresholds: Dict of metric names to threshold configs
                   e.g., {'llm_latency': {'warning': 5, 'critical': 10}}
    """
    for metric_name, value in metrics.items():
        if metric_name not in thresholds:
            continue
        
        threshold = thresholds[metric_name]
        
        if 'critical' in threshold and value >= threshold['critical']:
            create_alert_notification(
                alert_type=f'{metric_name}_critical',
                message=f'{metric_name} is at critical level: {value}',
                context={'metric': metric_name, 'value': value, 'threshold': threshold['critical']},
                severity='error'
            )
        elif 'warning' in threshold and value >= threshold['warning']:
            create_alert_notification(
                alert_type=f'{metric_name}_warning',
                message=f'{metric_name} is elevated: {value}',
                context={'metric': metric_name, 'value': value, 'threshold': threshold['warning']},
                severity='warning'
            )
