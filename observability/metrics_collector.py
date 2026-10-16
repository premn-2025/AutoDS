"""
MetricsCollector - Centralized metrics collection
Provides utility functions for tracking various pipeline metrics via logging
"""

import os
import logging
from typing import Any, Dict, List, Optional


logger = logging.getLogger(__name__)


class MetricsCollector:
    """
    Centralized metrics collection for the AutoDS pipeline.
    Provides static methods for tracking various types of metrics via logging.
    """
    
    _initialized = False
    
    @classmethod
    def initialize(cls):
        """Initialize metrics collection."""
        if cls._initialized:
            return
        cls._initialized = True
        logger.info("MetricsCollector initialized")
    
    @staticmethod
    def track_pipeline_start(pipeline_id: str, config: Dict[str, Any]):
        """
        Track pipeline execution start.
        
        Args:
            pipeline_id: Unique pipeline identifier
            config: Pipeline configuration
        """
        logger.info(
            "Pipeline started",
            extra={
                'pipeline_id': pipeline_id,
                'data_source': config.get("data_source", "unknown")
            }
        )
    
    @staticmethod
    def track_pipeline_completion(
        pipeline_id: str,
        duration: float,
        status: str,
        error: Optional[str] = None
    ):
        """
        Track pipeline execution completion.
        
        Args:
            pipeline_id: Unique pipeline identifier
            duration: Execution duration in seconds
            status: 'success' or 'failure'
            error: Error message if failed
        """
        logger.info(
            f"Pipeline completed: {status}",
            extra={
                'pipeline_id': pipeline_id,
                'duration': duration,
                'status': status,
                'error': error
            }
        )
    
    @staticmethod
    def track_model_training(model_name: str, metrics: Dict[str, float]):
        """
        Track ML model training metrics.
        
        Args:
            model_name: Name of the model
            metrics: Dict of metric names to values
        """
        logger.info(
            f"Model training metrics: {model_name}",
            extra={
                'model_name': model_name,
                **{k: v for k, v in metrics.items() if isinstance(v, (int, float))}
            }
        )
    
    @staticmethod
    def track_model_evaluation(
        model_name: str,
        problem_type: str,
        metrics: Dict[str, float]
    ):
        """
        Track model evaluation metrics.
        
        Args:
            model_name: Name of the model
            problem_type: classification or regression
            metrics: Evaluation metrics
        """
        logger.info(
            f"Model evaluation metrics: {model_name}",
            extra={
                'model_name': model_name,
                'problem_type': problem_type,
                **{k: v for k, v in metrics.items() if isinstance(v, (int, float))}
            }
        )
    
    @staticmethod
    def track_data_quality(stage: str, quality_metrics: Dict[str, float]):
        """
        Track data quality metrics throughout the pipeline.
        
        Args:
            stage: Pipeline stage (ingestion, preprocessing, etc.)
            quality_metrics: Quality metrics dict
        """
        logger.info(
            f"Data quality metrics: {stage}",
            extra={
                'stage': stage,
                **{k: v for k, v in quality_metrics.items() if isinstance(v, (int, float))}
            }
        )
    
    @staticmethod
    def track_agent_decision(
        agent_name: str,
        decision: str,
        confidence: float
    ):
        """
        Track agent decision with confidence score.
        
        Args:
            agent_name: Name of the agent
            decision: Description of the decision
            confidence: Confidence score (0-1)
        """
        log_level = logging.WARNING if confidence < 0.6 else logging.INFO
        logger.log(
            log_level,
            f"Agent decision: {decision}",
            extra={
                'agent_name': agent_name,
                'decision': decision,
                'confidence': confidence
            }
        )
    
    @staticmethod
    def track_feature_importance(
        model_name: str,
        feature_importance: Dict[str, float]
    ):
        """
        Track feature importance scores.
        
        Args:
            model_name: Name of the model
            feature_importance: Dict of feature names to importance scores
        """
        logger.info(
            f"Feature importance: {model_name}",
            extra={
                'model_name': model_name,
                'feature_importance': feature_importance
            }
        )
    
    @staticmethod
    def track_preprocessing_stats(
        rows_before: int,
        rows_after: int,
        columns_before: int,
        columns_after: int
    ):
        """
        Track preprocessing statistics.
        
        Args:
            rows_before: Row count before preprocessing
            rows_after: Row count after preprocessing
            columns_before: Column count before preprocessing
            columns_after: Column count after preprocessing
        """
        row_reduction = (1 - rows_after / rows_before) * 100 if rows_before > 0 else 0
        
        logger.info(
            "Preprocessing stats",
            extra={
                'rows_before': rows_before,
                'rows_after': rows_after,
                'columns_before': columns_before,
                'columns_after': columns_after,
                'row_reduction_pct': row_reduction
            }
        )
    
    @staticmethod
    def track_service_health(
        service_name: str,
        is_healthy: bool,
        component_status: Dict[str, bool]
    ):
        """
        Track service health status.
        
        Args:
            service_name: Name of the service
            is_healthy: Overall health status
            component_status: Health status of individual components
        """
        log_level = logging.INFO if is_healthy else logging.WARNING
        logger.log(
            log_level,
            f"Service health: {service_name}",
            extra={
                'service_name': service_name,
                'is_healthy': is_healthy,
                'component_status': component_status
            }
        )
