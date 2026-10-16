"""
EvaluationAgent - Comprehensive model evaluation with AI insights
Generates detailed evaluation reports and recommendations
"""

from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np

# Sklearn metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)

from agents.base_agent import BaseAgent


class EvaluationAgent(BaseAgent):
    """
    Agent responsible for comprehensive model evaluation.
    Features:
    - Multiple metric calculation
    - AI-powered insights generation
    - Model comparison
    - Recommendations for improvement
    """
    
    def __init__(self):
        super().__init__("Evaluation")
    
    def _run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate trained models and generate insights.
        
        Args:
            input_data: Dict containing:
                - trained_models: Dict of trained model objects
                - test_data: Dict with X_test and y_test
                - problem_type: classification or regression
                - training_metrics: Metrics from training
                
        Returns:
            Dict with evaluation results and recommendations
        """
        trained_models = input_data.get('trained_models', {})
        test_data = input_data.get('test_data', {})
        problem_type = input_data.get('problem_type', 'classification')
        training_metrics = input_data.get('training_metrics', {})
        target_column = input_data.get('target_column', 'target')
        selected_model = input_data.get('selected_model')
        
        if not trained_models:
            raise ValueError("No trained models provided for evaluation")
        
        X_test = test_data.get('X_test')
        y_test = test_data.get('y_test')
        
        if X_test is None or y_test is None:
            raise ValueError("Test data (X_test, y_test) is required")
        
        self.logger.info(f"Evaluating {len(trained_models)} models")
        
        # Evaluate each model
        evaluation_results = {}
        
        for model_name, model in trained_models.items():
            try:
                metrics = self._evaluate_model(
                    model=model,
                    model_name=model_name,
                    X_test=X_test,
                    y_test=y_test,
                    problem_type=problem_type
                )
                evaluation_results[model_name] = metrics
                
                # Track evaluation metrics
                self._track_evaluation_metrics(model_name, metrics, problem_type)
                
            except Exception as e:
                self.logger.error(f"Evaluation failed for {model_name}: {str(e)}")
                evaluation_results[model_name] = {'error': str(e)}
        
        # Compare models
        comparison = self._compare_models(evaluation_results, problem_type)
        
        # The model is selected on cross-validation score during training. Re-selecting
        # on the test set would leak test data into model selection, so the test-set
        # ranking is reported for information only.
        comparison['test_set_leader'] = comparison.get('best_model')
        if selected_model in evaluation_results and 'error' not in evaluation_results[selected_model]:
            comparison['best_model'] = selected_model
            comparison['best_score'] = evaluation_results[selected_model].get(comparison['primary_metric'], 0)
            comparison['selection_method'] = 'cross_validation'
        else:
            comparison['selection_method'] = 'test_set'
        
        # Generate AI insights
        insights = self._generate_insights(
            evaluation_results=evaluation_results,
            training_metrics=training_metrics,
            problem_type=problem_type,
            best_model=comparison.get('best_model')
        )
        
        # Track decision confidence
        best_score = comparison.get('best_score', 0.7)
        self.track_decision("evaluation_complete", min(best_score + 0.1, 1.0))
        
        return {
            'evaluation_results': evaluation_results,
            'comparison': comparison,
            'insights': insights,
            'best_model': comparison.get('best_model'),
            'recommendations': insights.get('recommendations', [])
        }
    
    def _evaluate_model(
        self,
        model: Any,
        model_name: str,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        problem_type: str
    ) -> Dict[str, Any]:
        """
        Evaluate a single model with comprehensive metrics.
        
        Args:
            model: Trained model
            model_name: Model name
            X_test: Test features
            y_test: Test labels
            problem_type: classification or regression
            
        Returns:
            Dict with evaluation metrics
        """
        # Get predictions
        y_pred = model.predict(X_test)
        
        metrics = {
            'model_name': model_name,
            'problem_type': problem_type,
            'n_test_samples': len(y_test)
        }
        
        if problem_type == 'classification':
            metrics.update(self._calculate_classification_metrics(y_test, y_pred, model, X_test))
        else:
            metrics.update(self._calculate_regression_metrics(y_test, y_pred))
        
        return metrics
    
    def _calculate_classification_metrics(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        model: Any,
        X_test: pd.DataFrame
    ) -> Dict[str, Any]:
        """Calculate classification metrics."""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
        
        # Handle binary vs multiclass
        n_classes = len(np.unique(y_true))
        average = 'binary' if n_classes == 2 else 'weighted'
        
        try:
            metrics['precision'] = float(precision_score(y_true, y_pred, average=average, zero_division=0))
            metrics['recall'] = float(recall_score(y_true, y_pred, average=average, zero_division=0))
            metrics['f1'] = float(f1_score(y_true, y_pred, average=average, zero_division=0))
        except Exception as e:
            self.logger.warning(f"Error calculating precision/recall/f1: {e}")
        
        # ROC AUC (for binary classification with probability support)
        try:
            if hasattr(model, 'predict_proba') and n_classes == 2:
                y_proba = model.predict_proba(X_test)[:, 1]
                metrics['roc_auc'] = float(roc_auc_score(y_true, y_proba))
            elif hasattr(model, 'predict_proba') and n_classes > 2:
                y_proba = model.predict_proba(X_test)
                metrics['roc_auc'] = float(roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted'))
        except Exception as e:
            self.logger.warning(f"ROC AUC calculation failed: {e}")
        
        # Confusion matrix
        try:
            cm = confusion_matrix(y_true, y_pred)
            metrics['confusion_matrix'] = cm.tolist()
        except Exception as e:
            self.logger.warning(f"Confusion matrix failed: {e}")
        
        # Classification report
        try:
            report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            metrics['classification_report'] = report
        except Exception as e:
            self.logger.warning(f"Classification report failed: {e}")
        
        return metrics
    
    def _calculate_regression_metrics(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """Calculate regression metrics."""
        metrics = {}
        
        metrics['mse'] = float(mean_squared_error(y_true, y_pred))
        metrics['rmse'] = float(np.sqrt(metrics['mse']))
        metrics['mae'] = float(mean_absolute_error(y_true, y_pred))
        metrics['r2'] = float(r2_score(y_true, y_pred))
        
        # Additional metrics
        metrics['mape'] = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100)
        
        # Residual statistics
        residuals = y_true - y_pred
        metrics['residual_mean'] = float(np.mean(residuals))
        metrics['residual_std'] = float(np.std(residuals))
        
        return metrics
    
    def _compare_models(
        self,
        evaluation_results: Dict[str, Dict],
        problem_type: str
    ) -> Dict[str, Any]:
        """
        Compare all evaluated models and rank them.
        
        Args:
            evaluation_results: Dict of model evaluation results
            problem_type: classification or regression
            
        Returns:
            Comparison results with ranking
        """
        # Define primary metric for comparison
        if problem_type == 'classification':
            primary_metric = 'f1'  # F1 for balanced evaluation
            higher_is_better = True
        else:
            primary_metric = 'r2'
            higher_is_better = True
        
        # Build comparison table
        comparison_data = []
        
        for model_name, metrics in evaluation_results.items():
            if 'error' in metrics:
                continue
            
            row = {
                'model': model_name,
                'score': metrics.get(primary_metric, 0)
            }
            
            if problem_type == 'classification':
                row.update({
                    'accuracy': metrics.get('accuracy', 0),
                    'precision': metrics.get('precision', 0),
                    'recall': metrics.get('recall', 0),
                    'f1': metrics.get('f1', 0),
                    'roc_auc': metrics.get('roc_auc', 0)
                })
            else:
                row.update({
                    'r2': metrics.get('r2', 0),
                    'rmse': metrics.get('rmse', 0),
                    'mae': metrics.get('mae', 0)
                })
            
            comparison_data.append(row)
        
        # Sort by primary metric
        comparison_data.sort(
            key=lambda x: x['score'],
            reverse=higher_is_better
        )
        
        # Determine best model
        best_model = comparison_data[0]['model'] if comparison_data else None
        best_score = comparison_data[0]['score'] if comparison_data else 0
        
        return {
            'ranking': comparison_data,
            'best_model': best_model,
            'best_score': best_score,
            'primary_metric': primary_metric,
            'n_models_compared': len(comparison_data)
        }
    
    def _track_evaluation_metrics(
        self,
        model_name: str,
        metrics: Dict[str, Any],
        problem_type: str
    ):
        """Track evaluation metrics."""
        self.logger.info(
            f"Evaluation metrics: {model_name}",
            extra={
                'model_name': model_name,
                'problem_type': problem_type,
                **{k: v for k, v in metrics.items() if isinstance(v, (int, float))}
            }
        )
    
    def _generate_insights(
        self,
        evaluation_results: Dict[str, Dict],
        training_metrics: Dict[str, Dict],
        problem_type: str,
        best_model: Optional[str]
    ) -> Dict[str, Any]:
        """
        Use Gemini to generate evaluation insights and recommendations.
        
        Args:
            evaluation_results: Evaluation metrics for all models
            training_metrics: Training metrics for context
            problem_type: classification or regression
            best_model: Name of best performing model
            
        Returns:
            Insights and recommendations
        """
        # Prepare summary for LLM
        eval_summary = {}
        for model_name, metrics in evaluation_results.items():
            if 'error' not in metrics:
                if problem_type == 'classification':
                    eval_summary[model_name] = {
                        'accuracy': metrics.get('accuracy', 0),
                        'f1': metrics.get('f1', 0),
                        'precision': metrics.get('precision', 0),
                        'recall': metrics.get('recall', 0)
                    }
                else:
                    eval_summary[model_name] = {
                        'r2': metrics.get('r2', 0),
                        'rmse': metrics.get('rmse', 0),
                        'mae': metrics.get('mae', 0)
                    }
        
        prompt = f"""
        Analyze these machine learning model evaluation results and provide insights.

        Problem Type: {problem_type}
        Best Model: {best_model}
        
        Evaluation Results:
        {eval_summary}

        Provide a JSON response with:
        {{
            "summary": "Brief summary of model performance (2-3 sentences)",
            "strengths": ["Key strength 1", "Key strength 2"],
            "weaknesses": ["Potential issue 1", "Potential issue 2"],
            "recommendations": [
                "Specific actionable recommendation 1",
                "Specific actionable recommendation 2",
                "Specific actionable recommendation 3"
            ],
            "production_readiness": "HIGH|MEDIUM|LOW",
            "production_readiness_rationale": "Why this rating"
        }}

        Be specific and actionable in your recommendations.
        Consider overfitting, underfitting, and model generalization.
        """
        
        try:
            insights = self.call_gemini_json(prompt, temperature=0.4)
            return insights
            
        except Exception as e:
            self.logger.warning(f"AI insights generation failed: {e}")
            return self._get_default_insights(evaluation_results, problem_type, best_model)
    
    def _get_default_insights(
        self,
        evaluation_results: Dict[str, Dict],
        problem_type: str,
        best_model: Optional[str]
    ) -> Dict[str, Any]:
        """Generate default insights when AI fails."""
        # Get best model metrics
        best_metrics = evaluation_results.get(best_model, {}) if best_model else {}
        
        # Determine performance level
        if problem_type == 'classification':
            score = best_metrics.get('f1', 0)
        else:
            score = best_metrics.get('r2', 0)
        
        if score >= 0.9:
            readiness = "HIGH"
            summary = f"Excellent model performance. {best_model} achieved strong results."
        elif score >= 0.7:
            readiness = "MEDIUM"
            summary = f"Good model performance. {best_model} shows promising results with room for improvement."
        else:
            readiness = "LOW"
            summary = f"Model performance needs improvement. Consider feature engineering or more data."
        
        recommendations = [
            "Perform hyperparameter tuning for further optimization",
            "Validate model on additional holdout data",
            "Consider ensemble methods for improved performance"
        ]
        
        if score < 0.7:
            recommendations.insert(0, "Review feature engineering and consider additional features")
            recommendations.insert(1, "Check for data quality issues or class imbalance")
        
        return {
            'summary': summary,
            'strengths': ["Model training completed successfully"],
            'weaknesses': ["Limited hyperparameter optimization"],
            'recommendations': recommendations,
            'production_readiness': readiness,
            'production_readiness_rationale': f"Based on {problem_type} score of {score:.2f}"
        }
