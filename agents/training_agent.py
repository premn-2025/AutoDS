"""
TrainingAgent - Trains ML models with full observability
Handles model training, cross-validation, and hyperparameter tuning
"""

import time
import json
from typing import Any, Dict, List, Optional, Tuple
import pickle
import os

import pandas as pd
import numpy as np

# Sklearn imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR

# XGBoost
try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from agents.base_agent import BaseAgent


class TrainingAgent(BaseAgent):
    """
    Agent responsible for training ML models.
    Features:
    - Multiple model training
    - Cross-validation
    - Hyperparameter tuning
    - Model persistence
    - Training metrics tracking
    """
    
    # Model mappings
    CLASSIFIERS = {
        'random_forest': RandomForestClassifier,
        'logistic_regression': LogisticRegression,
        'gradient_boosting': GradientBoostingClassifier,
        'decision_tree': DecisionTreeClassifier,
        'knn': KNeighborsClassifier,
        'naive_bayes': GaussianNB,
        'svm': SVC,
    }
    
    REGRESSORS = {
        'random_forest': RandomForestRegressor,
        'linear_regression': LinearRegression,
        'ridge': Ridge,
        'lasso': Lasso,
        'gradient_boosting': GradientBoostingRegressor,
        'svr': SVR,
    }
    
    # Hyperparameter grids for tuning
    HYPERPARAMETER_GRIDS = {
        'random_forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
        },
        'logistic_regression': {
            'C': [0.01, 0.1, 1.0, 10.0],
            'max_iter': [1000],
        },
        'gradient_boosting': {
            'n_estimators': [50, 100],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5],
        },
        'decision_tree': {
            'max_depth': [None, 5, 10, 20],
            'min_samples_split': [2, 5, 10],
        },
        'knn': {
            'n_neighbors': [3, 5, 7, 11],
            'weights': ['uniform', 'distance'],
        },
        'svm': {
            'C': [0.1, 1.0, 10.0],
            'kernel': ['rbf', 'linear'],
        },
        'ridge': {
            'alpha': [0.1, 1.0, 10.0, 100.0],
        },
        'lasso': {
            'alpha': [0.01, 0.1, 1.0, 10.0],
        },
        'svr': {
            'C': [0.1, 1.0, 10.0],
            'kernel': ['rbf', 'linear'],
        },
        'xgboost': {
            'n_estimators': [50, 100],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5, 7],
        },
    }
    
    def __init__(self):
        super().__init__("Training")
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            self.CLASSIFIERS['xgboost'] = XGBClassifier
            self.REGRESSORS['xgboost'] = XGBRegressor
    
    def _run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train ML models on the preprocessed data.
        
        Args:
            input_data: Dict containing:
                - data: Preprocessed DataFrame
                - target_column: Target variable name
                - problem_type: classification or regression
                - selected_models: List of models to train
                - test_size: Train/test split ratio
                
        Returns:
            Dict with trained models and training metrics
        """
        df = input_data.get('data')
        if df is None:
            raise ValueError("No data provided for training")
        
        target_column = input_data.get('target_column')
        problem_type = input_data.get('problem_type', 'classification')
        selected_models = input_data.get('selected_models', ['random_forest'])
        model_configs = input_data.get('model_configs', {})
        test_size = input_data.get('test_size', 0.2)
        enable_tuning = input_data.get('enable_tuning', False)  # Hyperparameter tuning flag
        
        if not target_column or target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found")
        
        self.logger.info(
            f"Starting training: {len(selected_models)} models, "
            f"problem_type={problem_type}"
        )
        
        # Prepare data
        X, y, X_train, X_test, y_train, y_test = self._prepare_data(
            df, target_column, test_size, problem_type
        )
        
        self.logger.info(f"Data split: train={len(X_train)}, test={len(X_test)}")
        
        # Train each model
        trained_models = {}
        training_metrics = {}
        
        for model_name in selected_models:
            try:
                self.logger.info(f"Training model: {model_name}")
                
                model, metrics = self._train_model(
                    model_name=model_name,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    problem_type=problem_type,
                    config=model_configs.get(model_name, {}),
                    enable_tuning=enable_tuning
                )
                
                trained_models[model_name] = model
                training_metrics[model_name] = metrics
                
                # Track model metrics
                self._track_model_metrics(model_name, metrics, problem_type)
                
            except Exception as e:
                self.logger.error(f"Failed to train {model_name}: {str(e)}")
                training_metrics[model_name] = {'error': str(e)}
        
        # Find best model
        best_model_name = self._find_best_model(training_metrics, problem_type)
        
        self.logger.info(f"Training complete. Best model: {best_model_name}")
        
        # Generate AI-powered training insights
        ai_training_insights = self._generate_training_insights(
            training_metrics, best_model_name, problem_type, len(X.columns)
        )
        
        # Track decision confidence
        if best_model_name and best_model_name in training_metrics:
            best_score = training_metrics[best_model_name].get('cv_score_mean', 0.7)
            self.track_decision("model_training", best_score)
        
        return {
            'trained_models': trained_models,
            'training_metrics': training_metrics,
            'best_model': best_model_name,
            'ai_insights': ai_training_insights,
            'test_data': {
                'X_test': X_test,
                'y_test': y_test
            },
            'feature_names': list(X.columns),
            'problem_type': problem_type,
            'target_column': target_column
        }
    
    def _prepare_data(
        self,
        df: pd.DataFrame,
        target_column: str,
        test_size: float,
        problem_type: str = 'classification'
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare data for training by splitting into train/test sets.
        
        Args:
            df: Full DataFrame
            target_column: Target column name
            test_size: Fraction for test set
            problem_type: 'classification' or 'regression'
            
        Returns:
            Tuple of (X, y, X_train, X_test, y_train, y_test)
        """
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Ensure all features are numeric
        X = X.select_dtypes(include=[np.number])
        
        if X.empty:
            raise ValueError("No numeric features available for training")
        
        # Handle any remaining NaN values
        X = X.fillna(X.mean())
        
        # Split data - use stratification for classification to ensure balanced splits
        stratify_param = y if problem_type == 'classification' else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=stratify_param
        )
        
        self.logger.info(
            f"Data split: Train={len(X_train)}, Test={len(X_test)}, "
            f"Features={len(X.columns)}"
        )
        
        return X, y, X_train, X_test, y_train, y_test
    
    def _train_model(
        self,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        problem_type: str,
        config: Dict[str, Any],
        enable_tuning: bool = False
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Train a single model with cross-validation and optional hyperparameter tuning.
        
        Args:
            model_name: Name of the model
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            problem_type: classification or regression
            config: Hyperparameters
            enable_tuning: Whether to perform GridSearchCV tuning
            
        Returns:
            Tuple of (trained model, metrics dict)
        """
        start_time = time.time()
        
        # Get model class
        model_registry = self.CLASSIFIERS if problem_type == 'classification' else self.REGRESSORS
        
        if model_name not in model_registry:
            raise ValueError(f"Unknown model: {model_name} for {problem_type}")
        
        model_class = model_registry[model_name]
        
        # Hyperparameter tuning with GridSearchCV
        if enable_tuning and model_name in self.HYPERPARAMETER_GRIDS:
            self.logger.info(f"Performing hyperparameter tuning for {model_name}")
            
            try:
                base_model = model_class()
                param_grid = self.HYPERPARAMETER_GRIDS[model_name]
                scoring = 'accuracy' if problem_type == 'classification' else 'r2'
                
                grid_search = GridSearchCV(
                    base_model, 
                    param_grid, 
                    cv=3,  # 3-fold for speed
                    scoring=scoring,
                    n_jobs=-1,
                    refit=True
                )
                
                grid_search.fit(X_train, y_train)
                
                model = grid_search.best_estimator_
                best_params = grid_search.best_params_
                tuning_results = {
                    'best_params': best_params,
                    'best_cv_score': float(grid_search.best_score_),
                    'tuned': True
                }
                
                self.logger.info(f"Best params for {model_name}: {best_params}")
                
            except Exception as e:
                self.logger.warning(f"Hyperparameter tuning failed for {model_name}: {e}")
                # Fallback to default training
                model = model_class(**config) if config else model_class()
                model.fit(X_train, y_train)
                tuning_results = {'tuned': False, 'error': str(e)}
        else:
            # Standard training without tuning
            try:
                model = model_class(**config)
            except TypeError:
                self.logger.warning(f"Invalid config for {model_name}, using defaults")
                model = model_class()
            
            model.fit(X_train, y_train)
            
            tuning_results = {'tuned': False}
        
        training_time = time.time() - start_time
        
        # Calculate metrics
        metrics = {
            'training_time': training_time,
            'model_name': model_name,
            'problem_type': problem_type,
            'config': config,
            **tuning_results  # Include tuning results
        }
        
        # Cross-validation score
        try:
            scoring = 'accuracy' if problem_type == 'classification' else 'r2'
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring=scoring)
            metrics['cv_scores'] = cv_scores.tolist()
            metrics['cv_score_mean'] = float(cv_scores.mean())
            metrics['cv_score_std'] = float(cv_scores.std())
        except Exception as e:
            self.logger.warning(f"Cross-validation failed: {e}")
            metrics['cv_score_mean'] = 0.0
        
        # Test set score
        try:
            test_score = model.score(X_test, y_test)
            metrics['test_score'] = float(test_score)
        except Exception as e:
            self.logger.warning(f"Test scoring failed: {e}")
            metrics['test_score'] = 0.0
        
        # Feature importance (if available)
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(X_train.columns, model.feature_importances_))
            # Sort by importance
            feature_importance = dict(sorted(
                feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10])  # Top 10
            metrics['feature_importance'] = feature_importance
        
        self.logger.info(
            f"Model {model_name} trained: "
            f"CV={metrics.get('cv_score_mean', 0):.4f}, "
            f"Test={metrics.get('test_score', 0):.4f}, "
            f"Time={training_time:.2f}s"
        )
        
        return model, metrics
    
    def _generate_training_insights(
        self,
        training_metrics: Dict[str, Dict],
        best_model: Optional[str],
        problem_type: str,
        n_features: int
    ) -> Dict[str, Any]:
        """Generate AI-powered insights about training results using Gemini."""
        metrics_summary = {}
        for name, m in training_metrics.items():
            if 'error' not in m:
                metrics_summary[name] = {
                    'cv_score': m.get('cv_score_mean', 0),
                    'test_score': m.get('test_score', 0),
                    'training_time': round(m.get('training_time', 0), 2)
                }
        
        prompt = f"""Analyze these ML model training results and provide actionable insights:

Problem Type: {problem_type}
Number of Features: {n_features}
Best Model: {best_model}

Model Performance:
{json.dumps(metrics_summary, indent=2)}

Provide:
1. Why the best model likely outperformed others
2. Whether the scores indicate overfitting or underfitting
3. Suggestions for improving model performance
4. Feature engineering recommendations
5. Next steps for model optimization

Format as JSON with keys: best_model_analysis, overfitting_assessment, improvement_suggestions, feature_recommendations, next_steps"""
        
        try:
            response = self.call_gemini(prompt, temperature=0.4)
            try:
                insights = json.loads(response)
            except json.JSONDecodeError:
                insights = {'raw_insights': response}
            return insights
        except Exception as e:
            self.logger.warning(f"AI training insights failed: {e}")
            return {
                'best_model_analysis': f"{best_model} achieved the highest cross-validation score",
                'improvement_suggestions': ['Try hyperparameter tuning', 'Add more features', 'Collect more data']
            }
    
    def _track_model_metrics(
        self,
        model_name: str,
        metrics: Dict[str, Any],
        problem_type: str
    ):
        """Track model training metrics."""
        self.logger.info(
            f"Model metrics: {model_name}",
            extra={
                'model_name': model_name,
                'problem_type': problem_type,
                'training_time': metrics.get('training_time', 0),
                'cv_score': metrics.get('cv_score_mean', 0),
                'test_score': metrics.get('test_score', 0)
            }
        )
    
    def _find_best_model(
        self,
        training_metrics: Dict[str, Dict],
        problem_type: str
    ) -> Optional[str]:
        """
        Find the best performing model.
        
        Args:
            training_metrics: Dict of model metrics
            problem_type: classification or regression
            
        Returns:
            Name of best model
        """
        best_model = None
        best_score = -float('inf')
        
        for model_name, metrics in training_metrics.items():
            if 'error' in metrics:
                continue
            
            # Use CV score for comparison
            score = metrics.get('cv_score_mean', 0)
            
            if score > best_score:
                best_score = score
                best_model = model_name
        
        return best_model
    
    def save_model(self, model: Any, model_name: str, output_path: str):
        """
        Save a trained model to disk.
        
        Args:
            model: Trained model object
            model_name: Name for the saved model
            output_path: Directory to save to
        """
        os.makedirs(output_path, exist_ok=True)
        filepath = os.path.join(output_path, f"{model_name}.pkl")
        
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        
        self.logger.info(f"Model saved to: {filepath}")
    
    def load_model(self, filepath: str) -> Any:
        """
        Load a saved model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded model object
        """
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        
        self.logger.info(f"Model loaded from: {filepath}")
        
        return model
