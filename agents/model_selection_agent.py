"""
ModelSelectionAgent - AI-powered model selection for ML pipelines
Analyzes data and recommends optimal algorithms
"""

from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np

from agents.base_agent import BaseAgent


class ModelSelectionAgent(BaseAgent):
    """
    Agent responsible for selecting appropriate ML models.
    Uses Gemini to analyze data characteristics and recommend models.
    """
    
    # Available models by problem type
    CLASSIFICATION_MODELS = {
        'random_forest': 'sklearn.ensemble.RandomForestClassifier',
        'xgboost': 'xgboost.XGBClassifier',
        'logistic_regression': 'sklearn.linear_model.LogisticRegression',
        'gradient_boosting': 'sklearn.ensemble.GradientBoostingClassifier',
        'svm': 'sklearn.svm.SVC',
        'knn': 'sklearn.neighbors.KNeighborsClassifier',
        'decision_tree': 'sklearn.tree.DecisionTreeClassifier',
        'naive_bayes': 'sklearn.naive_bayes.GaussianNB'
    }
    
    REGRESSION_MODELS = {
        'random_forest': 'sklearn.ensemble.RandomForestRegressor',
        'xgboost': 'xgboost.XGBRegressor',
        'linear_regression': 'sklearn.linear_model.LinearRegression',
        'ridge': 'sklearn.linear_model.Ridge',
        'lasso': 'sklearn.linear_model.Lasso',
        'gradient_boosting': 'sklearn.ensemble.GradientBoostingRegressor',
        'svr': 'sklearn.svm.SVR',
        'elastic_net': 'sklearn.linear_model.ElasticNet'
    }
    
    def __init__(self):
        super().__init__("ModelSelection")
    
    def _run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select appropriate ML models based on data characteristics.
        
        Args:
            input_data: Dict containing:
                - data: Preprocessed DataFrame
                - target_column: Target variable name
                - problem_type: Optional problem type override
                - preprocessing_strategy: Strategy from preprocessing
                
        Returns:
            Dict with problem type and recommended models
        """
        df = input_data.get('data')
        if df is None:
            raise ValueError("No data provided for model selection")
        
        target_column = input_data.get('target_column')
        forced_problem_type = input_data.get('problem_type')
        
        if not target_column or target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        # Determine problem type
        problem_type = self._determine_problem_type(df, target_column, forced_problem_type)
        
        self.logger.info(f"Problem type determined: {problem_type}")
        
        # Analyze data characteristics
        data_profile = self._profile_data(df, target_column)
        
        # Get AI recommendations
        recommendations = self._get_model_recommendations(
            df, target_column, problem_type, data_profile
        )
        
        # Log recommendations
        self.logger.info(
            f"Recommended {len(recommendations.get('recommended_models', []))} models",
            extra={'problem_type': problem_type}
        )
        
        # Track decision confidence
        confidence = recommendations.get('confidence', 0.8)
        self.track_decision("model_selection", confidence)
        
        return {
            'problem_type': problem_type,
            'recommended_models': recommendations.get('recommended_models', []),
            'model_configs': recommendations.get('model_configs', {}),
            'rationale': recommendations.get('rationale', ''),
            'data_profile': data_profile,
            'data': df,  # Pass through
            'target_column': target_column
        }
    
    def _determine_problem_type(
        self,
        df: pd.DataFrame,
        target_column: str,
        forced_type: Optional[str] = None
    ) -> str:
        """
        Determine if this is a classification or regression problem.
        
        Args:
            df: DataFrame with target
            target_column: Target column name
            forced_type: Optional override
            
        Returns:
            'classification' or 'regression'
        """
        if forced_type and forced_type in ['classification', 'regression']:
            return forced_type
        
        target = df[target_column]
        
        # Check data type
        if pd.api.types.is_string_dtype(target) or target.dtype.name == 'category':
            return 'classification'
        
        # Check number of unique values
        n_unique = target.nunique()
        n_total = len(target)
        
        # Heuristic: if unique values are less than 20 or less than 5% of total, likely classification
        if n_unique <= 20 or (n_unique / n_total) < 0.05:
            # Additional check: are all values integers? (only for numeric types)
            if pd.api.types.is_numeric_dtype(target):
                if np.all(target.dropna() == target.dropna().astype(int)):
                    return 'classification'
            else:
                # Non-numeric with few unique values is classification
                return 'classification'
        
        return 'regression'
    
    def _profile_data(self, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """
        Create a profile of the dataset for model selection.
        
        Args:
            df: DataFrame
            target_column: Target column name
            
        Returns:
            Data profile dictionary
        """
        # Separate features and target
        features = df.drop(columns=[target_column])
        target = df[target_column]
        
        profile = {
            'n_samples': len(df),
            'n_features': len(features.columns),
            'feature_types': {
                'numeric': len(features.select_dtypes(include=[np.number]).columns),
                'categorical': len(features.select_dtypes(include=['object', 'category', 'string']).columns)
            },
            'target_info': {
                'name': target_column,
                'dtype': str(target.dtype),
                'n_unique': int(target.nunique())
            },
            'data_characteristics': {}
        }
        
        # Sample size category
        if profile['n_samples'] < 1000:
            profile['data_characteristics']['size'] = 'small'
        elif profile['n_samples'] < 10000:
            profile['data_characteristics']['size'] = 'medium'
        else:
            profile['data_characteristics']['size'] = 'large'
        
        # Feature dimension category
        if profile['n_features'] < 10:
            profile['data_characteristics']['dimensionality'] = 'low'
        elif profile['n_features'] < 50:
            profile['data_characteristics']['dimensionality'] = 'medium'
        else:
            profile['data_characteristics']['dimensionality'] = 'high'
        
        # Check for class imbalance (classification)
        if pd.api.types.is_string_dtype(target) or target.nunique() <= 20:
            class_counts = target.value_counts()
            imbalance_ratio = class_counts.max() / class_counts.min() if class_counts.min() > 0 else float('inf')
            profile['data_characteristics']['class_imbalance'] = imbalance_ratio > 3
            profile['target_info']['class_distribution'] = class_counts.to_dict()
        
        # Check for feature correlation
        numeric_features = features.select_dtypes(include=[np.number])
        if len(numeric_features.columns) > 1:
            corr_matrix = numeric_features.corr().abs()
            high_corr = (corr_matrix > 0.8).sum().sum() - len(numeric_features.columns)
            profile['data_characteristics']['high_correlation'] = high_corr > 0
        
        return profile
    
    def _get_model_recommendations(
        self,
        df: pd.DataFrame,
        target_column: str,
        problem_type: str,
        data_profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Use Gemini to get model recommendations.
        
        Args:
            df: DataFrame
            target_column: Target column name
            problem_type: classification or regression
            data_profile: Data profile from _profile_data
            
        Returns:
            Model recommendations
        """
        available_models = (
            list(self.CLASSIFICATION_MODELS.keys()) if problem_type == 'classification'
            else list(self.REGRESSION_MODELS.keys())
        )
        
        prompt = f"""
        Select the best machine learning models for this problem.

        Problem Type: {problem_type}
        
        Data Profile:
        - Samples: {data_profile['n_samples']}
        - Features: {data_profile['n_features']}
        - Numeric features: {data_profile['feature_types']['numeric']}
        - Categorical features: {data_profile['feature_types']['categorical']}
        - Data size: {data_profile['data_characteristics'].get('size', 'unknown')}
        - Dimensionality: {data_profile['data_characteristics'].get('dimensionality', 'unknown')}
        - Class imbalance: {data_profile['data_characteristics'].get('class_imbalance', 'N/A')}
        - High feature correlation: {data_profile['data_characteristics'].get('high_correlation', 'unknown')}

        Target Info:
        - Name: {target_column}
        - Unique values: {data_profile['target_info']['n_unique']}
        
        Available Models: {available_models}

        Return a JSON object with:
        {{
            "recommended_models": ["model1", "model2", "model3"],
            "model_configs": {{
                "model1": {{"param1": value, "param2": value}},
                "model2": {{"param1": value}}
            }},
            "rationale": "Explanation of why these models were selected",
            "confidence": 0.85
        }}

        Select 2-4 models that are most appropriate for this data.
        Include sensible hyperparameter starting points in model_configs.
        Only use models from the available models list.
        """
        
        try:
            recommendations = self.call_gemini_json(prompt, temperature=0.3)
            
            # Validate recommended models
            valid_models = [m for m in recommendations.get('recommended_models', []) if m in available_models]
            
            if not valid_models:
                valid_models = self._get_default_models(problem_type, data_profile)
            
            recommendations['recommended_models'] = valid_models
            
            # Ensure model_configs exists
            if 'model_configs' not in recommendations:
                recommendations['model_configs'] = self._get_default_configs(valid_models, problem_type)
            
            return recommendations
            
        except Exception as e:
            self.logger.warning(f"AI model selection failed: {e}, using defaults")
            return self._get_default_recommendations(problem_type, data_profile)
    
    def _get_default_models(self, problem_type: str, data_profile: Dict) -> List[str]:
        """Get default model list based on data characteristics."""
        size = data_profile['data_characteristics'].get('size', 'medium')
        
        if problem_type == 'classification':
            if size == 'small':
                return ['logistic_regression', 'random_forest', 'knn']
            elif size == 'large':
                return ['xgboost', 'random_forest', 'gradient_boosting']
            else:
                return ['random_forest', 'xgboost', 'logistic_regression']
        else:
            if size == 'small':
                return ['linear_regression', 'ridge', 'random_forest']
            elif size == 'large':
                return ['xgboost', 'random_forest', 'gradient_boosting']
            else:
                return ['random_forest', 'xgboost', 'ridge']
    
    def _get_default_configs(self, models: List[str], problem_type: str) -> Dict[str, Dict]:
        """Get default hyperparameter configurations."""
        configs = {}
        
        for model in models:
            if model == 'random_forest':
                configs[model] = {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 5,
                    'random_state': 42
                }
            elif model == 'xgboost':
                configs[model] = {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'random_state': 42
                }
            elif model == 'logistic_regression':
                configs[model] = {
                    'max_iter': 1000,
                    'random_state': 42
                }
            elif model == 'gradient_boosting':
                configs[model] = {
                    'n_estimators': 100,
                    'max_depth': 5,
                    'learning_rate': 0.1,
                    'random_state': 42
                }
            elif model in ['linear_regression', 'ridge', 'lasso']:
                configs[model] = {}
            elif model == 'knn':
                configs[model] = {
                    'n_neighbors': 5
                }
            else:
                configs[model] = {'random_state': 42}
        
        return configs
    
    def _get_default_recommendations(self, problem_type: str, data_profile: Dict) -> Dict[str, Any]:
        """Get default recommendations when AI fails."""
        models = self._get_default_models(problem_type, data_profile)
        
        return {
            'recommended_models': models,
            'model_configs': self._get_default_configs(models, problem_type),
            'rationale': f"Default selection for {problem_type} with {data_profile['data_characteristics'].get('size', 'medium')} dataset",
            'confidence': 0.7
        }
