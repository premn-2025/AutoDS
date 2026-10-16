"""
PreprocessingAgent - Intelligent data preprocessing with AI guidance
Handles missing values, encoding, scaling, and feature engineering
"""

from typing import Any, Dict, List, Optional, Tuple
import json

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer

from agents.base_agent import BaseAgent


class PreprocessingAgent(BaseAgent):
    """
    Agent responsible for data preprocessing.
    Uses AI to determine optimal preprocessing strategy.
    """
    
    def __init__(self):
        super().__init__("Preprocessing")
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
    
    def _run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess data based on AI recommendations.
        
        Args:
            input_data: Dict containing:
                - data: DataFrame to preprocess
                - metadata: Data metadata from ingestion
                - analysis: AI analysis from ingestion
                - target_column: Target variable name
                
        Returns:
            Dict with preprocessed data and strategy used
        """
        df = input_data.get('data')
        if df is None:
            raise ValueError("No data provided for preprocessing")
        
        target_column = input_data.get('target_column')
        analysis = input_data.get('analysis', '')
        metadata = input_data.get('metadata', {})
        
        # Make a copy to avoid modifying original
        df = df.copy()
        
        self.logger.info(f"Starting preprocessing for data with shape: {df.shape}")
        
        # Get AI-recommended preprocessing strategy
        strategy = self._get_preprocessing_strategy(df, target_column, analysis, metadata)
        
        self.logger.info(f"Preprocessing strategy: {strategy.get('steps', [])}")
        
        # Apply preprocessing steps
        df = self._apply_preprocessing(df, target_column, strategy)
        
        # Track metrics after preprocessing
        self._track_preprocessing_metrics(df)
        
        # Track decision confidence based on strategy
        self.track_decision("preprocessing_complete", 0.85)
        
        return {
            'data': df,
            'strategy': strategy,
            'scalers': self.scalers,
            'encoders': self.encoders,
            'imputers': self.imputers,
            'feature_names': list(df.columns)
        }
    
    def _get_preprocessing_strategy(
        self,
        df: pd.DataFrame,
        target_column: Optional[str],
        analysis: str,
        metadata: Dict
    ) -> Dict[str, Any]:
        """
        Use Gemini to determine optimal preprocessing strategy.
        
        Args:
            df: DataFrame to preprocess
            target_column: Target variable name
            analysis: Previous AI analysis
            metadata: Data metadata
            
        Returns:
            Preprocessing strategy dict
        """
        # Analyze current state
        missing_info = df.isnull().sum()
        missing_cols = missing_info[missing_info > 0].to_dict()
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
        
        # Remove target from feature columns
        if target_column:
            if target_column in numeric_cols:
                numeric_cols.remove(target_column)
            if target_column in categorical_cols:
                categorical_cols.remove(target_column)
        
        prompt = f"""
        Determine the optimal preprocessing strategy for this dataset.

        Current State:
        - Shape: {df.shape}
        - Numeric columns: {numeric_cols}
        - Categorical columns: {categorical_cols}
        - Columns with missing values: {missing_cols}
        - Target column: {target_column}
        
        Previous Analysis:
        {analysis[:1000] if analysis else 'No previous analysis available'}

        Return a JSON object with this exact structure:
        {{
            "steps": [
                "step_name"
            ],
            "missing_value_strategy": {{
                "numeric": "mean|median|mode|drop",
                "categorical": "mode|most_frequent|drop"
            }},
            "encoding_strategy": {{
                "method": "onehot|label|target",
                "columns": ["col1", "col2"]
            }},
            "scaling_strategy": {{
                "method": "standard|minmax|robust",
                "columns": ["col1", "col2"]
            }},
            "feature_engineering": [
                {{"type": "interaction|polynomial|binning", "columns": ["col1", "col2"]}}
            ],
            "columns_to_drop": ["col_to_drop"]
        }}

        Only include steps that are actually needed based on the data.
        """
        
        try:
            strategy = self.call_gemini_json(prompt, temperature=0.3)
            
            # Validate and fix strategy
            strategy = self._validate_strategy(strategy, df, numeric_cols, categorical_cols)
            
            return strategy
            
        except Exception as e:
            self.logger.warning(f"AI strategy failed: {e}, using default strategy")
            return self._get_default_strategy(df, target_column, numeric_cols, categorical_cols, missing_cols)
    
    def _validate_strategy(
        self,
        strategy: Dict,
        df: pd.DataFrame,
        numeric_cols: List[str],
        categorical_cols: List[str]
    ) -> Dict:
        """Validate and fix the preprocessing strategy."""
        # Ensure required keys exist
        if 'steps' not in strategy:
            strategy['steps'] = []
        
        # Validate encoding columns
        if 'encoding_strategy' in strategy:
            valid_cols = [c for c in strategy['encoding_strategy'].get('columns', []) if c in categorical_cols]
            strategy['encoding_strategy']['columns'] = valid_cols if valid_cols else categorical_cols
        
        # Validate scaling columns
        if 'scaling_strategy' in strategy:
            valid_cols = [c for c in strategy['scaling_strategy'].get('columns', []) if c in numeric_cols]
            strategy['scaling_strategy']['columns'] = valid_cols if valid_cols else numeric_cols
        
        # Validate columns to drop
        if 'columns_to_drop' in strategy:
            valid_cols = [c for c in strategy['columns_to_drop'] if c in df.columns]
            strategy['columns_to_drop'] = valid_cols
        
        return strategy
    
    def _get_default_strategy(
        self,
        df: pd.DataFrame,
        target_column: Optional[str],
        numeric_cols: List[str],
        categorical_cols: List[str],
        missing_cols: Dict
    ) -> Dict[str, Any]:
        """Get default preprocessing strategy when AI fails."""
        steps = []
        
        if missing_cols:
            steps.append("handle_missing_values")
        
        # Check if any categorical columns need encoding (including target)
        target_is_categorical = (target_column and target_column in df.columns and 
                                 pd.api.types.is_string_dtype(df[target_column]) or 
                                 (target_column and target_column in df.columns and df[target_column].dtype.name == 'category'))
        if categorical_cols or target_is_categorical:
            steps.append("encode_categorical")
        if numeric_cols:
            steps.append("scale_numeric")
        
        return {
            'steps': steps,
            'missing_value_strategy': {
                'numeric': 'median',
                'categorical': 'most_frequent'
            },
            'encoding_strategy': {
                'method': 'onehot' if len(categorical_cols) <= 5 else 'label',
                'columns': categorical_cols
            },
            'scaling_strategy': {
                'method': 'standard',
                'columns': numeric_cols
            },
            'feature_engineering': [],
            'columns_to_drop': []
        }
    
    def _apply_preprocessing(
        self,
        df: pd.DataFrame,
        target_column: Optional[str],
        strategy: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Apply preprocessing steps based on strategy.
        
        Args:
            df: DataFrame to preprocess
            target_column: Target variable name
            strategy: Preprocessing strategy
            
        Returns:
            Preprocessed DataFrame
        """
        # Store original row count
        original_rows = len(df)
        
        # 1. Drop specified columns
        columns_to_drop = strategy.get('columns_to_drop', [])
        if columns_to_drop:
            valid_drops = [c for c in columns_to_drop if c in df.columns and c != target_column]
            if valid_drops:
                df = df.drop(columns=valid_drops)
                self.logger.info(f"Dropped columns: {valid_drops}")
        
        # 2. Handle missing values
        if 'handle_missing_values' in strategy.get('steps', []) or df.isnull().sum().sum() > 0:
            df = self._handle_missing_values(df, target_column, strategy.get('missing_value_strategy', {}))
        
        # 3. Encode categorical variables
        if 'encode_categorical' in strategy.get('steps', []):
            df = self._encode_categorical(df, target_column, strategy.get('encoding_strategy', {}))
        
        # 4. Scale numeric features
        if 'scale_numeric' in strategy.get('steps', []):
            df = self._scale_numeric(df, target_column, strategy.get('scaling_strategy', {}))
        
        # 5. Feature engineering
        if strategy.get('feature_engineering'):
            df = self._apply_feature_engineering(df, target_column, strategy['feature_engineering'])
        
        # Log changes
        self.logger.info(
            f"Preprocessing complete: {original_rows} rows -> {len(df)} rows, "
            f"{len(df.columns)} columns"
        )
        
        return df
    
    def _handle_missing_values(
        self,
        df: pd.DataFrame,
        target_column: Optional[str],
        strategy: Dict[str, str]
    ) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        numeric_strategy = strategy.get('numeric', 'median')
        categorical_strategy = strategy.get('categorical', 'most_frequent')
        
        # Numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_column and target_column in numeric_cols:
            numeric_cols.remove(target_column)
        
        if numeric_cols:
            if numeric_strategy == 'drop':
                df = df.dropna(subset=numeric_cols)
            else:
                imputer = SimpleImputer(strategy=numeric_strategy)
                df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
                self.imputers['numeric'] = imputer
        
        # Categorical columns
        categorical_cols = df.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
        if target_column and target_column in categorical_cols:
            categorical_cols.remove(target_column)
        
        if categorical_cols:
            if categorical_strategy == 'drop':
                df = df.dropna(subset=categorical_cols)
            else:
                for col in categorical_cols:
                    if df[col].isnull().sum() > 0:
                        mode_value = df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown'
                        df[col] = df[col].fillna(mode_value)
        
        # Handle target column missing values
        if target_column and target_column in df.columns:
            df = df.dropna(subset=[target_column])
        
        self.logger.info(f"Missing values handled: numeric={numeric_strategy}, categorical={categorical_strategy}")
        
        return df
    
    def _encode_categorical(
        self,
        df: pd.DataFrame,
        target_column: Optional[str],
        strategy: Dict[str, Any]
    ) -> pd.DataFrame:
        """Encode categorical variables."""
        method = strategy.get('method', 'onehot')
        columns = strategy.get('columns', [])
        
        # Get categorical columns if not specified
        if not columns:
            columns = df.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
        
        # Remove target from encoding
        if target_column and target_column in columns:
            columns.remove(target_column)
        
        # Always encode target if it's categorical (before early return)
        if target_column and target_column in df.columns:
            if pd.api.types.is_string_dtype(df[target_column]) or df[target_column].dtype.name == 'category':
                le = LabelEncoder()
                df[target_column] = le.fit_transform(df[target_column].astype(str))
                self.encoders[target_column] = le
                self.logger.info(f"Encoded target column with classes: {list(le.classes_)}")
        
        if not columns:
            return df
        
        self.logger.info(f"Encoding {len(columns)} categorical columns using {method}")
        
        if method == 'onehot':
            # Use pandas get_dummies for simplicity
            df = pd.get_dummies(df, columns=columns, drop_first=True)
        
        elif method == 'label':
            for col in columns:
                if col in df.columns:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    self.encoders[col] = le
        
        return df
    
    def _scale_numeric(
        self,
        df: pd.DataFrame,
        target_column: Optional[str],
        strategy: Dict[str, Any]
    ) -> pd.DataFrame:
        """Scale numeric features."""
        method = strategy.get('method', 'standard')
        columns = strategy.get('columns', [])
        
        # Get numeric columns if not specified
        if not columns:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target from scaling
        if target_column and target_column in columns:
            columns.remove(target_column)
        
        if not columns:
            return df
        
        self.logger.info(f"Scaling {len(columns)} numeric columns using {method}")
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
        elif method == 'robust':
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()
        
        # Apply scaling
        df[columns] = scaler.fit_transform(df[columns])
        self.scalers['main'] = scaler
        self.scalers['columns'] = columns
        
        return df
    
    def _apply_feature_engineering(
        self,
        df: pd.DataFrame,
        target_column: Optional[str],
        engineering_steps: List[Dict]
    ) -> pd.DataFrame:
        """Apply feature engineering steps."""
        for step in engineering_steps:
            step_type = step.get('type', '')
            columns = step.get('columns', [])
            
            # Validate columns exist
            valid_cols = [c for c in columns if c in df.columns]
            if len(valid_cols) < 2:
                continue
            
            if step_type == 'interaction' and len(valid_cols) >= 2:
                # Create interaction features
                col1, col2 = valid_cols[0], valid_cols[1]
                if df[col1].dtype in [np.float64, np.int64] and df[col2].dtype in [np.float64, np.int64]:
                    df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
                    self.logger.info(f"Created interaction feature: {col1}_x_{col2}")
            
            elif step_type == 'polynomial':
                # Create polynomial features
                for col in valid_cols:
                    if df[col].dtype in [np.float64, np.int64]:
                        df[f'{col}_squared'] = df[col] ** 2
                        self.logger.info(f"Created polynomial feature: {col}_squared")
            
            elif step_type == 'binning':
                # Create binned features
                for col in valid_cols:
                    if df[col].dtype in [np.float64, np.int64]:
                        df[f'{col}_binned'] = pd.qcut(df[col], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
                        self.logger.info(f"Created binned feature: {col}_binned")
        
        return df
    
    def _track_preprocessing_metrics(self, df: pd.DataFrame):
        """Track preprocessing metrics."""
        missing_pct = df.isnull().sum().sum() / df.size * 100
        
        self.logger.info(
            "Preprocessing metrics",
            extra={
                'rows': len(df),
                'columns': len(df.columns),
                'missing_pct': missing_pct,
                'scalers_count': len(self.scalers),
                'encoders_count': len(self.encoders)
            }
        )
