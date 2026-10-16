"""
DataIngestionAgent - Handles data loading, validation, and initial analysis
Supports multiple file formats with AI-powered data profiling
"""

import os
from typing import Any, Dict, Optional
from io import StringIO

import pandas as pd
import numpy as np

from agents.base_agent import BaseAgent


class DataIngestionAgent(BaseAgent):
    """
    Agent responsible for ingesting and profiling data.
    Features:
    - Multiple format support (CSV, Excel, Parquet, JSON)
    - AI-powered data analysis using Gemini
    - Automatic data quality metrics
    - Sampling for large datasets
    """
    
    SUPPORTED_FORMATS = {
        '.csv': 'csv',
        '.xlsx': 'excel',
        '.xls': 'excel',
        '.parquet': 'parquet',
        '.json': 'json'
    }
    
    def __init__(self):
        super().__init__("DataIngestion")
        self.max_sample_size = 10000
    
    def _run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ingest and analyze data from various sources.
        
        Args:
            input_data: Dict containing:
                - source: Path to data file or DataFrame
                - target_column: Name of target variable
                - sample_size: Optional max rows to load
                
        Returns:
            Dict with data, metadata, and AI analysis
        """
        source = input_data.get('source')
        target_column = input_data.get('target_column')
        sample_size = input_data.get('sample_size', self.max_sample_size)
        
        if source is None:
            raise ValueError("Data source is required")
        
        # Load data
        df = self._load_data(source, sample_size)
        
        self.logger.info(f"Loaded data with shape: {df.shape}")
        
        # Validate target column (case-insensitive)
        if target_column:
            # Try exact match first
            if target_column not in df.columns:
                # Try case-insensitive match
                columns_lower = {col.lower(): col for col in df.columns}
                if target_column.lower() in columns_lower:
                    # Update target column to match actual case
                    target_column = columns_lower[target_column.lower()]
                    input_data['target_column'] = target_column
                    self.logger.info(f"Target column matched (case-insensitive): '{target_column}'")
                else:
                    raise ValueError(f"Target column '{target_column}' not found in data. Available: {list(df.columns)}")
        
        # Track data quality metrics
        self._track_data_metrics(df, 'ingestion')
        
        # Get metadata
        metadata = self._extract_metadata(df, target_column)
        
        # AI-powered analysis
        analysis = self._analyze_data(df, target_column)
        
        # Track decision confidence
        self.track_decision("data_ingestion_complete", 0.9)
        
        return {
            'data': df,
            'metadata': metadata,
            'analysis': analysis,
            'target_column': target_column
        }
    
    def _load_data(self, source: Any, sample_size: int) -> pd.DataFrame:
        """
        Load data from various sources.
        
        Args:
            source: File path, URL, or DataFrame
            sample_size: Maximum rows to load
            
        Returns:
            Pandas DataFrame
        """
        # If already a DataFrame, return it
        if isinstance(source, pd.DataFrame):
            df = source
            if len(df) > sample_size:
                df = df.sample(n=sample_size, random_state=42)
            return df
        
        # String source (file path or URL)
        if isinstance(source, str):
            # Detect format
            ext = os.path.splitext(source.lower())[1]
            format_type = self.SUPPORTED_FORMATS.get(ext)
            
            if not format_type:
                raise ValueError(f"Unsupported file format: {ext}. Supported: {list(self.SUPPORTED_FORMATS.keys())}")
            
            self.logger.info(f"Loading {format_type} file: {source}")
            
            try:
                if format_type == 'csv':
                    df = pd.read_csv(source, nrows=sample_size)
                elif format_type == 'excel':
                    df = pd.read_excel(source, nrows=sample_size)
                elif format_type == 'parquet':
                    df = pd.read_parquet(source)
                    if len(df) > sample_size:
                        df = df.head(sample_size)
                elif format_type == 'json':
                    df = pd.read_json(source)
                    if len(df) > sample_size:
                        df = df.head(sample_size)
                else:
                    raise ValueError(f"Unknown format type: {format_type}")
                    
            except FileNotFoundError:
                raise FileNotFoundError(f"Data file not found: {source}")
            except Exception as e:
                raise ValueError(f"Error loading data: {str(e)}")
            
            return df
        
        raise ValueError(f"Invalid source type: {type(source)}")
    
    def _extract_metadata(self, df: pd.DataFrame, target_column: Optional[str]) -> Dict[str, Any]:
        """
        Extract comprehensive metadata from DataFrame.
        
        Args:
            df: Input DataFrame
            target_column: Name of target variable
            
        Returns:
            Dict with metadata
        """
        # Basic info
        metadata = {
            'shape': df.shape,
            'rows': len(df),
            'columns': len(df.columns),
            'column_names': list(df.columns),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
        }
        
        # Data types breakdown
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        metadata['column_types'] = {
            'numeric': numeric_cols,
            'categorical': categorical_cols,
            'datetime': datetime_cols
        }
        
        # Missing values
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        metadata['missing_values'] = {
            'counts': missing.to_dict(),
            'percentages': missing_pct.to_dict(),
            'total_missing': int(missing.sum()),
            'total_missing_pct': round(missing.sum() / df.size * 100, 2)
        }
        
        # Target variable info
        if target_column and target_column in df.columns:
            target_series = df[target_column]
            metadata['target'] = {
                'name': target_column,
                'dtype': str(target_series.dtype),
                'unique_values': int(target_series.nunique()),
                'missing': int(target_series.isnull().sum())
            }
            
            # Determine problem type
            if target_series.dtype in ['object', 'category'] or target_series.nunique() <= 20:
                metadata['target']['problem_type'] = 'classification'
                metadata['target']['class_distribution'] = target_series.value_counts().to_dict()
            else:
                metadata['target']['problem_type'] = 'regression'
                metadata['target']['statistics'] = {
                    'mean': float(target_series.mean()),
                    'std': float(target_series.std()),
                    'min': float(target_series.min()),
                    'max': float(target_series.max())
                }
        
        # Numeric column statistics
        if numeric_cols:
            metadata['numeric_stats'] = df[numeric_cols].describe().to_dict()
        
        return metadata
    
    def _track_data_metrics(self, df: pd.DataFrame, stage: str):
        """
        Track data quality metrics.
        
        Args:
            df: DataFrame to analyze
            stage: Pipeline stage name
        """
        # Basic metrics
        missing_pct = df.isnull().sum().sum() / df.size * 100
        memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        duplicate_pct = df.duplicated().sum() / len(df) * 100
        
        self.logger.info(
            f"Data metrics tracked",
            extra={
                'stage': stage,
                'rows': len(df),
                'columns': len(df.columns),
                'missing_pct': missing_pct,
                'memory_mb': memory_mb,
                'duplicate_pct': duplicate_pct
            }
        )
    
    def _analyze_data(self, df: pd.DataFrame, target_column: Optional[str]) -> str:
        """
        Use Gemini to analyze data characteristics and provide recommendations.
        
        Args:
            df: DataFrame to analyze
            target_column: Name of target variable
            
        Returns:
            AI analysis as string
        """
        # Prepare data sample for analysis
        sample = df.head(10).to_string()
        
        # Get statistics
        stats_info = {
            'shape': df.shape,
            'dtypes': {str(k): str(v) for k, v in df.dtypes.items()},
            'missing': df.isnull().sum().to_dict(),
            'numeric_stats': df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {}
        }
        
        prompt = f"""
        Analyze this dataset and provide insights for machine learning:

        Dataset Sample (first 10 rows):
        {sample}

        Dataset Statistics:
        - Shape: {stats_info['shape']}
        - Column types: {stats_info['dtypes']}
        - Missing values: {stats_info['missing']}
        - Target column: {target_column or 'Not specified'}

        Provide a concise analysis covering:
        1. Data Quality Assessment
           - Missing value patterns
           - Potential data quality issues
           - Outlier indicators
        
        2. Feature Analysis
           - Important features for prediction
           - Potential feature engineering opportunities
           - Columns that may need transformation
        
        3. Preprocessing Recommendations
           - Specific steps needed before modeling
           - Encoding strategies for categorical variables
           - Scaling recommendations
        
        4. Modeling Considerations
           - Suggested problem type (classification/regression)
           - Potential challenges for modeling
           - Recommended approaches

        Be specific and actionable in your recommendations.
        """
        
        try:
            analysis = self.call_gemini(prompt, temperature=0.4)
            return analysis
        except Exception as e:
            self.logger.warning(f"AI analysis failed: {e}, using basic analysis")
            return self._get_basic_analysis(df, target_column)
    
    def _get_basic_analysis(self, df: pd.DataFrame, target_column: Optional[str]) -> str:
        """
        Generate basic analysis when AI is unavailable.
        
        Args:
            df: DataFrame to analyze
            target_column: Name of target variable
            
        Returns:
            Basic analysis string
        """
        analysis = []
        
        # Missing values
        missing = df.isnull().sum()
        if missing.sum() > 0:
            high_missing = missing[missing > len(df) * 0.1]
            if len(high_missing) > 0:
                analysis.append(f"⚠️ High missing values in: {list(high_missing.index)}")
            analysis.append(f"📊 Total missing: {missing.sum()} ({missing.sum()/df.size*100:.1f}%)")
        
        # Data types
        numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
        categorical_cols = len(df.select_dtypes(include=['object', 'category', 'string']).columns)
        analysis.append(f"📈 Numeric columns: {numeric_cols}, Categorical: {categorical_cols}")
        
        # Target info
        if target_column and target_column in df.columns:
            target = df[target_column]
            if target.dtype in ['object', 'category'] or target.nunique() <= 20:
                analysis.append(f"🎯 Classification target with {target.nunique()} classes")
            else:
                analysis.append(f"🎯 Regression target: mean={target.mean():.2f}, std={target.std():.2f}")
        
        # Recommendations
        analysis.append("\n📋 Recommendations:")
        if missing.sum() > 0:
            analysis.append("  - Handle missing values (imputation or removal)")
        if categorical_cols > 0:
            analysis.append("  - Encode categorical variables")
        if numeric_cols > 0:
            analysis.append("  - Consider feature scaling for numeric columns")
        
        return "\n".join(analysis)
