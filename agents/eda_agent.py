"""
EDAAgent - Exploratory Data Analysis with AI-powered insights
Generates visualizations, statistics, and data quality reports
"""

import os
import base64
from io import BytesIO
from typing import Any, Dict, List, Optional
import json

import pandas as pd
import numpy as np

# Visualization imports
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from agents.base_agent import BaseAgent


class EDAAgent(BaseAgent):
    """
    Agent responsible for Exploratory Data Analysis.
    Features:
    - Statistical summaries
    - Data quality assessment
    - Correlation analysis
    - Distribution visualizations
    - Outlier detection
    - AI-powered insights
    """
    
    def __init__(self):
        super().__init__("EDA")
        self.figures = []
        
    def _run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive EDA on the data.
        
        Args:
            input_data: Dict containing:
                - data: DataFrame to analyze
                - target_column: Name of target variable
                - metadata: Data metadata from ingestion
                
        Returns:
            Dict with EDA results, visualizations, and insights
        """
        df = input_data.get('data')
        if df is None:
            raise ValueError("No data provided for EDA")
        
        target_column = input_data.get('target_column')
        
        self.logger.info(f"Starting EDA for data with shape: {df.shape}")
        
        # 1. Basic Statistics
        basic_stats = self._get_basic_statistics(df)
        
        # 2. Data Quality Assessment
        quality_report = self._assess_data_quality(df)
        
        # 3. Column Analysis
        column_analysis = self._analyze_columns(df, target_column)
        
        # 4. Correlation Analysis
        correlations = self._analyze_correlations(df, target_column)
        
        # 5. Distribution Analysis
        distributions = self._analyze_distributions(df, target_column)
        
        # 6. Outlier Detection
        outliers = self._detect_outliers(df)
        
        # 7. Generate Visualizations
        visualizations = self._generate_visualizations(df, target_column, correlations)
        
        # 8. AI-Powered Insights
        insights = self._generate_ai_insights(
            df, target_column, basic_stats, quality_report, 
            correlations, outliers
        )
        
        # Track EDA metrics
        self._track_eda_metrics(df, quality_report)
        
        self.track_decision("eda_complete", 0.9)
        
        return {
            'basic_stats': basic_stats,
            'quality_report': quality_report,
            'column_analysis': column_analysis,
            'correlations': correlations,
            'distributions': distributions,
            'outliers': outliers,
            'visualizations': visualizations,
            'insights': insights,
            'data': df,
            'target_column': target_column
        }
    
    def _get_basic_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get basic dataset statistics."""
        stats = {
            'shape': {'rows': df.shape[0], 'columns': df.shape[1]},
            'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(df.select_dtypes(include=['object', 'category', 'string']).columns),
            'datetime_columns': len(df.select_dtypes(include=['datetime64']).columns),
        }
        
        # Numerical summary
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            stats['numerical_summary'] = numeric_df.describe().to_dict()
        
        # Categorical summary
        categorical_df = df.select_dtypes(include=['object', 'category', 'string'])
        if not categorical_df.empty:
            stats['categorical_summary'] = {
                col: {
                    'unique': df[col].nunique(),
                    'top_values': df[col].value_counts().head(5).to_dict()
                }
                for col in categorical_df.columns
            }
        
        return stats
    
    def _assess_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess data quality issues."""
        quality = {
            'missing_values': {},
            'duplicates': {},
            'data_types': {},
            'overall_score': 100.0
        }
        
        # Missing values analysis
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        quality['missing_values'] = {
            'total_missing': int(missing.sum()),
            'columns_with_missing': int((missing > 0).sum()),
            'by_column': {col: {'count': int(missing[col]), 'percentage': float(missing_pct[col])} 
                         for col in df.columns if missing[col] > 0}
        }
        
        # Penalize for missing values
        if quality['missing_values']['total_missing'] > 0:
            quality['overall_score'] -= min(20, missing.sum() / len(df) * 100)
        
        # Duplicate analysis
        duplicate_rows = df.duplicated().sum()
        quality['duplicates'] = {
            'duplicate_rows': int(duplicate_rows),
            'duplicate_percentage': round(duplicate_rows / len(df) * 100, 2)
        }
        
        # Penalize for duplicates
        if duplicate_rows > 0:
            quality['overall_score'] -= min(10, duplicate_rows / len(df) * 50)
        
        # Data type consistency
        quality['data_types'] = {
            'mixed_types': self._check_mixed_types(df)
        }
        
        quality['overall_score'] = max(0, round(quality['overall_score'], 1))
        
        return quality
    
    def _check_mixed_types(self, df: pd.DataFrame) -> List[str]:
        """Check for columns with mixed data types."""
        mixed_cols = []
        for col in df.columns:
            if pd.api.types.is_string_dtype(df[col]):
                # Check if numeric values are mixed with strings
                numeric_count = pd.to_numeric(df[col], errors='coerce').notna().sum()
                if 0 < numeric_count < len(df[col].dropna()):
                    mixed_cols.append(col)
        return mixed_cols
    
    def _analyze_columns(self, df: pd.DataFrame, target_column: Optional[str]) -> Dict[str, Any]:
        """Analyze each column in detail."""
        analysis = {}
        
        for col in df.columns:
            col_analysis = {
                'dtype': str(df[col].dtype),
                'unique_values': int(df[col].nunique()),
                'missing_count': int(df[col].isnull().sum()),
                'is_target': col == target_column
            }
            
            if pd.api.types.is_numeric_dtype(df[col]):
                col_analysis['type'] = 'numeric'
                col_analysis['stats'] = {
                    'mean': float(df[col].mean()) if not df[col].isnull().all() else None,
                    'std': float(df[col].std()) if not df[col].isnull().all() else None,
                    'min': float(df[col].min()) if not df[col].isnull().all() else None,
                    'max': float(df[col].max()) if not df[col].isnull().all() else None,
                    'median': float(df[col].median()) if not df[col].isnull().all() else None,
                    'skewness': float(df[col].skew()) if not df[col].isnull().all() else None,
                }
            else:
                col_analysis['type'] = 'categorical'
                col_analysis['top_values'] = df[col].value_counts().head(5).to_dict()
            
            analysis[col] = col_analysis
        
        return analysis
    
    def _analyze_correlations(self, df: pd.DataFrame, target_column: Optional[str]) -> Dict[str, Any]:
        """Analyze correlations between features."""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty or len(numeric_df.columns) < 2:
            return {'message': 'Not enough numeric columns for correlation analysis'}
        
        # Full correlation matrix
        corr_matrix = numeric_df.corr()
        
        result = {
            'correlation_matrix': corr_matrix.round(3).to_dict(),
            'high_correlations': [],
            'target_correlations': {}
        }
        
        # Find highly correlated pairs (excluding self-correlation)
        for i, col1 in enumerate(corr_matrix.columns):
            for j, col2 in enumerate(corr_matrix.columns):
                if i < j:
                    corr_val = corr_matrix.loc[col1, col2]
                    if abs(corr_val) > 0.7:
                        result['high_correlations'].append({
                            'feature1': col1,
                            'feature2': col2,
                            'correlation': round(corr_val, 3)
                        })
        
        # Target correlations
        if target_column and target_column in numeric_df.columns:
            target_corr = corr_matrix[target_column].drop(target_column).sort_values(key=abs, ascending=False)
            result['target_correlations'] = target_corr.round(3).to_dict()
        
        return result
    
    def _analyze_distributions(self, df: pd.DataFrame, target_column: Optional[str]) -> Dict[str, Any]:
        """Analyze distributions of features."""
        distributions = {}
        
        numeric_df = df.select_dtypes(include=[np.number])
        
        for col in numeric_df.columns:
            data = df[col].dropna()
            if len(data) == 0:
                continue
                
            distributions[col] = {
                'skewness': float(data.skew()),
                'kurtosis': float(data.kurtosis()),
                'is_normal': abs(data.skew()) < 0.5 and abs(data.kurtosis()) < 3,
                'percentiles': {
                    '5%': float(data.quantile(0.05)),
                    '25%': float(data.quantile(0.25)),
                    '50%': float(data.quantile(0.50)),
                    '75%': float(data.quantile(0.75)),
                    '95%': float(data.quantile(0.95)),
                }
            }
        
        # Target distribution for classification
        if target_column and target_column in df.columns:
            if pd.api.types.is_string_dtype(df[target_column]) or df[target_column].nunique() < 20:
                value_counts = df[target_column].value_counts()
                distributions['target_distribution'] = {
                    'values': value_counts.to_dict(),
                    'is_balanced': value_counts.min() / value_counts.max() > 0.5
                }
        
        return distributions
    
    def _detect_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect outliers using IQR method."""
        outliers = {}
        numeric_df = df.select_dtypes(include=[np.number])
        
        for col in numeric_df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            outlier_count = outlier_mask.sum()
            
            if outlier_count > 0:
                outliers[col] = {
                    'count': int(outlier_count),
                    'percentage': round(outlier_count / len(df) * 100, 2),
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound)
                }
        
        return {
            'outliers_by_column': outliers,
            'total_columns_with_outliers': len(outliers)
        }
    
    def _generate_visualizations(
        self, 
        df: pd.DataFrame, 
        target_column: Optional[str],
        correlations: Dict
    ) -> Dict[str, str]:
        """Generate visualization images as base64 strings."""
        visualizations = {}
        
        try:
            # 1. Correlation Heatmap
            numeric_df = df.select_dtypes(include=[np.number])
            if len(numeric_df.columns) >= 2:
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', center=0, ax=ax)
                ax.set_title('Feature Correlation Heatmap')
                plt.tight_layout()
                visualizations['correlation_heatmap'] = self._fig_to_base64(fig)
                plt.close(fig)
            
            # 2. Missing Values Chart
            missing = df.isnull().sum()
            if missing.sum() > 0:
                fig, ax = plt.subplots(figsize=(10, 6))
                missing[missing > 0].plot(kind='bar', ax=ax, color='coral')
                ax.set_title('Missing Values by Column')
                ax.set_ylabel('Count')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                visualizations['missing_values'] = self._fig_to_base64(fig)
                plt.close(fig)
            
            # 3. Distribution plots for numeric columns (top 6)
            numeric_cols = numeric_df.columns[:6]
            if len(numeric_cols) > 0:
                n_cols = min(3, len(numeric_cols))
                n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
                axes = np.array(axes).flatten() if len(numeric_cols) > 1 else [axes]
                
                for i, col in enumerate(numeric_cols):
                    sns.histplot(df[col].dropna(), kde=True, ax=axes[i], color='steelblue')
                    axes[i].set_title(f'Distribution: {col}')
                
                # Hide empty subplots
                for i in range(len(numeric_cols), len(axes)):
                    axes[i].set_visible(False)
                
                plt.tight_layout()
                visualizations['distributions'] = self._fig_to_base64(fig)
                plt.close(fig)
            
            # 4. Target Distribution (for classification)
            if target_column and target_column in df.columns:
                if df[target_column].nunique() < 20:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    df[target_column].value_counts().plot(kind='bar', ax=ax, color='teal')
                    ax.set_title(f'Target Distribution: {target_column}')
                    ax.set_ylabel('Count')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    visualizations['target_distribution'] = self._fig_to_base64(fig)
                    plt.close(fig)
            
            # 5. Box plots for outlier visualization (top 6 numeric columns)
            if len(numeric_cols) > 0:
                fig, ax = plt.subplots(figsize=(12, 6))
                df[numeric_cols].boxplot(ax=ax)
                ax.set_title('Box Plots (Outlier Detection)')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                visualizations['boxplots'] = self._fig_to_base64(fig)
                plt.close(fig)
        
        except Exception as e:
            self.logger.error(f"Error generating visualizations: {e}")
            visualizations['error'] = str(e)
        
        return visualizations
    
    def _fig_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string."""
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    
    def _generate_ai_insights(
        self,
        df: pd.DataFrame,
        target_column: Optional[str],
        basic_stats: Dict,
        quality_report: Dict,
        correlations: Dict,
        outliers: Dict
    ) -> Dict[str, Any]:
        """Generate AI-powered insights about the data."""
        prompt = f"""Analyze this dataset and provide actionable insights:

Dataset Overview:
- Shape: {basic_stats['shape']}
- Numeric columns: {basic_stats['numeric_columns']}
- Categorical columns: {basic_stats['categorical_columns']}

Data Quality:
- Overall Score: {quality_report['overall_score']}%
- Missing Values: {quality_report['missing_values']['total_missing']} total
- Duplicates: {quality_report['duplicates']['duplicate_rows']} rows

Target Column: {target_column}

High Correlations: {correlations.get('high_correlations', [])}

Outliers: {outliers.get('total_columns_with_outliers', 0)} columns with outliers

Provide:
1. Key findings (3-5 bullet points)
2. Data quality issues to address
3. Feature engineering suggestions
4. Preprocessing recommendations
5. Potential challenges for modeling

Format as JSON with keys: key_findings, quality_issues, feature_suggestions, preprocessing_recommendations, challenges"""

        try:
            response = self.call_gemini(prompt, temperature=0.4)
            
            # Try to parse as JSON
            try:
                insights = json.loads(response)
            except json.JSONDecodeError:
                insights = {
                    'raw_insights': response,
                    'key_findings': [
                        f"Dataset has {basic_stats['shape']['rows']} rows and {basic_stats['shape']['columns']} columns",
                        f"Data quality score: {quality_report['overall_score']}%",
                        f"Found {outliers.get('total_columns_with_outliers', 0)} columns with outliers"
                    ]
                }
            
            return insights
            
        except Exception as e:
            self.logger.warning(f"AI insights failed: {e}")
            return {
                'key_findings': [
                    f"Dataset has {basic_stats['shape']['rows']} rows and {basic_stats['shape']['columns']} columns",
                    f"Data quality score: {quality_report['overall_score']}%",
                    f"{basic_stats['numeric_columns']} numeric and {basic_stats['categorical_columns']} categorical features",
                    f"Found {quality_report['missing_values']['total_missing']} missing values",
                    f"{outliers.get('total_columns_with_outliers', 0)} columns have outliers"
                ],
                'quality_issues': list(quality_report['missing_values']['by_column'].keys()) if quality_report['missing_values']['by_column'] else [],
                'preprocessing_recommendations': [
                    "Handle missing values before training",
                    "Consider scaling numeric features",
                    "Encode categorical variables"
                ]
            }
    
    def _track_eda_metrics(self, df: pd.DataFrame, quality_report: Dict):
        """Track EDA metrics."""
        try:
            self.logger.info(
                "EDA metrics",
                extra={
                    'data_rows': df.shape[0],
                    'data_columns': df.shape[1],
                    'quality_score': quality_report['overall_score'],
                    'missing_values': quality_report['missing_values']['total_missing'],
                    'duplicates': quality_report['duplicates']['duplicate_rows']
                }
            )
        except Exception as e:
            self.logger.warning(f"Failed to track EDA metrics: {e}")
