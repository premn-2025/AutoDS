"""
ForecastingAgent - Time Series Forecasting Agent
Handles time series analysis and forecasting using various algorithms
"""

import time
import json
from typing import Any, Dict, List, Optional, Tuple
import pickle
import os

import pandas as pd
import numpy as np

# Sklearn imports
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

# Try importing specialized time series libraries
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller, acf, pacf
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

from agents.base_agent import BaseAgent


class ForecastingAgent(BaseAgent):
    """
    Agent responsible for time series forecasting.
    Features:
    - Multiple forecasting algorithms
    - Automatic seasonality detection
    - Trend analysis
    - Forecast visualization data
    """
    
    FORECASTING_MODELS = ['exponential_smoothing', 'arima', 'prophet', 'linear_trend', 'moving_average']
    
    def __init__(self):
        super().__init__("Forecasting")
        self.models = {}
        self.forecasts = {}
    
    def _run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform time series forecasting.
        
        Args:
            input_data: Dict containing:
                - data: DataFrame with time series data
                - date_column: Column containing dates
                - target_column: Column to forecast
                - forecast_periods: Number of periods to forecast
                - frequency: Data frequency ('D', 'W', 'M', 'H', etc.)
                
        Returns:
            Dict with forecasts, model metrics, and visualization data
        """
        df = input_data.get('data')
        if df is None:
            raise ValueError("No data provided for forecasting")
        
        date_column = input_data.get('date_column')
        target_column = input_data.get('target_column')
        forecast_periods = input_data.get('forecast_periods', 30)
        frequency = input_data.get('frequency', 'D')
        # Map legacy pandas aliases (removed in pandas 3) to their current names
        try:
            pd.tseries.frequencies.to_offset(frequency)
        except ValueError:
            frequency = {'M': 'ME', 'H': 'h', 'Y': 'YE', 'A': 'YE', 'Q': 'QE', 'T': 'min', 'S': 's'}.get(frequency, frequency)
        selected_models = input_data.get('models', ['exponential_smoothing', 'moving_average'])
        
        self.logger.info(f"Starting forecasting for {target_column}, {forecast_periods} periods ahead")
        
        # Prepare time series data
        ts_data = self._prepare_time_series(df, date_column, target_column, frequency)
        
        # Analyze time series characteristics
        ts_analysis = self._analyze_time_series(ts_data)
        
        # Train forecasting models
        model_results = {}
        forecasts = {}
        
        for model_name in selected_models:
            try:
                self.logger.info(f"Training {model_name} model...")
                forecast, metrics = self._train_forecast_model(
                    ts_data, model_name, forecast_periods, frequency
                )
                model_results[model_name] = metrics
                forecasts[model_name] = forecast
                
                self.logger.info(f"{model_name} metrics: RMSE={metrics.get('rmse', 0):.4f}, MAE={metrics.get('mae', 0):.4f}")
                
            except Exception as e:
                self.logger.warning(f"Failed to train {model_name}: {e}")
                model_results[model_name] = {'error': str(e)}
        
        # Find best model
        best_model = self._find_best_model(model_results)
        
        # Generate visualization data
        viz_data = self._generate_visualization_data(ts_data, forecasts, best_model)
        
        # Generate AI-powered forecasting insights
        ai_insights = self._generate_forecasting_insights(
            ts_analysis, model_results, best_model, forecast_periods
        )
        
        self.track_decision("forecasting_complete", 0.85)
        
        return {
            'analysis': ts_analysis,
            'model_results': model_results,
            'forecasts': forecasts,
            'best_model': best_model,
            'best_forecast': forecasts.get(best_model, []),
            'visualization': viz_data,
            'forecast_periods': forecast_periods,
            'frequency': frequency,
            'ai_insights': ai_insights
        }
    
    def _prepare_time_series(
        self,
        df: pd.DataFrame,
        date_column: Optional[str],
        target_column: str,
        frequency: str
    ) -> pd.Series:
        """Prepare time series data for forecasting."""
        df = df.copy()
        
        # Find or create date index
        if date_column and date_column in df.columns:
            df[date_column] = pd.to_datetime(df[date_column])
            df = df.sort_values(date_column)
            df = df.set_index(date_column)
        elif df.index.dtype == 'datetime64[ns]':
            df = df.sort_index()
        else:
            # Create a date range index
            df.index = pd.date_range(start='2020-01-01', periods=len(df), freq=frequency)
        
        # Get target series
        ts = df[target_column].copy()
        
        # Handle missing values
        ts = ts.interpolate(method='linear').bfill().ffill()
        
        return ts
    
    def _analyze_time_series(self, ts: pd.Series) -> Dict[str, Any]:
        """Analyze time series characteristics."""
        analysis = {
            'length': len(ts),
            'start_date': str(ts.index[0]) if hasattr(ts.index[0], 'isoformat') else str(ts.index[0]),
            'end_date': str(ts.index[-1]) if hasattr(ts.index[-1], 'isoformat') else str(ts.index[-1]),
            'mean': float(ts.mean()),
            'std': float(ts.std()),
            'min': float(ts.min()),
            'max': float(ts.max()),
            'trend': 'unknown',
            'seasonality': 'unknown',
            'stationarity': 'unknown'
        }
        
        # Trend detection
        if len(ts) > 10:
            first_half = ts[:len(ts)//2].mean()
            second_half = ts[len(ts)//2:].mean()
            if second_half > first_half * 1.1:
                analysis['trend'] = 'upward'
            elif second_half < first_half * 0.9:
                analysis['trend'] = 'downward'
            else:
                analysis['trend'] = 'stable'
        
        # Stationarity test (if statsmodels available)
        if STATSMODELS_AVAILABLE and len(ts) > 20:
            try:
                adf_result = adfuller(ts.dropna())
                analysis['stationarity'] = 'stationary' if adf_result[1] < 0.05 else 'non-stationary'
                analysis['adf_statistic'] = float(adf_result[0])
                analysis['adf_pvalue'] = float(adf_result[1])
            except Exception:
                pass
        
        return analysis
    
    def _train_forecast_model(
        self,
        ts: pd.Series,
        model_name: str,
        forecast_periods: int,
        frequency: str
    ) -> Tuple[List[float], Dict[str, Any]]:
        """Train a specific forecasting model."""
        start_time = time.time()
        
        # Split for validation
        train_size = int(len(ts) * 0.8)
        train, test = ts[:train_size], ts[train_size:]
        
        forecast = []
        fitted_values = []
        
        if model_name == 'moving_average':
            # Simple moving average
            window = min(7, len(train) // 3)
            fitted_values = train.rolling(window=window).mean().dropna().tolist()
            last_ma = train.rolling(window=window).mean().iloc[-1]
            forecast = [last_ma] * forecast_periods
            
        elif model_name == 'linear_trend':
            # Linear regression on time
            from sklearn.linear_model import LinearRegression
            X_train = np.arange(len(train)).reshape(-1, 1)
            X_future = np.arange(len(train), len(train) + forecast_periods).reshape(-1, 1)
            
            model = LinearRegression()
            model.fit(X_train, train.values)
            
            fitted_values = model.predict(X_train).tolist()
            forecast = model.predict(X_future).tolist()
            
        elif model_name == 'exponential_smoothing' and STATSMODELS_AVAILABLE:
            try:
                seasonal_periods = self._detect_seasonality(ts)
                if seasonal_periods and seasonal_periods < len(train) // 2:
                    model = ExponentialSmoothing(
                        train,
                        seasonal_periods=seasonal_periods,
                        trend='add',
                        seasonal='add'
                    )
                else:
                    model = ExponentialSmoothing(train, trend='add')
                
                fitted_model = model.fit()
                fitted_values = fitted_model.fittedvalues.tolist()
                forecast = fitted_model.forecast(forecast_periods).tolist()
            except Exception as e:
                self.logger.warning(f"Exponential smoothing failed: {e}, falling back to linear")
                return self._train_forecast_model(ts, 'linear_trend', forecast_periods, frequency)
                
        elif model_name == 'arima' and STATSMODELS_AVAILABLE:
            try:
                model = ARIMA(train, order=(1, 1, 1))
                fitted_model = model.fit()
                fitted_values = fitted_model.fittedvalues.tolist()
                forecast = fitted_model.forecast(forecast_periods).tolist()
            except Exception as e:
                self.logger.warning(f"ARIMA failed: {e}, falling back to linear")
                return self._train_forecast_model(ts, 'linear_trend', forecast_periods, frequency)
                
        elif model_name == 'prophet' and PROPHET_AVAILABLE:
            try:
                prophet_df = pd.DataFrame({
                    'ds': train.index,
                    'y': train.values
                })
                model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
                model.fit(prophet_df)
                
                future = model.make_future_dataframe(periods=forecast_periods, freq=frequency)
                prophet_forecast = model.predict(future)
                
                fitted_values = prophet_forecast['yhat'][:len(train)].tolist()
                forecast = prophet_forecast['yhat'][-forecast_periods:].tolist()
            except Exception as e:
                self.logger.warning(f"Prophet failed: {e}, falling back to linear")
                return self._train_forecast_model(ts, 'linear_trend', forecast_periods, frequency)
        else:
            # Default to moving average
            return self._train_forecast_model(ts, 'moving_average', forecast_periods, frequency)
        
        training_time = time.time() - start_time
        
        # Calculate metrics on test set
        X_test_pred = []
        if model_name in ['linear_trend']:
            X_test = np.arange(train_size, len(ts)).reshape(-1, 1)
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            X_train = np.arange(len(train)).reshape(-1, 1)
            model.fit(X_train, train.values)
            X_test_pred = model.predict(X_test)
        elif len(fitted_values) >= len(test):
            X_test_pred = fitted_values[-len(test):]
        else:
            X_test_pred = [np.mean(train)] * len(test)
        
        metrics = {
            'rmse': float(np.sqrt(mean_squared_error(test, X_test_pred[:len(test)]))),
            'mae': float(mean_absolute_error(test, X_test_pred[:len(test)])),
            'training_time': training_time
        }
        
        try:
            metrics['r2'] = float(r2_score(test, X_test_pred[:len(test)]))
        except:
            metrics['r2'] = 0.0
        
        return forecast, metrics
    
    def _detect_seasonality(self, ts: pd.Series) -> Optional[int]:
        """Detect seasonality period in time series."""
        if not STATSMODELS_AVAILABLE or len(ts) < 30:
            return None
        
        try:
            # Use ACF to detect seasonality
            acf_values = acf(ts.dropna(), nlags=min(50, len(ts) // 2))
            
            # Find peaks in ACF
            for i in range(2, len(acf_values) - 1):
                if acf_values[i] > acf_values[i-1] and acf_values[i] > acf_values[i+1]:
                    if acf_values[i] > 0.3:  # Significant correlation
                        return i
            return None
        except:
            return None
    
    def _find_best_model(self, model_results: Dict[str, Dict]) -> str:
        """Find the best performing model based on RMSE."""
        best_model = 'moving_average'
        best_rmse = float('inf')
        
        for model_name, metrics in model_results.items():
            if 'error' not in metrics:
                rmse = metrics.get('rmse', float('inf'))
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_model = model_name
        
        return best_model
    
    def _generate_forecasting_insights(
        self,
        ts_analysis: Dict,
        model_results: Dict,
        best_model: str,
        forecast_periods: int
    ) -> Dict[str, Any]:
        """Generate AI-powered insights about forecasting results using Gemini."""
        results_summary = {}
        for name, m in model_results.items():
            if 'error' not in m:
                results_summary[name] = {
                    'rmse': round(m.get('rmse', 0), 4),
                    'mae': round(m.get('mae', 0), 4)
                }
        
        prompt = f"""Analyze these time series forecasting results and provide insights:

Time Series Analysis: {json.dumps(ts_analysis, indent=2, default=str)}
Best Model: {best_model}
Forecast Periods: {forecast_periods}

Model Results:
{json.dumps(results_summary, indent=2)}

Provide:
1. Assessment of the time series characteristics
2. Why the best model performed well for this data
3. Confidence level of the forecast
4. Potential risks or limitations
5. Recommendations for improving forecast accuracy

Format as JSON with keys: ts_assessment, best_model_rationale, confidence_level, risks, recommendations"""
        
        try:
            response = self.call_gemini(prompt, temperature=0.4)
            try:
                insights = json.loads(response)
            except json.JSONDecodeError:
                insights = {'raw_insights': response}
            return insights
        except Exception as e:
            self.logger.warning(f"AI forecasting insights failed: {e}")
            return {
                'best_model_rationale': f"{best_model} achieved the lowest error metrics",
                'recommendations': ['Try longer historical data', 'Consider seasonal decomposition']
            }
    
    def _generate_visualization_data(
        self,
        ts: pd.Series,
        forecasts: Dict[str, List[float]],
        best_model: str
    ) -> Dict[str, Any]:
        """Generate data for visualization."""
        # Historical data
        historical = {
            'dates': [str(d) for d in ts.index.tolist()],
            'values': ts.tolist()
        }
        
        # Forecast data
        last_date = ts.index[-1]
        freq = pd.infer_freq(ts.index) or 'D'
        
        forecast_dates = pd.date_range(
            start=last_date,
            periods=len(forecasts.get(best_model, [])) + 1,
            freq=freq
        )[1:]  # Skip first as it's the last historical date
        
        forecast_viz = {
            'dates': [str(d) for d in forecast_dates.tolist()],
            'values': forecasts.get(best_model, [])
        }
        
        return {
            'historical': historical,
            'forecast': forecast_viz,
            'best_model': best_model
        }
