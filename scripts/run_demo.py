#!/usr/bin/env python
"""
Demo Script - Run the AutoDS Pipeline locally
Shows end-to-end pipeline execution with sample data
"""

import os
import sys
import json
import argparse
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

from agents.orchestrator import OrchestratorAgent
from agents.data_ingestion_agent import DataIngestionAgent
from agents.preprocessing_agent import PreprocessingAgent
from agents.model_selection_agent import ModelSelectionAgent
from agents.training_agent import TrainingAgent
from agents.evaluation_agent import EvaluationAgent


def create_sample_data():
    """Create sample datasets for demo."""
    import pandas as pd
    import numpy as np
    
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # Classification dataset (Iris-like)
    np.random.seed(42)
    n_samples = 200
    
    classification_data = pd.DataFrame({
        'sepal_length': np.random.normal(5.8, 0.8, n_samples),
        'sepal_width': np.random.normal(3.0, 0.4, n_samples),
        'petal_length': np.random.normal(3.7, 1.8, n_samples),
        'petal_width': np.random.normal(1.2, 0.8, n_samples),
        'species': np.random.choice(['setosa', 'versicolor', 'virginica'], n_samples)
    })
    
    classification_path = os.path.join(data_dir, 'iris_sample.csv')
    classification_data.to_csv(classification_path, index=False)
    print(f"✅ Created classification dataset: {classification_path}")
    
    # Regression dataset (Housing-like)
    regression_data = pd.DataFrame({
        'bedrooms': np.random.randint(1, 6, n_samples),
        'bathrooms': np.random.randint(1, 4, n_samples),
        'sqft': np.random.randint(800, 4000, n_samples),
        'age': np.random.randint(0, 50, n_samples),
        'location_score': np.random.uniform(0, 10, n_samples).round(2),
        'price': np.random.randint(150000, 800000, n_samples)
    })
    
    regression_path = os.path.join(data_dir, 'housing_sample.csv')
    regression_data.to_csv(regression_path, index=False)
    print(f"✅ Created regression dataset: {regression_path}")
    
    # Dataset with missing values (for testing preprocessing)
    messy_data = classification_data.copy()
    # Introduce some missing values
    mask = np.random.random(messy_data.shape) < 0.1
    messy_data = messy_data.mask(mask)
    messy_data['species'] = classification_data['species']  # Keep target intact
    
    messy_path = os.path.join(data_dir, 'iris_messy.csv')
    messy_data.to_csv(messy_path, index=False)
    print(f"✅ Created messy dataset: {messy_path}")
    
    return {
        'classification': classification_path,
        'regression': regression_path,
        'messy': messy_path
    }


def run_pipeline_demo(
    data_path: str,
    target_column: str,
    config: dict = None
):
    """
    Run the AutoDS pipeline on provided data.
    
    Args:
        data_path: Path to data file
        target_column: Target column for prediction
        config: Optional configuration
    """
    print("\n" + "="*60)
    print("🚀 AutoDS Pipeline - Demo Run")
    print("="*60)
    print(f"📁 Data: {data_path}")
    print(f"🎯 Target: {target_column}")
    print(f"⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60 + "\n")
    
    # Create orchestrator
    print("📋 Initializing Pipeline Agents...")
    orchestrator = OrchestratorAgent()
    
    # Register all agents
    orchestrator.register_agent('ingestion', DataIngestionAgent())
    orchestrator.register_agent('preprocessing', PreprocessingAgent())
    orchestrator.register_agent('model_selection', ModelSelectionAgent())
    orchestrator.register_agent('training', TrainingAgent())
    orchestrator.register_agent('evaluation', EvaluationAgent())
    
    print("✅ All agents registered\n")
    
    # Run pipeline
    print("🔄 Starting Pipeline Execution...\n")
    
    try:
        # Build pipeline config
        pipeline_config = {
            'data_source': data_path,
            'target_column': target_column,
            **(config or {})
        }
        
        result = orchestrator.run_pipeline(pipeline_config)
        
        print("\n" + "="*60)
        print("✅ Pipeline Completed Successfully!")
        print("="*60)
        
        # Print summary
        if result.get('status') == 'completed':
            print("\n📊 Results Summary:")
            print("-" * 40)
            
            # Get evaluation results
            eval_result = result.get('evaluation', {})
            best_model = eval_result.get('best_model', {})
            
            if best_model:
                print(f"🏆 Best Model: {best_model.get('name', 'N/A')}")
                print(f"📈 Primary Metric: {best_model.get('primary_metric_value', 'N/A'):.4f}")
            
            # Get model comparison
            comparison = eval_result.get('model_comparison', [])
            if comparison:
                print("\n📋 Model Rankings:")
                for i, model in enumerate(comparison[:5], 1):
                    print(f"  {i}. {model.get('model_name', 'N/A')} - Score: {model.get('score', 0):.4f}")
            
            # AI Insights
            insights = eval_result.get('ai_insights', {})
            if insights.get('recommendation'):
                print(f"\n💡 AI Recommendation:")
                print(f"   {insights.get('recommendation', 'N/A')}")
        
        print("\n" + "="*60)
        
        return result
        
    except Exception as e:
        print(f"\n❌ Pipeline Failed: {str(e)}")
        raise


def main():
    """Main entry point for demo script."""
    parser = argparse.ArgumentParser(
        description='AutoDS Pipeline Demo',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create sample data and run classification demo
  python run_demo.py --create-data --demo classification

  # Run with custom data
  python run_demo.py --data path/to/data.csv --target target_column

  # Run regression demo
  python run_demo.py --demo regression
        """
    )
    
    parser.add_argument(
        '--create-data',
        action='store_true',
        help='Create sample datasets'
    )
    
    parser.add_argument(
        '--demo',
        choices=['classification', 'regression', 'messy'],
        help='Run a specific demo'
    )
    
    parser.add_argument(
        '--data',
        type=str,
        help='Path to data file'
    )
    
    parser.add_argument(
        '--target',
        type=str,
        help='Target column name'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to config JSON file'
    )
    
    args = parser.parse_args()
    
    # Create sample data if requested
    datasets = {}
    if args.create_data or args.demo:
        print("📦 Creating sample datasets...")
        datasets = create_sample_data()
        print()
    
    # Determine what to run
    if args.data and args.target:
        # Run with custom data
        config = {}
        if args.config:
            with open(args.config, 'r') as f:
                config = json.load(f)
        
        run_pipeline_demo(args.data, args.target, config)
        
    elif args.demo:
        # Run demo
        if args.demo == 'classification':
            data_path = datasets.get('classification') or os.path.join(
                os.path.dirname(os.path.dirname(__file__)), 'data', 'iris_sample.csv'
            )
            run_pipeline_demo(data_path, 'species')
            
        elif args.demo == 'regression':
            data_path = datasets.get('regression') or os.path.join(
                os.path.dirname(os.path.dirname(__file__)), 'data', 'housing_sample.csv'
            )
            run_pipeline_demo(data_path, 'price')
            
        elif args.demo == 'messy':
            data_path = datasets.get('messy') or os.path.join(
                os.path.dirname(os.path.dirname(__file__)), 'data', 'iris_messy.csv'
            )
            run_pipeline_demo(data_path, 'species')
    
    else:
        # Show help
        parser.print_help()
        print("\n💡 Quick Start:")
        print("   python run_demo.py --create-data --demo classification")


if __name__ == '__main__':
    main()
