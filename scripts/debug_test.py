"""Debug test - runs the orchestrator directly"""
import sys
import traceback
sys.path.insert(0, r'D:\AutoDS(Copilot)\autods-pipeline')

from agents.orchestrator import OrchestratorAgent
from agents.data_ingestion_agent import DataIngestionAgent
from agents.preprocessing_agent import PreprocessingAgent
from agents.model_selection_agent import ModelSelectionAgent
from agents.training_agent import TrainingAgent
from agents.evaluation_agent import EvaluationAgent

def main():
    print("Creating orchestrator...")
    orchestrator = OrchestratorAgent()
    
    print("Registering agents...")
    orchestrator.register_agent('ingestion', DataIngestionAgent())
    orchestrator.register_agent('preprocessing', PreprocessingAgent())
    orchestrator.register_agent('model_selection', ModelSelectionAgent())
    orchestrator.register_agent('training', TrainingAgent())
    orchestrator.register_agent('evaluation', EvaluationAgent())
    
    # Build the same config as app.py would
    pipeline_config = {
        'pipeline_id': 'test-123',
        'data_source': r'D:\AutoDS(Copilot)\autods-pipeline\data\iris_sample.csv',
        'target_column': 'species',
    }
    
    print(f"Pipeline config: {pipeline_config}")
    print(f"Config type: {type(pipeline_config)}")
    
    try:
        print("Running pipeline...")
        result = orchestrator.run_pipeline(pipeline_config)
        print("SUCCESS!")
        print(f"Result keys: {result.keys() if isinstance(result, dict) else result}")
    except Exception as e:
        print(f"ERROR: {e}")
        traceback.print_exc()

if __name__ == '__main__':
    main()
