"""
OrchestratorAgent - Central coordinator for the AutoDS pipeline
Manages agent execution flow and creates intelligent execution plans
"""

import json
import time
import uuid
from typing import Any, Dict, List, Optional
from datetime import datetime

from agents.base_agent import BaseAgent


class OrchestratorAgent(BaseAgent):
    """
    Orchestrator agent that coordinates the entire AutoDS pipeline.
    Uses Gemini to create intelligent execution plans based on task configuration.
    """
    
    def __init__(self):
        super().__init__("Orchestrator")
        self.agents: Dict[str, BaseAgent] = {}
        self.pipeline_state: Dict[str, Any] = {}
        self.execution_history: List[Dict] = []
    
    def register_agent(self, name: str, agent: BaseAgent):
        """
        Register an agent in the pipeline.
        
        Args:
            name: Unique identifier for the agent
            agent: Agent instance to register
        """
        self.agents[name] = agent
        self.logger.info(f"Registered agent: {name} ({agent.name})")
    
    def get_registered_agents(self) -> List[str]:
        """Get list of registered agent names."""
        return list(self.agents.keys())
    
    def run_pipeline(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the complete AutoDS pipeline.
        
        Args:
            config: Pipeline configuration including:
                - pipeline_id: Optional unique identifier
                - data_source: Path to data file
                - target_column: Target variable name
                - problem_type: Optional (classification/regression)
                - custom_steps: Optional custom execution steps
                
        Returns:
            Dict containing results from all pipeline steps
        """
        if not isinstance(config, dict):
            raise TypeError(f"config must be a dict, got {type(config)}")
        
        pipeline_id = config.get('pipeline_id', str(uuid.uuid4())[:8])
        start_time = time.time()
        
        self.logger.info(
            f"Starting pipeline execution",
            extra={'pipeline_id': pipeline_id, 'config': config}
        )
        
        try:
            # Initialize pipeline state
            self.pipeline_state = {
                'pipeline_id': pipeline_id,
                'config': config,
                'start_time': datetime.now().isoformat(),
                'status': 'running'
            }
            
            # Create execution plan using Gemini
            plan = self._create_execution_plan(config)
            
            self.logger.info(
                f"Execution plan created with {len(plan.get('steps', []))} steps",
                extra={'plan': plan}
            )
            
            # Execute each step
            results = {}
            step_data = {'config': config}  # Initial data passed between steps
            
            for i, step in enumerate(plan['steps']):
                agent_name = step['agent']
                agent = self.agents.get(agent_name)
                
                if not agent:
                    raise ValueError(f"Agent '{agent_name}' not found. Registered: {list(self.agents.keys())}")
                
                self.logger.info(
                    f"Executing step {i+1}/{len(plan['steps'])}: {agent_name}",
                    extra={'step': step, 'pipeline_id': pipeline_id}
                )
                
                # Prepare input for this step
                step_input = self._prepare_step_input(step, step_data, config)
                
                # Execute agent
                step_result = agent.execute(step_input)
                
                # Store result and update step_data for next agent
                results[agent_name] = step_result
                step_data.update(step_result)
                step_data[f'{agent_name}_result'] = step_result
            
            # Track success
            duration = time.time() - start_time
            
            # Update pipeline state
            self.pipeline_state.update({
                'status': 'completed',
                'end_time': datetime.now().isoformat(),
                'duration': duration
            })
            
            self.logger.info(
                f"Pipeline completed successfully in {duration:.2f}s",
                extra={'pipeline_id': pipeline_id, 'duration': duration}
            )
            
            # Record execution history
            self.execution_history.append({
                'pipeline_id': pipeline_id,
                'status': 'success',
                'duration': duration,
                'timestamp': datetime.now().isoformat()
            })
            
            return {
                'pipeline_id': pipeline_id,
                'status': 'success',
                'duration': duration,
                'results': results,
                'plan': plan
            }
                
        except Exception as e:
            duration = time.time() - start_time
            error_type = type(e).__name__
            
            # Update pipeline state
            self.pipeline_state.update({
                'status': 'failed',
                'end_time': datetime.now().isoformat(),
                'duration': duration,
                'error': str(e),
                'error_type': error_type
            })
            
            self.logger.error(
                f"Pipeline failed: {str(e)}",
                extra={
                    'pipeline_id': pipeline_id,
                    'error_type': error_type,
                    'duration': duration
                },
                exc_info=True
            )
            
            # Record execution history
            self.execution_history.append({
                'pipeline_id': pipeline_id,
                'status': 'failed',
                'error': str(e),
                'duration': duration,
                'timestamp': datetime.now().isoformat()
            })
            
            raise
    
    def _create_execution_plan(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use Gemini to create an intelligent execution plan.
        
        Args:
            config: Pipeline configuration
            
        Returns:
            Execution plan with ordered steps
        """
        # Check for custom steps
        if 'custom_steps' in config:
            return {'steps': config['custom_steps']}
        
        # Build prompt for Gemini
        registered_agents = list(self.agents.keys())
        
        prompt = f"""
        You are an AI orchestrator for a data science pipeline.
        
        Available agents: {registered_agents}
        
        Task configuration:
        - Data source: {config.get('data_source', 'Not specified')}
        - Target column: {config.get('target_column', 'Not specified')}
        - Problem type: {config.get('problem_type', 'auto-detect')}
        
        Create an execution plan that:
        1. Uses the appropriate agents in the correct order
        2. Ensures data flows properly between agents
        3. Handles the full ML pipeline from data ingestion to evaluation
        
        Standard pipeline flow:
        1. ingestion - Load and analyze data
        2. preprocessing - Clean and transform data
        3. model_selection - Choose appropriate models
        4. training - Train the selected models
        5. evaluation - Evaluate model performance
        
        Return a JSON object with this structure:
        {{
            "steps": [
                {{"agent": "agent_name", "description": "what this step does"}}
            ]
        }}
        
        Only include agents that exist in the registered agents list.
        """
        
        try:
            plan = self.call_gemini_json(prompt, temperature=0.3)
            
            # Validate plan
            if 'steps' not in plan or not plan['steps']:
                raise ValueError("Invalid plan: no steps defined")
            
            for step in plan['steps']:
                if step.get('agent') not in self.agents:
                    # Try to find matching agent
                    agent_name = step.get('agent', '').lower()
                    matched = False
                    for registered in registered_agents:
                        if agent_name in registered.lower() or registered.lower() in agent_name:
                            step['agent'] = registered
                            matched = True
                            break
                    if not matched:
                        self.logger.warning(f"Agent '{step.get('agent')}' not found, will skip")
            
            # Track decision confidence
            self.track_decision("execution_plan_created", 0.85)
            
            return plan
            
        except Exception as e:
            self.logger.warning(f"Failed to create AI-driven plan: {e}, using default plan")
            
            # Fallback to default plan
            return self._get_default_plan(config)
    
    def _get_default_plan(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Get default execution plan when AI planning fails."""
        default_steps = []
        
        # Build plan based on registered agents
        # Added 'eda' after ingestion for exploratory analysis
        standard_order = ['ingestion', 'eda', 'preprocessing', 'model_selection', 'training', 'evaluation']
        
        for agent_name in standard_order:
            if agent_name in self.agents:
                default_steps.append({
                    'agent': agent_name,
                    'description': f'Execute {agent_name} agent'
                })
        
        return {'steps': default_steps}
    
    def _prepare_step_input(
        self,
        step: Dict[str, Any],
        step_data: Dict[str, Any],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Prepare input data for a pipeline step.
        
        Args:
            step: Current step configuration
            step_data: Accumulated data from previous steps
            config: Original pipeline configuration
            
        Returns:
            Input dict for the agent
        """
        agent_name = step['agent']
        
        # Build input based on agent type
        if agent_name == 'ingestion':
            return {
                'source': config.get('data_source'),
                'target_column': config.get('target_column'),
                'sample_size': config.get('sample_size', 10000)
            }
        
        elif agent_name == 'preprocessing':
            return {
                'data': step_data.get('data'),
                'metadata': step_data.get('metadata'),
                'analysis': step_data.get('analysis'),
                'target_column': config.get('target_column')
            }
        
        elif agent_name == 'eda':
            return {
                'data': step_data.get('data'),
                'target_column': config.get('target_column'),
                'metadata': step_data.get('metadata')
            }
        
        elif agent_name == 'model_selection':
            return {
                'data': step_data.get('data'),
                'target_column': config.get('target_column'),
                'problem_type': config.get('problem_type'),
                'preprocessing_strategy': step_data.get('strategy')
            }
        
        elif agent_name == 'training':
            return {
                'data': step_data.get('data'),
                'target_column': config.get('target_column'),
                'problem_type': step_data.get('problem_type'),
                'selected_models': step_data.get('recommended_models', ['random_forest']),
                'test_size': config.get('test_size', 0.2),
                'enable_tuning': config.get('enable_tuning', False)  # Hyperparameter tuning flag
            }
        
        elif agent_name == 'evaluation':
            return {
                'trained_models': step_data.get('trained_models', {}),
                'test_data': step_data.get('test_data'),
                'problem_type': step_data.get('problem_type'),
                'selected_model': step_data.get('best_model'),
                'target_column': config.get('target_column')
            }
        
        else:
            # Generic input for custom agents
            return {
                **step_data,
                'config': config,
                'step': step
            }
    
    def _run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Implementation of base agent's _run method.
        For orchestrator, this delegates to run_pipeline.
        """
        return self.run_pipeline(input_data)
    
    def get_pipeline_state(self) -> Dict[str, Any]:
        """Get current pipeline state."""
        return self.pipeline_state.copy()
    
    def get_execution_history(self) -> List[Dict]:
        """Get pipeline execution history."""
        return self.execution_history.copy()
