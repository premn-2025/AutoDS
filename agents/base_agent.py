"""
BaseAgent - Foundation class for all AutoDS agents
Provides LLM integration (Local Ollama, Gemini, or Vertex AI)
"""

import os
import time
import json
import logging
import requests
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from datetime import datetime

# Google Cloud imports (optional)
try:
    import vertexai
    from vertexai.generative_models import GenerativeModel, GenerationConfig
    VERTEX_AI_AVAILABLE = True
except ImportError:
    VERTEX_AI_AVAILABLE = False

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


class BaseAgent(ABC):
    """
    Base class for all AutoDS agents.
    Provides:
    - Local LLM support via Ollama (default)
    - Optional Gemini/Vertex AI integration
    - Error handling and retry logic
    - Structured logging
    """
    
    def __init__(
        self,
        name: str,
        model_name: str = None,
        use_local_model: bool = None,
        ollama_base_url: str = None
    ):
        """
        Initialize the base agent.
        
        Args:
            name: Agent name for identification
            model_name: Model to use (default: from env or 'gemini-flash-latest')
            use_local_model: Whether to use local Ollama model (auto-detect if None)
            ollama_base_url: Ollama API URL (default: http://localhost:11434)
        """
        self.name = name
        
        # Auto-detect: use Gemini if API key is available, otherwise try local
        gemini_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
        if use_local_model is None:
            self.use_local_model = not bool(gemini_key) and os.getenv('USE_LOCAL_MODEL', 'false').lower() == 'true'
        else:
            self.use_local_model = use_local_model
        
        # Set model name based on mode
        if self.use_local_model:
            self.model_name = model_name or os.getenv('OLLAMA_MODEL', 'llama3.2')
            self.ollama_base_url = ollama_base_url or os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        else:
            self.model_name = model_name or os.getenv('LLM_MODEL', 'gemini-flash-latest')
        
        self.model = None
        
        # Setup structured logging
        self._setup_logging()
        
        # Initialize LLM
        self._initialize_llm()
        
        self.logger.info(f"Agent '{name}' initialized successfully")
    
    def _setup_logging(self):
        """Configure structured logging"""
        self.logger = logging.getLogger(f"autods.{self.name}")
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s | %(levelname)s | [%(name)s] | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def _initialize_llm(self):
        """Initialize the LLM (Gemini API preferred, then Vertex AI, then Ollama)"""
        try:
            if self.use_local_model:
                # Check if Ollama is running
                if self._check_ollama_available():
                    self.logger.info(f"Using local Ollama model: {self.model_name}")
                    # Ensure model is pulled
                    self._ensure_model_available()
                else:
                    self.logger.warning(f"Ollama not available at {self.ollama_base_url}, will use fallback responses")
            
            # Prioritize Gemini API (simpler setup, just needs API key)
            elif GENAI_AVAILABLE:
                api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
                if api_key:
                    genai.configure(api_key=api_key)
                    self.model = genai.GenerativeModel(self.model_name)
                    self.logger.info(f"Initialized Gemini API with model: {self.model_name}")
                elif VERTEX_AI_AVAILABLE:
                    # Fall back to Vertex AI if no API key but SDK available
                    self._init_vertex_ai()
                else:
                    self.logger.warning("No Gemini API key found, LLM calls will use fallback")
                    
            elif VERTEX_AI_AVAILABLE:
                self._init_vertex_ai()
            else:
                self.logger.warning("No LLM SDK available, using fallback responses")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM: {str(e)}")
    
    def _init_vertex_ai(self):
        """Initialize Vertex AI (requires GCP credentials)"""
        project_id = os.getenv('GCP_PROJECT_ID', os.getenv('GOOGLE_CLOUD_PROJECT', 'autods-observability'))
        region = os.getenv('GCP_REGION', 'us-central1')
        
        vertexai.init(project=project_id, location=region)
        self.model = GenerativeModel(self.model_name)
        self.logger.info(f"Initialized Vertex AI with model: {self.model_name}")
    
    def _check_ollama_available(self) -> bool:
        """Check if Ollama server is running"""
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def _ensure_model_available(self):
        """Pull the model if not already available"""
        try:
            # Check if model exists
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m.get('name', '').split(':')[0] for m in models]
                
                if self.model_name.split(':')[0] not in model_names:
                    self.logger.info(f"Pulling model {self.model_name}... (this may take a while)")
                    pull_response = requests.post(
                        f"{self.ollama_base_url}/api/pull",
                        json={"name": self.model_name},
                        timeout=600  # 10 minutes for large models
                    )
                    if pull_response.status_code == 200:
                        self.logger.info(f"Model {self.model_name} pulled successfully")
                    else:
                        self.logger.warning(f"Failed to pull model: {pull_response.text}")
                else:
                    self.logger.info(f"Model {self.model_name} already available")
        except Exception as e:
            self.logger.warning(f"Could not check/pull model: {e}")
    
    def _call_ollama(self, prompt: str, temperature: float = 0.7, max_tokens: int = 4096) -> str:
        """
        Call local Ollama model.
        
        Args:
            prompt: The prompt to send
            temperature: Sampling temperature
            max_tokens: Maximum output tokens
            
        Returns:
            Model response text
        """
        try:
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens
                    }
                },
                timeout=120  # 2 minutes timeout
            )
            
            if response.status_code == 200:
                return response.json().get('response', '')
            else:
                self.logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                return self._get_fallback_response(prompt)
                
        except requests.exceptions.Timeout:
            self.logger.error("Ollama request timed out")
            return self._get_fallback_response(prompt)
        except Exception as e:
            self.logger.error(f"Ollama call failed: {e}")
            return self._get_fallback_response(prompt)
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute agent task.
        
        Args:
            input_data: Input data for the agent
            
        Returns:
            Dict containing agent results
        """
        start_time = time.time()
        execution_id = f"{self.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            self.logger.info(
                f"Starting execution",
                extra={'execution_id': execution_id, 'input_keys': list(input_data.keys())}
            )
            
            # Execute agent-specific logic
            result = self._run(input_data)
            
            # Track success
            duration = time.time() - start_time
            
            self.logger.info(
                f"Execution completed successfully in {duration:.2f}s",
                extra={'execution_id': execution_id, 'duration': duration}
            )
            
            # Add execution metadata to result
            result['_metadata'] = {
                'agent': self.name,
                'execution_id': execution_id,
                'duration': duration,
                'status': 'success',
                'timestamp': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            error_type = type(e).__name__
            
            self.logger.error(
                f"Execution failed: {str(e)}",
                extra={
                    'execution_id': execution_id,
                    'error_type': error_type,
                    'duration': duration
                },
                exc_info=True
            )
            
            raise
    
    @abstractmethod
    def _run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Agent-specific execution logic. Must be implemented by subclasses.
        
        Args:
            input_data: Input data for the agent
            
        Returns:
            Dict containing agent results
        """
        raise NotImplementedError("Subclasses must implement _run method")
    
    def call_gemini(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: str = "text"
    ) -> str:
        """
        Call LLM (local Ollama or cloud Gemini).
        
        Args:
            prompt: The prompt to send to the LLM
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum output tokens
            response_format: Expected response format ("text" or "json")
            
        Returns:
            LLM response text
        """
        start_time = time.time()
        prompt_tokens = len(prompt.split())  # Approximate
        
        try:
            # Use local Ollama model
            if self.use_local_model:
                response_text = self._call_ollama(prompt, temperature, max_tokens)
            # Use cloud model
            elif self.model:
                # Check if model is from google.generativeai (Gemini API) or vertexai
                is_genai_model = GENAI_AVAILABLE and type(self.model).__module__.startswith('google.generativeai')
                
                if is_genai_model:
                    # Gemini API uses dict for generation_config
                    response = self.model.generate_content(
                        prompt,
                        generation_config={
                            'temperature': temperature,
                            'max_output_tokens': max_tokens
                        }
                    )
                    response_text = response.text
                elif VERTEX_AI_AVAILABLE:
                    # Vertex AI uses GenerationConfig object
                    generation_config = GenerationConfig(
                        temperature=temperature,
                        max_output_tokens=max_tokens,
                    )
                    response = self.model.generate_content(
                        prompt,
                        generation_config=generation_config
                    )
                    response_text = response.text
                else:
                    response = self.model.generate_content(prompt)
                    response_text = response.text
            else:
                self.logger.warning("No LLM available, using fallback response")
                return self._get_fallback_response(prompt)
            
            # Calculate metrics
            duration = time.time() - start_time
            completion_tokens = len(response_text.split())  # Approximate
            total_tokens = prompt_tokens + completion_tokens
            
            # Estimate cost (Gemini pricing - approximate)
            cost = (prompt_tokens * 0.00000125) + (completion_tokens * 0.00000375)
            
            self.logger.info(
                "LLM call completed",
                extra={
                    'model': self.model_name,
                    'duration': duration,
                    'prompt_tokens': prompt_tokens,
                    'completion_tokens': completion_tokens,
                    'cost': cost,
                    'agent': self.name
                }
            )
            
            return response_text
            
        except Exception as e:
            duration = time.time() - start_time
            error_type = type(e).__name__
            
            self.logger.error(
                f"LLM call failed: {str(e)}",
                extra={'duration': duration, 'error_type': error_type}
            )
            
            raise
    
    def call_gemini_json(
        self,
        prompt: str,
        temperature: float = 0.3
    ) -> Dict[str, Any]:
        """
        Call Gemini and parse JSON response.
        
        Args:
            prompt: The prompt (should request JSON output)
            temperature: Sampling temperature
            
        Returns:
            Parsed JSON as dictionary
        """
        response = self.call_gemini(
            prompt + "\n\nRespond ONLY with valid JSON, no markdown or extra text.",
            temperature=temperature,
            response_format="json"
        )
        
        # Clean response and parse JSON
        response = response.strip()
        if response.startswith('```json'):
            response = response[7:]
        if response.startswith('```'):
            response = response[3:]
        if response.endswith('```'):
            response = response[:-3]
        
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON response: {e}")
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            raise ValueError(f"Could not parse LLM response as JSON: {response[:200]}")
    
    def _get_fallback_response(self, prompt: str) -> str:
        """
        Provide fallback response when LLM is not available.
        Useful for testing and development.
        """
        self.logger.warning("Using fallback response - LLM not available")
        
        if "preprocessing" in prompt.lower():
            return json.dumps({
                "steps": ["handle_missing_values", "encode_categorical", "scale_numeric"],
                "code": "# Fallback preprocessing steps"
            })
        elif "model" in prompt.lower() and "select" in prompt.lower():
            return json.dumps({
                "problem_type": "classification",
                "recommended_models": ["random_forest", "xgboost"],
                "rationale": "Fallback model selection"
            })
        elif "analyz" in prompt.lower():
            return json.dumps({
                "analysis": "Fallback data analysis",
                "recommendations": ["Check data quality", "Handle missing values"]
            })
        else:
            return json.dumps({"response": "Fallback response - LLM not configured"})
    
    def track_decision(self, decision: str, confidence: float):
        """
        Track agent decision with confidence score.
        
        Args:
            decision: Description of the decision made
            confidence: Confidence score (0-1)
        """
        self.logger.info(
            f"Decision made: {decision}",
            extra={'confidence': confidence, 'agent': self.name}
        )
        
        if confidence < 0.6:
            self.logger.warning(
                f"Low confidence decision: {decision}",
                extra={'confidence': confidence}
            )
