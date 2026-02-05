"""
AI Client Wrapper for Phase 6

This module provides a unified interface for interacting with different LLM providers
(OpenRouter, Google Gemini) with error handling, timeouts, and provider abstraction.

SUPPORTED PROVIDERS:
- OpenRouter (supports multiple models via unified API)
- Google Gemini (Google's AI models)

KEY FEATURES:
- Provider abstraction (easy to switch)
- Automatic retry logic
- Timeout handling
- Error handling
- Response validation
"""

import os
import time
import logging
from typing import Optional, Dict, Any, List
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AIProvider(Enum):
    """Supported AI providers"""
    OPENROUTER = "openrouter"
    GEMINI = "gemini"


class AIClient:
    """
    Unified AI client for interacting with LLM providers.
    
    Abstracts provider-specific logic and provides a consistent interface
    for generating AI feedback.
    """
    
    def __init__(
        self,
        provider: str = "openrouter",
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 2
    ):
        """
        Initialize AI client.
        
        Args:
            provider: AI provider name ('openrouter' or 'gemini')
            api_key: API key (if None, loads from environment)
            model: Model name (if None, uses default)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        self.provider = AIProvider(provider.lower())
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Load API key
        if api_key:
            self.api_key = api_key
        else:
            self.api_key = self._load_api_key()
        
        # Set model
        if model:
            self.model = model
        else:
            self.model = self._get_default_model()
        
        # Initialize provider-specific client
        self.client = self._initialize_client()
        
        logger.info(f"Initialized AI client: provider={self.provider.value}, model={self.model}")
    
    def _load_api_key(self) -> str:
        """
        Load API key from environment variables.
        
        Returns:
            API key string
        
        Raises:
            ValueError: If API key not found
        """
        if self.provider == AIProvider.OPENROUTER:
            key = os.getenv('OPENROUTER_API_KEY')
            if not key:
                raise ValueError(
                    "OPENROUTER_API_KEY not found in environment. "
                    "Please set it in .env file or pass directly."
                )
            return key
        
        elif self.provider == AIProvider.GEMINI:
            key = os.getenv('GEMINI_API_KEY')
            if not key:
                raise ValueError(
                    "GEMINI_API_KEY not found in environment. "
                    "Please set it in .env file or pass directly."
                )
            return key
        
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def _get_default_model(self) -> str:
        """
        Get default model for the provider.
        
        Returns:
            Default model name
        """
        if self.provider == AIProvider.OPENROUTER:
            # Default to a cost-effective, fast model
            return os.getenv('OPENROUTER_MODEL', 'meta-llama/llama-3.1-8b-instruct:free')
        
        elif self.provider == AIProvider.GEMINI:
            return os.getenv('GEMINI_MODEL', 'gemini-1.5-flash')
        
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def _initialize_client(self):
        """
        Initialize provider-specific client.
        
        Returns:
            Initialized client object
        """
        if self.provider == AIProvider.OPENROUTER:
            try:
                from openai import OpenAI
                return OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=self.api_key
                )
            except ImportError:
                raise ImportError(
                    "OpenAI library not installed. Install with: pip install openai>=1.0.0"
                )
        
        elif self.provider == AIProvider.GEMINI:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                return genai.GenerativeModel(self.model)
            except ImportError:
                raise ImportError(
                    "Google Generative AI library not installed. "
                    "Install with: pip install google-generativeai"
                )
        
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def generate_response(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1500
    ) -> str:
        """
        Generate AI response for the given prompts.
        
        Args:
            system_prompt: System prompt defining AI's role
            user_prompt: User prompt with specific task
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens in response
        
        Returns:
            Generated response text
        
        Raises:
            Exception: If generation fails after retries
        """
        for attempt in range(self.max_retries + 1):
            try:
                if self.provider == AIProvider.OPENROUTER:
                    return self._generate_openrouter(
                        system_prompt, user_prompt, temperature, max_tokens
                    )
                
                elif self.provider == AIProvider.GEMINI:
                    return self._generate_gemini(
                        system_prompt, user_prompt, temperature, max_tokens
                    )
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                
                if attempt < self.max_retries:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"All retry attempts failed: {str(e)}")
                    raise Exception(f"AI generation failed: {str(e)}")
    
    def _generate_openrouter(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int
    ) -> str:
        """
        Generate response using OpenRouter.
        
        Args:
            system_prompt: System prompt
            user_prompt: User prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens
        
        Returns:
            Generated response text
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=self.timeout
        )
        
        return response.choices[0].message.content
    
    def _generate_gemini(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int
    ) -> str:
        """
        Generate response using Google Gemini.
        
        Args:
            system_prompt: System prompt
            user_prompt: User prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens
        
        Returns:
            Generated response text
        """
        # Gemini combines system and user prompts
        combined_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        # Configure generation
        generation_config = {
            'temperature': temperature,
            'max_output_tokens': max_tokens,
        }
        
        response = self.client.generate_content(
            combined_prompt,
            generation_config=generation_config
        )
        
        return response.text
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check if AI client is properly configured and accessible.
        
        Returns:
            Health check status dict
        """
        try:
            # Try a minimal generation
            test_response = self.generate_response(
                system_prompt="You are a helpful assistant.",
                user_prompt="Say 'OK' if you can read this.",
                temperature=0.0,
                max_tokens=10
            )
            
            return {
                'status': 'healthy',
                'provider': self.provider.value,
                'model': self.model,
                'test_response': test_response[:50] if test_response else None
            }
        
        except Exception as e:
            return {
                'status': 'unhealthy',
                'provider': self.provider.value,
                'model': self.model,
                'error': str(e)
            }
    
    def get_provider_info(self) -> Dict[str, str]:
        """
        Get information about current provider configuration.
        
        Returns:
            Provider info dict
        """
        return {
            'provider': self.provider.value,
            'model': self.model,
            'timeout': str(self.timeout),
            'max_retries': str(self.max_retries),
            'api_key_set': 'Yes' if self.api_key else 'No'
        }


def get_ai_client(provider: Optional[str] = None) -> AIClient:
    """
    Factory function to get configured AI client.
    
    Args:
        provider: Provider name (if None, uses environment variable)
    
    Returns:
        Configured AIClient instance
    """
    if provider is None:
        provider = os.getenv('AI_PROVIDER', 'openrouter')
    
    return AIClient(provider=provider)


# Example usage
if __name__ == "__main__":
    # Example with OpenRouter
    try:
        client = AIClient(provider="openrouter")
        
        response = client.generate_response(
            system_prompt="You are a resume advisor.",
            user_prompt="Provide 3 tips for improving a resume.",
            temperature=0.7,
            max_tokens=500
        )
        
        print("Response:", response)
        print("\nHealth Check:", client.health_check())
        
    except Exception as e:
        print(f"Error: {e}")
