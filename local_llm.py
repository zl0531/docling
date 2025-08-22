import os
import requests
import json
import logging
from typing import Optional

_logger = logging.getLogger(__name__)

class LocalLLM:
    """Local LLM class for calling local LLM services."""
    
    def __init__(self):
        # Get configuration from environment variables
        self.app_key = os.getenv("LOCAL_LLM_APP_KEY")
        self.secret_key = os.getenv("LOCAL_LLM_SECRET_KEY")
        self.app_code = os.getenv("LOCAL_LLM_APP_CODE")
        self.base_url = os.getenv("LOCAL_LLM_URL_OFFICE", "https://ea-ai-gateway.corp.kuaishou.com/ea-ai-gateway/open/v1")
        self.default_model = os.getenv("LOCAL_LLM_DEFAULT_MODEL", "deepseek_v3")
        
        # Token cache
        self.access_token = None
        self.token_expires_at = 0
        
        # Log configuration for debugging
        _logger.info(f"LocalLLM initialized with app_key: {self.app_key[:8] if self.app_key else None}..., base_url: {self.base_url}")
        
    def _get_access_token(self) -> str:
        """Get access token from the token service."""
        # Check if we have a valid cached token
        import time
        current_time = time.time()
        if self.access_token and current_time < self.token_expires_at:
            _logger.debug("Using cached access token")
            return self.access_token
            
        # Validate required configuration
        if not self.app_key or not self.secret_key:
            raise ValueError("LOCAL_LLM_APP_KEY and LOCAL_LLM_SECRET_KEY must be set in environment variables")
            
        # Get new token
        token_url = "https://is-gateway.corp.kuaishou.com/token/get"
        payload = {
            "appKey": self.app_key,
            "secretKey": self.secret_key
        }
        
        _logger.debug(f"Requesting access token from {token_url}")
        _logger.debug(f"Payload: appKey={self.app_key[:8] if self.app_key else None}...")
        
        try:
            response = requests.post(token_url, json=payload)
            _logger.debug(f"Token response status: {response.status_code}")
            
            # Log response content for debugging (but don't log sensitive data)
            if response.status_code != 200:
                _logger.error(f"Token request failed with status {response.status_code}")
                _logger.error(f"Response headers: {dict(response.headers)}")
                try:
                    error_content = response.json()
                    _logger.error(f"Error response: {error_content}")
                except:
                    _logger.error(f"Error response text: {response.text}")
                response.raise_for_status()
                
            result = response.json()
            _logger.debug(f"Token response keys: {result.keys()}")
            
            if "result" in result and "accessToken" in result["result"]:
                self.access_token = result["result"]["accessToken"]
                # Set expiration time (usually 24 hours, but we'll refresh earlier)
                self.token_expires_at = current_time + 23 * 3600  # Refresh after 23 hours
                _logger.info("Successfully obtained access token")
                return self.access_token
            else:
                error_msg = f"Failed to get access token. Response: {result}"
                _logger.error(error_msg)
                raise Exception(error_msg)
        except requests.exceptions.RequestException as e:
            _logger.error(f"Network error getting access token: {e}")
            raise
        except Exception as e:
            _logger.error(f"Error getting access token: {e}")
            raise
    
    def translate(self, text: str, src: str = "en", dest: str = "zh") -> str:
        """Translate text using the local LLM service.
        
        Args:
            text: Text to translate
            src: Source language (default: en)
            dest: Target language (default: zh for Chinese)
            
        Returns:
            Translated text
        """
        if not text.strip():
            return text
            
        # Get access token
        try:
            token = self._get_access_token()
        except Exception as e:
            _logger.error(f"Failed to get access token, returning original text: {e}")
            return text
            
        # Prepare the prompt
        if dest == "zh":
            prompt = f"请将以下英文文本翻译成中文：\n\n{text}"
        else:
            prompt = f"Please translate the following text from {src} to {dest}:\n\n{text}"
            
        # System message to ensure proper format
        system_message = "You are a professional translator. Please translate the given text accurately while maintaining the original meaning and context."
        
        # Prepare the API request
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "X-App-Key": self.app_key,
            "X-App-Code": self.app_code
        }
        
        payload = {
            "model": self.default_model,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3
        }
        
        _logger.debug(f"Sending translation request to {url}")
        _logger.debug(f"Model: {self.default_model}")
        _logger.debug(f"Prompt length: {len(prompt)} characters")
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            _logger.debug(f"Translation response status: {response.status_code}")
            
            if response.status_code != 200:
                _logger.error(f"Translation request failed with status {response.status_code}")
                _logger.error(f"Response headers: {dict(response.headers)}")
                try:
                    error_content = response.json()
                    _logger.error(f"Error response: {error_content}")
                except:
                    _logger.error(f"Error response text: {response.text}")
                response.raise_for_status()
                
            result = response.json()
            _logger.debug(f"Translation response keys: {result.keys()}")
            
            if "choices" in result and len(result["choices"]) > 0:
                translated_text = result["choices"][0]["message"]["content"]
                # Clean up the response if needed
                translated_text = translated_text.strip()
                _logger.debug(f"Translation successful, response length: {len(translated_text)} characters")
                return translated_text
            else:
                error_msg = f"Unexpected response format: {result}"
                _logger.error(error_msg)
                return text
        except requests.exceptions.Timeout:
            _logger.error("Translation request timed out")
            return text
        except requests.exceptions.RequestException as e:
            _logger.error(f"Network error translating text: {e}")
            return text
        except Exception as e:
            _logger.error(f"Error translating text: {e}")
            return text