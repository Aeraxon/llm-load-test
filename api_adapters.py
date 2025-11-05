"""
API Adapters for different LLM backends
Supports: Ollama, vLLM, LM Studio, llama.cpp, OpenAI-compatible APIs
"""

import requests
import json
from typing import Dict, Any, Iterator, Tuple, Optional


class APIAdapter:
    """Base class for API adapters"""

    def __init__(self, base_url: str, api_key: Optional[str] = None):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key

    def make_request(self, model: str, prompt: str, timeout: int = 120) -> Tuple[bool, float, float, str]:
        """
        Makes a request to the API and returns results

        Returns:
            Tuple[success: bool, total_time: float, ttft: float, error_msg: str]
        """
        raise NotImplementedError

    def check_connection(self) -> bool:
        """Check if the API is reachable"""
        raise NotImplementedError


class OllamaAdapter(APIAdapter):
    """Adapter for Ollama API"""

    def make_request(self, model: str, prompt: str, timeout: int = 120) -> Tuple[bool, float, float, str]:
        import time

        try:
            start_time = time.time()
            ttft_measured = False
            first_token_time = 0.0

            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": True
                },
                timeout=timeout,
                stream=True
            )

            if response.status_code == 200:
                full_response = ""

                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line.decode('utf-8'))

                            if not ttft_measured and 'response' in data and data['response']:
                                first_token_time = time.time() - start_time
                                ttft_measured = True

                            if 'response' in data:
                                full_response += data['response']

                            if data.get('done', False):
                                break

                        except json.JSONDecodeError:
                            continue

                elapsed_time = time.time() - start_time

                if not ttft_measured:
                    first_token_time = elapsed_time

                return True, elapsed_time, first_token_time, ""
            else:
                return False, 0.0, 0.0, f"HTTP {response.status_code}"

        except requests.exceptions.Timeout:
            return False, 0.0, 0.0, "Timeout"
        except requests.exceptions.ConnectionError:
            return False, 0.0, 0.0, "Connection Error"
        except Exception as e:
            return False, 0.0, 0.0, str(e)

    def check_connection(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False


class OpenAICompatibleAdapter(APIAdapter):
    """
    Adapter for OpenAI-compatible APIs
    Works with: vLLM, LM Studio, llama.cpp server, Text Generation WebUI, etc.
    """

    def make_request(self, model: str, prompt: str, timeout: int = 120) -> Tuple[bool, float, float, str]:
        import time

        try:
            start_time = time.time()
            ttft_measured = False
            first_token_time = 0.0

            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            response = requests.post(
                f"{self.base_url}/v1/completions",
                headers=headers,
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": True,
                    "max_tokens": 2000
                },
                timeout=timeout,
                stream=True
            )

            if response.status_code == 200:
                full_response = ""

                for line in response.iter_lines():
                    if line:
                        line_str = line.decode('utf-8')
                        if line_str.startswith('data: '):
                            line_str = line_str[6:]

                        if line_str.strip() == '[DONE]':
                            break

                        try:
                            data = json.loads(line_str)

                            if 'choices' in data and len(data['choices']) > 0:
                                text = data['choices'][0].get('text', '')

                                if text and not ttft_measured:
                                    first_token_time = time.time() - start_time
                                    ttft_measured = True

                                full_response += text

                        except json.JSONDecodeError:
                            continue

                elapsed_time = time.time() - start_time

                if not ttft_measured:
                    first_token_time = elapsed_time

                return True, elapsed_time, first_token_time, ""
            else:
                return False, 0.0, 0.0, f"HTTP {response.status_code}"

        except requests.exceptions.Timeout:
            return False, 0.0, 0.0, "Timeout"
        except requests.exceptions.ConnectionError:
            return False, 0.0, 0.0, "Connection Error"
        except Exception as e:
            return False, 0.0, 0.0, str(e)

    def check_connection(self) -> bool:
        try:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            response = requests.get(f"{self.base_url}/v1/models", headers=headers, timeout=5)
            return response.status_code == 200
        except:
            return False


class VLLMAdapter(OpenAICompatibleAdapter):
    """Adapter for vLLM - uses OpenAI-compatible API"""
    pass


class LMStudioAdapter(OpenAICompatibleAdapter):
    """Adapter for LM Studio - uses OpenAI-compatible API"""
    pass


class LlamaCppAdapter(OpenAICompatibleAdapter):
    """Adapter for llama.cpp server - uses OpenAI-compatible API"""
    pass


def create_adapter(api_type: str, base_url: str, api_key: Optional[str] = None) -> APIAdapter:
    """
    Factory function to create the appropriate adapter

    Args:
        api_type: Type of API (ollama, vllm, lmstudio, llamacpp, openai)
        base_url: Base URL of the API
        api_key: Optional API key for authentication

    Returns:
        APIAdapter instance
    """
    api_type = api_type.lower()

    adapters = {
        'ollama': OllamaAdapter,
        'vllm': VLLMAdapter,
        'lmstudio': LMStudioAdapter,
        'llamacpp': LlamaCppAdapter,
        'openai': OpenAICompatibleAdapter,
    }

    adapter_class = adapters.get(api_type)
    if not adapter_class:
        raise ValueError(f"Unknown API type: {api_type}. Supported types: {', '.join(adapters.keys())}")

    return adapter_class(base_url, api_key)
