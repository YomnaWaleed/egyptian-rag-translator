# src/ llm/ollama_client.py
"""
Ollama Cloud API client for LLM translation
"""
import requests
from src.config import settings


class OllamaClient:
    """Client for Ollama Cloud API"""

    def __init__(self, api_key=None, api_url=None, model=None):
        """
        Initialize Ollama client

        Args:
            api_key: Ollama API key
            api_url: Ollama API endpoint URL
            model: Model name to use
        """
        self.api_key = api_key or settings.OLLAMA_API_KEY
        self.api_url = api_url or settings.OLLAMA_API_URL
        self.model = model or settings.LLM_MODEL

        if not self.api_key:
            raise ValueError("OLLAMA_API_KEY not set")

        print(f"🔧 Ollama Client initialized")
        print(f"   Model: {self.model}")
        print(f"   API URL: {self.api_url}")

    def generate(self, system_prompt, user_prompt, timeout=240):
        """
        Generate response from Ollama

        Args:
            system_prompt: System prompt
            user_prompt: User prompt
            timeout: Request timeout in seconds

        Returns:
            tuple: (response_text, full_response_dict)
        """
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }

            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
            }

            response = requests.post(
                self.api_url, headers=headers, json=payload, timeout=timeout
            )

            if response.status_code == 200:
                result = response.json()
                # Ollama format uses 'message' -> 'content'
                llm_output = result["message"]["content"]
                return llm_output, result
            else:
                print(f"❌ API Error: {response.status_code}")
                print(f"   Response: {response.text}")
                return None, None

        except Exception as e:
            print(f"❌ LLM Error: {e}")
            return None, None

    def chat(self, messages, timeout=240):
        """
        Send chat messages to Ollama

        Args:
            messages: List of message dicts with 'role' and 'content'
            timeout: Request timeout in seconds

        Returns:
            tuple: (response_text, full_response_dict)
        """
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }

            payload = {"model": self.model, "messages": messages, "stream": False}

            response = requests.post(
                self.api_url, headers=headers, json=payload, timeout=timeout
            )

            if response.status_code == 200:
                result = response.json()
                llm_output = result["message"]["content"]
                return llm_output, result
            else:
                print(f"❌ API Error: {response.status_code}")
                print(f"   Response: {response.text}")
                return None, None

        except Exception as e:
            print(f"❌ LLM Error: {e}")
            return None, None


if __name__ == "__main__":
    # Test Ollama client
    client = OllamaClient()

    system_prompt = "You are a helpful assistant."
    user_prompt = "Say hello in German."

    response, full_response = client.generate(system_prompt, user_prompt)

    if response:
        print(f"\n✅ Response: {response}")
    else:
        print(f"\n❌ Failed to get response")
