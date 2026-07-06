"""LLM client for API calls."""

import json
import logging
import re
from typing import Any, Optional
from openai import OpenAI
# Add import at top (if google-genai not installed, install first)
from google import genai
from google.genai import types
import time


logger = logging.getLogger(__name__)

import time
class LLMClient:
    """Base class for LLM clients."""

    def __init__(self, model_name: str, base_url: str, api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key
        self.client = OpenAI(api_key=self.api_key, base_url=base_url)


    def extract_json_from_response(self, response: str) -> Any:
        """Extract JSON from LLM response (could be dict or list)."""
        if not response:
            return None

        # 1. Preferentially extract markdown code blocks
        code_block_pattern = r'```(?:json|JSON)?\s*\n?(.*?)\n?```'
        matches = re.findall(code_block_pattern, response, re.DOTALL)
        for candidate in matches:
            parsed = self._try_parse_json(candidate.strip())
            if parsed is not None:
                return parsed

        # 2. Try to extract outermost {...} or [...]
        for pattern in (r'(\{.*\})', r'(\[.*\])'):
            # Use non-greedy but spanning the whole string with multiline
            # Actually we need balanced parentheses, so use custom function
            balanced = self._extract_balanced_json(response, pattern[1])
            if balanced:
                parsed = self._try_parse_json(balanced)
                if parsed is not None:
                    return parsed

        # 3. Finally, try to parse the entire string directly
        return self._try_parse_json(response)

    def _extract_balanced_json(self, text: str, open_char: str = '{') -> str:
        """Extract the outermost balanced JSON fragment (supports {} or [])."""
        close_char = '}' if open_char == '{' else ']'
        start = text.find(open_char)
        if start == -1:
            return ""
        stack = 0
        for i in range(start, len(text)):
            if text[i] == open_char:
                stack += 1
            elif text[i] == close_char:
                stack -= 1
                if stack == 0:
                    return text[start:i+1]
        return ""

    def _try_parse_json(self, s: str) -> Any:
        """Try to parse JSON, automatically fixing common issues."""
        if not s:
            return None
        s = s.strip()
        # Remove control characters (except necessary whitespace)
        s = ''.join(ch for ch in s if ord(ch) >= 32 or ch in '\n\r\t')
        # Try direct parsing
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass

        # Fix trailing commas (common in LLM-generated JSON)
        try:
            s_fixed = re.sub(r',\s*}', '}', s)   # {...,} -> {...}
            s_fixed = re.sub(r',\s*]', ']', s_fixed) # [...,] -> [...]
            return json.loads(s_fixed)
        except json.JSONDecodeError:
            pass

        # Try replacing single quotes with double quotes (imperfect but often works)
        try:
            # Only replace paired single quotes, avoid breaking existing double quotes
            s_fixed = re.sub(r"(?<!\\)'([^'\\]*(?:\\.[^'\\]*)*)'", r'"\1"', s)
            return json.loads(s_fixed)
        except json.JSONDecodeError:
            pass

        # If still failing, try ast.literal_eval (more lenient)
        try:
            import ast
            return ast.literal_eval(s)
        except (ValueError, SyntaxError):
            pass

        logger.error(f"Failed to parse JSON after all attempts: {s[:200]}")
        return None

    def call(self, prompt: str, **kwargs) -> Any:
        """Abstract method for calling LLM."""
        raise NotImplementedError


class OpenAIClient(LLMClient):
    """OpenAI API client."""

    def call(self, prompt: str, temperature: float = 0.1, max_tokens: int = 8190) -> Any:
        """Call OpenAI API."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                stream=False,
                temperature=0
              #  response_format={ "type": "json_object" }
            )
            content = response.choices[0].message.content
            #logger.debug(f"LLM response: {content[:100]}...")
            return content
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            return ""
    def call_with_history(self, messages, temperature: float = 0.1, max_tokens: int = 8190) -> Any:
        """Call OpenAI API with history messages."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                stream=False
              #  response_format={ "type": "json_object" }
            )
            content = response.choices[0].message.content
            #logger.debug(f"LLM response: {content[:100]}...")
            return content
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            return ""
    def call_with_evidence(self, messages, temperature: float = 0.1, max_tokens: int = 8190) -> Any:
        """Call OpenAI API and return the full message object."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                stream=False
              #  response_format={ "type": "json_object" }
            )
            content = response.choices[0].message
            #logger.debug(f"LLM response: {content[:100]}...")
            return content
        except Exception as e:  
            logger.error(f"OpenAI API call failed: {e}")
            return None
        
    def call_with_image(self, prompt: str, image_url: str, temperature=0.0, max_tokens=4096):
        # response = self.client.chat.completions.create(
        #     model=self.model_name,  # or gpt-4o
        #     messages=[
        #         {
        #             "role": "user",
        #             "content": [
        #                 {"type": "text", "text": prompt},
        #                 {"type": "image_url", "image_url": {"url": image_url}}
        #             ]
        #         }
        #     ]
        # )
        import base64
        def encode_image_to_base64(image_path: str) -> str:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        #image_path = "test_data/preditction/piplines/pdf-9f4f0167-6f3e-4386-8cbd-db18f1808c5b/images/table_p6_b28_69_153_526_342.png"
        image_url = f"data:image/png;base64,{encode_image_to_base64(image_url)}"
        response = self.client.chat.completions.create(
            model="qwen-vl-plus",  # Using qwen-vl-plus as an example; you can replace with other model names. Model list: https://help.aliyun.com/zh/model-studio/getting-started/models
            messages=[{"role": "user",
                       "content": [
                            {"type": "image_url",
                            "image_url": {"url": image_url}},
                            {"type": "text", "text": prompt},
                    ]}]
    )   
        #print(response.choices[0].message.content)
        return response.choices[0].message.content

    def call_with_usage(self, prompt: str, **kwargs):
        """Return (content, usage) tuple; usage may be None."""
        
        try:
            st = time.time()
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            )
            content = response.choices[0].message.content
            usage = response.usage
            cost = time.time() - st
            logger.info(f"LLM call success, cost: {cost}s, usage: {usage}")
            return content, [usage.completion_tokens, usage.prompt_tokens, usage.total_tokens, cost]
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return "", ""
        
class MockLLMClient(LLMClient):
    """Mock client for testing."""

    def call(self, prompt: str, **kwargs) -> Any:
        """Mock call for testing."""
        return """```json
[
  {
    "catalyst_name": "Test Catalyst",
    "yield_g": 5.2,
    "additional_items": [
      {
        "item_name": "test_field",
        "define": "Test field definition",
        "value": "test_value",
        "suggested_data_type": "String"
      }
    ]
  }
]
```"""

# Add GeminiClient as a subclass of LLMClient
class GeminiClient(LLMClient):
    """Google Gemini API client."""

    def __init__(self, model_name: str, api_key: str, base_url: Optional[str] = None):
        # Gemini usually does not require base_url, but keep parameter for compatibility with parent
        super().__init__(model_name, base_url, api_key)
        # Initialize Gemini client (using API Key)
        self.client = genai.Client(api_key=self.api_key)

    def call(self, prompt: str, temperature: float = 0.0, **kwargs) -> Any:
        """Call Gemini API, return text content."""
        try:
            # Build request config
            config = types.GenerateContentConfig(
                temperature=temperature,
                # Add other supported parameters here, e.g., top_p, top_k, etc.
                **{k: v for k, v in kwargs.items() if k in ['top_p', 'top_k', 'stop_sequences', 'system_instruction']}
            )
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config
            )
            # Extract text content
            content = response.text
            logger.debug(f"Gemini response: {content[:500]}...")
            return content
        except Exception as e:
            logger.error(f"Gemini API call failed: {e}")
            return None

    def call_with_image(self, prompt: str, image_base64: str, temperature: float = 0.0, max_tokens: int = 4096) -> str:
        """Multimodal call: text + image (base64 encoded)."""
        try:
            # Gemini supports inline data or file upload; here we use inline base64
            # Note: The base64 string should not include the data:image/... prefix
            from PIL import Image
            import io
            import base64

            # Decode base64 to image bytes
            image_data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_data))

            config = types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[prompt, image],
                config=config
            )
            return response.text
        except Exception as e:
            logger.error(f"Gemini call_with_image failed: {e}")
            return ""

    def call_with_usage(self, prompt: str, **kwargs) -> tuple:
        """Return (content, usage_info) tuple; usage_info format compatible with OpenAIClient."""
        try:
            start_time = time.time()
            config = types.GenerateContentConfig(
                temperature=kwargs.get('temperature', 0.1),
                max_output_tokens=kwargs.get('max_tokens', 8192),
            )
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config
            )
            content = response.text
            cost = time.time() - start_time

            # Extract usage statistics from Gemini if available
            # Gemini SDK may have usage_metadata attribute on response
            usage_tokens = [0, 0, 0]  # [completion_tokens, prompt_tokens, total_tokens]
            if hasattr(response, 'usage_metadata'):
                prompt_tokens = response.usage_metadata.prompt_token_count
                candidates_tokens = response.usage_metadata.candidates_token_count
                total_tokens = response.usage_metadata.total_token_count
                usage_tokens = [candidates_tokens, prompt_tokens, total_tokens]
            logger.info(f"Gemini call success, cost: {cost:.2f}s, usage: {usage_tokens}")
            return content, usage_tokens + [cost]
        except Exception as e:
            logger.error(f"Gemini call_with_usage failed: {e}")
            return "", ""
        
if __name__ == "__main__":
    # Simple test
    # Initialize Gemini client
    gemini = GeminiClient(
        model_name="gemini-2.5-flash",
        api_key=""
    )

    # # Normal call
    # response = gemini.call("Tell me a joke about programmers.", temperature=0.8)
    # print(response)

    # Call with usage stats
    content, usage = gemini.call_with_usage("Explain what recursion is", temperature=0.2)
    print(content, usage)