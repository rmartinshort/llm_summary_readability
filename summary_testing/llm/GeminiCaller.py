import json
from google import genai
from dataclasses import dataclass
from typing import Dict, Any


class GeminiCaller:
    def __init__(self, api_key: str, model: str):
        """
        Initializes the GeminiStructuredOutputCaller with an API key and model name.

        Args:
            api_key (str): The API key for accessing the Gemini LLM.
            model (str): The name of the Gemini model to use.
        """
        self.api_key = api_key
        self.model_name = model
        self.client = genai.Client(api_key=api_key)

    def invoke(
        self,
        system_template: dataclass,
        input_string: str,
        temperature: float = 0.0,
        max_tokens: str = 1000,
    ) -> Dict[str, Any]:
        try:
            res = self.client.models.generate_content(
                model=self.model_name,
                contents=[input_string],
                config=genai.types.GenerateContentConfig(
                    system_instruction=system_template.system_message,
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                ),
            )
            res = json.loads(res.model_dump_json())
        except Exception as e:
            print(f"Error calling {self.model_name}: {e}")
            res = {}

        return res

    @staticmethod
    def token_counter(model_output: Any) -> Dict[str, int]:
        """
        Counts the number of input and output tokens used by the Gemini LLM.

        Args:
            model_output (Dict[str, Any]): The output from the Gemini LLM.

        Returns:
            Dict[str, int]: A dictionary containing the number of input and output tokens.
        """
        usage_stats = model_output["usage_metadata"]
        input_tokens = usage_stats["prompt_token_count"]
        output_tokens = usage_stats["candidates_token_count"]
        return {"input_tokens": input_tokens, "output_tokens": output_tokens}
