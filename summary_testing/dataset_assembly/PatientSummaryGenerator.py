import time
import random
import json
import os
from typing import Any


class PatientSummaryGenerator:
    """
    A class to generate patient summaries using a language model.

    Attributes:
        llm_caller (Any): The language model caller used for generating summaries.
        system_prompt (str): The system prompt to guide the language model.
    """

    def __init__(self, llm_caller: Any, system_prompt: str) -> None:
        """
        Initializes the PatientSummaryGenerator with the specified language model caller and system prompt.

        Args:
            llm_caller (Any): The language model caller for generating summaries.
            system_prompt (str): The system prompt for the language model.
        """
        self._llm_caller: Any = llm_caller
        self._system_prompt: str = system_prompt

    @property
    def system_prompt(self) -> str:
        """
        Returns the system prompt used by the language model.

        Returns:
            str: The system prompt.
        """
        return self._system_prompt

    @property
    def llm_caller(self) -> Any:
        """
        Returns the language model caller.

        Returns:
            Any: The language model caller.
        """
        return self._llm_caller

    def generate(self, df: Any, output_path: str, temperature: float = 0) -> Any:
        """
        Generates summaries for the provided dataset using the language model.

        Args:
            df (Any): The dataset containing patient information to summarize.
            output_path (str): The path to save the generated summaries in JSON format.
            temperature (float): The sampling temperature for the language model.

        Returns:
            Any: The dataset containing the generated summaries.
        """
        caller = self.llm_caller
        system_prompt = self.system_prompt

        def summarize_with_token_count(input_text: str, temperature: float) -> dict:
            """
            Summarizes the input text and counts the tokens used.

            Args:
                input_text (str): The text to summarize.
                temperature (float): The sampling temperature for the language model.

            Returns:
                dict: A dictionary containing the summary and token count.
            """
            result = caller.invoke(
                system_template=system_prompt,
                input_string=input_text,
                temperature=temperature,
            )
            if isinstance(result, type(None)):
                summary = "MODEL ERROR"
                token_count = 0
            else:
                summary = result["candidates"][0]["content"]["parts"][0]["text"]
                token_count = json.dumps(caller.token_counter(result))
            return {"summary": summary, "token_count": token_count}

        def summarize(example: dict, temperature: float) -> dict:
            """
            Summarizes a single example from the dataset with retry logic.

            Args:
                example (dict): The example to summarize.
                temperature (float): The sampling temperature for the language model.

            Returns:
                dict: A dictionary containing the summary and token count.
            """
            retries = 0
            max_retries = 2

            model_input = example["model_input"]
            input_text = f"""
            Here is the text that you must summarize:
            {model_input}
                """
            while retries < max_retries:
                try:
                    return summarize_with_token_count(input_text, temperature)
                except Exception as e:
                    sleep_time = (2**retries) + random.random()
                    time.sleep(sleep_time)
                    retries += 1
            return {"summary": "Failed after maximum retries", "token_count": 0}

        summarized_dataset = df.map(summarize, fn_kwargs={"temperature": temperature})
        if output_path:
            output_file_splits = os.path.splitext(output_path)
            if output_file_splits[1] != ".json":
                output_path = output_file_splits[0] + ".json"
            summarized_dataset.to_json(output_path)
        return summarized_dataset
