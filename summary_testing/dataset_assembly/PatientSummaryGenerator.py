import time
import random
import json
import os

class PatientSummaryGenerator:

    def __init__(self, llm_caller, system_prompt):

        self._llm_caller = llm_caller
        self._system_prompt = system_prompt

    @property
    def system_prompt(self):
        return self._system_prompt

    @property
    def llm_caller(self):
        return self._llm_caller

    def generate(self, df, output_path):

        caller = self.llm_caller
        system_prompt = self.system_prompt

        def summarize_with_token_count(input_text):

            result = caller.invoke(system_template=system_prompt, input_string=input_text)
            if isinstance(result, type(None)):
                summary = "MODEL ERROR"
                token_count = 0
            else:
                summary = result["candidates"][0]["content"]["parts"][0]["text"]
                # this will be a string
                token_count = json.dumps(caller.token_counter(result))
            return {
                "summary": summary,
                "token_count": token_count
            }

        def summarize(example):

            retries = 0
            max_retries = 2

            model_input = example["model_input"]
            input_text = f"""
            Here is the text that you must summarize:
            {model_input}
                """
            while retries < max_retries:
                try:
                    return summarize_with_token_count(input_text)
                except Exception as e:
                    # very simple backoff
                    sleep_time = (2 ** retries) + random.random()
                    time.sleep(sleep_time)
                    retries += 1
            return {
                "summary": "Failed after maximum retries",
                "token_count": 0
            }

        summarized_dataset = df.map(summarize)
        if output_path:
            output_file_splits = os.path.splitext(output_path)
            if output_file_splits[1] != ".json":
                output_path = output_file_splits[0] + ".json"
            summarized_dataset.to_json(output_path)
        return summarized_dataset