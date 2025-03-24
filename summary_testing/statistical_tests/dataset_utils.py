import json
import textstat
from summary_testing.config import COSTS


def syllables_per_word(input_text):
    n_words = textstat.lexicon_count(input_text, removepunct=True)
    n_syllables_per_word = textstat.syllable_count(input_text) / n_words
    return n_syllables_per_word


def text_analysis(input_row, model):
    model_output = input_row["summary"]
    model_input = input_row["model_input"]
    tokens = json.loads(input_row["token_count"])
    total_cost = (
        COSTS[model]["input"] * tokens["input_tokens"]
        + COSTS[model]["output"] * tokens["output_tokens"]
    )

    n_words_per_sentence_input = textstat.words_per_sentence(model_input)
    n_words_per_sentence_output = textstat.words_per_sentence(model_output)

    n_syllables_per_word_input = syllables_per_word(model_input)
    n_syllables_per_word_output = syllables_per_word(model_output)

    summary_reading_ease = textstat.flesch_reading_ease(model_output)
    input_reading_ease = textstat.flesch_reading_ease(model_input)

    return {
        "summary_words_per_sentence": n_words_per_sentence_output,
        "input_words_per_sentence": n_words_per_sentence_input,
        "summary_syllables_per_word": n_syllables_per_word_output,
        "input_syllables_per_word": n_syllables_per_word_input,
        "input_flesch_reading_ease": input_reading_ease,
        "summary_flesch_reading_ease": summary_reading_ease,
        "total_cost": total_cost,
    }


def merge_two_summary_datasets(
    df1,
    df2,
    df1_name="v1",
    df2_name="v2",
    model_df1="gemini-flash-2.0",
    model_df2="gemini-flash-2.0",
):
    df1_w_stats = df1.map(text_analysis, fn_kwargs={"model": model_df1})
    df2_w_stats = df2.map(text_analysis, fn_kwargs={"model": model_df2})

    df1_pd = (
        df1_w_stats["train"]
        .to_pandas()[
            [
                "patient_id",
                "model_input",
                "input_flesch_reading_ease",
                "input_syllables_per_word",
                "input_words_per_sentence",
                "summary",
                "summary_flesch_reading_ease",
                "summary_syllables_per_word",
                "summary_words_per_sentence",
                "total_cost",
            ]
        ]
        .rename(
            columns={
                "summary": f"{df1_name}_summary",
                "summary_flesch_reading_ease": f"{df1_name}_summary_flesch_reading_ease",
                "summary_syllables_per_word": f"{df1_name}_summary_syllables_per_word",
                "summary_words_per_sentence": f"{df1_name}_summary_words_per_sentence",
                "total_cost": f"{df1_name}_total_cost",
            }
        )
    )
    df2_pd = (
        df2_w_stats["train"]
        .to_pandas()[
            [
                "patient_id",
                "summary",
                "summary_flesch_reading_ease",
                "summary_syllables_per_word",
                "summary_words_per_sentence",
                "total_cost",
            ]
        ]
        .rename(
            columns={
                "summary": f"{df2_name}_summary",
                "summary_flesch_reading_ease": f"{df2_name}_summary_flesch_reading_ease",
                "summary_syllables_per_word": f"{df2_name}_summary_syllables_per_word",
                "summary_words_per_sentence": f"{df2_name}_summary_words_per_sentence",
                "total_cost": f"{df2_name}_total_cost",
            }
        )
    )
    dataset_pd = df1_pd.merge(df2_pd, on=["patient_id"])[
        [
            "patient_id",
            "model_input",
            "input_flesch_reading_ease",
            "input_syllables_per_word",
            "input_words_per_sentence",
            f"{df1_name}_summary",
            f"{df2_name}_summary",
            f"{df1_name}_summary_flesch_reading_ease",
            f"{df2_name}_summary_flesch_reading_ease",
            f"{df1_name}_summary_syllables_per_word",
            f"{df2_name}_summary_syllables_per_word",
            f"{df1_name}_summary_words_per_sentence",
            f"{df2_name}_summary_words_per_sentence",
            f"{df1_name}_total_cost",
            f"{df2_name}_total_cost",
        ]
    ]

    return dataset_pd
