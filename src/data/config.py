from src.data.prompt_syntax import BasicSyntax
from src.data.tasks import (
    Diann2023T1PromptBuilderBIO,
    DipromatsT1PromptBuilder,
    DipromatsT2PromptBuilder,
    DipromatsT3PromptBuilder,
    Exist2022T1PromptBuilder,
    Exist2022T2PromptBuilder,
    Exist2023T1PromptBuilder,
    Exist2023T2PromptBuilder,
    Exist2023T3PromptBuilder,
    SqacSquad2024PromptBuilder,
)

CLASS_BUILDER = "class_builder"
SYSTEM_PROMPT = "system_prompt"
PROMPT_SYNTAX = "syntax_prompt"
TEXT_KEY = "text_key"
TRANSFORM = "transform"

TASK_CONFIG = {
        "diann_2023_t1": {
            CLASS_BUILDER: Diann2023T1PromptBuilderBIO(),
            SYSTEM_PROMPT: None,
            PROMPT_SYNTAX: None,
            TEXT_KEY: "tokens",
            TRANSFORM: lambda row: " ".join(row["tokens"]),
        },
        "dipromats_2023_t1": {
            CLASS_BUILDER: DipromatsT1PromptBuilder(
                "This task (Propaganda Identification) consists on determining whether in a tweet propaganda techniques are used or not."
                "This is a classification task and the labels are 'true' or 'false'."
            ),
            SYSTEM_PROMPT: "You are a machine learning model that excels on solving classification problems.",
            PROMPT_SYNTAX: BasicSyntax(),
            TEXT_KEY: "text",
        },
        "dipromats_2023_t2": {
            CLASS_BUILDER: DipromatsT2PromptBuilder,
            SYSTEM_PROMPT: None,
            PROMPT_SYNTAX: None,
            TEXT_KEY: None,
        },
        "dipromats_2023_t3": {
            CLASS_BUILDER: DipromatsT3PromptBuilder,
            SYSTEM_PROMPT: None,
            PROMPT_SYNTAX: None,
            TEXT_KEY: None,
        },
        "exist_2022_t1": {
            CLASS_BUILDER: Exist2022T1PromptBuilder,
            SYSTEM_PROMPT: None,
            PROMPT_SYNTAX: None,
            TEXT_KEY: None,
        },
        "exist_2022_t2": {
            CLASS_BUILDER: Exist2022T2PromptBuilder,
            SYSTEM_PROMPT: None,
            PROMPT_SYNTAX: None,
            TEXT_KEY: None,
        },
        "exist_2023_t1": {
            CLASS_BUILDER: Exist2023T1PromptBuilder,
            SYSTEM_PROMPT: None,
            PROMPT_SYNTAX: None,
            TEXT_KEY: None,
        },
        "exist_2023_t2": {
            CLASS_BUILDER: Exist2023T2PromptBuilder,
            SYSTEM_PROMPT: None,
            PROMPT_SYNTAX: None,
            TEXT_KEY: None,
        },
        "exist_2023_t3": {
            CLASS_BUILDER: Exist2023T3PromptBuilder,
            SYSTEM_PROMPT: None,
            PROMPT_SYNTAX: None,
            TEXT_KEY: None,
        },
        "sqac_squad_2024_t1": {
            CLASS_BUILDER: SqacSquad2024PromptBuilder,
            SYSTEM_PROMPT: None,
            PROMPT_SYNTAX: None,
            TEXT_KEY: None,
        },
    }
