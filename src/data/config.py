from src.data.prompt_syntax import BasicSyntax, CustomSyntax
from src.data.tasks import (
    Diann2023T1PromptBuilderBIO,
    Diann2023T1PromptBuilderTokenIdentification,
    Diann2023T1PromptBuilderGenerativeNER,
    Diann2023T1ContextualPromptBuilderNER,
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
from pyevall.utils.utils import PyEvALLUtils

CLASS_BUILDER = "class_builder"
SYSTEM_PROMPT = "system_prompt"
PROMPT_SYNTAX = "syntax_prompt"
TEXT_KEY = "text_key"
TRANSFORM = "transform"
K = "k"
EVALUATION = "evaluation"
USE_BIO = "use_bio"


DIPROMATS_TASK3 = {
    "True": {
        "1 appeal to commonality": [
            "1 appeal to commonality - ad populum",
            "1 appeal to commonality - flag waving",
        ],
        "2 discrediting the opponent": [
            "2 discrediting the opponent - absurdity appeal",
            "2 discrediting the opponent - demonization",
            "2 discrediting the opponent - doubt",
            "2 discrediting the opponent - fear appeals (destructive)",
            "2 discrediting the opponent - name calling",
            "2 discrediting the opponent - propaganda slinging",
            "2 discrediting the opponent - scapegoating",
            "2 discrediting the opponent - undiplomatic assertiveness/whataboutism",
        ],
        "3 loaded language": [],
        "4 appeal to authority": [
            "4 appeal to authority - appeal to false authority",
            "4 appeal to authority - bandwagoning",
        ],
    },
    "False": [],
}

DIPROMATS_TASK2 = {
    "True": [
        "1 appeal to commonality",
        "2 discrediting the opponent",
        "3 loaded language",
        "4 appeal to authority",
    ],
    "False": [],
}

TASK_CONFIG = {
    "diann_2023_t1": [
        {
            CLASS_BUILDER: Diann2023T1PromptBuilderBIO(
                prompt_start="The task is a sequence labeling task that follows a BIO format."
                "Each token in the abstract must be annotated as being at the beginning (“B-DIS”), inside (“I-DIS”) or outside (“O”) the mention of a disability.",
                prompt_guide="Here are some examples to guide you:",
                prompt_end="Now, label the following sequence:",
            ),
            SYSTEM_PROMPT: "You are an expert at solving sequence labeling problems.",
            PROMPT_SYNTAX: BasicSyntax(),
            TEXT_KEY: "tokens",
            TRANSFORM: lambda row: " ".join(row["tokens"]),
            USE_BIO: True,
        },
        {
            CLASS_BUILDER: Diann2023T1PromptBuilderTokenIdentification(
                prompt_start=(
                    "Your task is to identify all disability mentions in the text. "
                    "For each mention, please output a list of the tokens (in the correct order) "
                    "that make up that mention. "
                    "If the text does not contain any disability mentions, return an empty list."
                ),
                prompt_guide="Here are some examples to guide you:",
                prompt_end="Now, extract the disability mentions as lists of tokens from the following sequence:",
            ),
            SYSTEM_PROMPT: "You are an expert at identifying disability mentions in text.",
            PROMPT_SYNTAX: BasicSyntax(),
            TEXT_KEY: "tokens",
            TRANSFORM: lambda row: " ".join(row["tokens"]),
            USE_BIO: False,
        },
        {
            CLASS_BUILDER: Diann2023T1PromptBuilderGenerativeNER(
                prompt_start=(
                    "Your task is to identify all disability mentions in the text. "
                    "For each mention, please output the disability present (if any) in that mention."
                    "Your answer must be a list of strings, where each string is the explicit mention of a disability."
                ),
                prompt_guide="Here are some examples to guide you:",
                prompt_end="Now, extract the disability mentions from the following sequence:",
            ),
            SYSTEM_PROMPT: "You are an expert linguist in English and Spanish. You are an expert at identifying disability mentions in text.",
            PROMPT_SYNTAX: BasicSyntax(),
            TEXT_KEY: "tokens",
            TRANSFORM: lambda row: " ".join(row["tokens"]),
            USE_BIO: False,
        },
        {
            CLASS_BUILDER: Diann2023T1ContextualPromptBuilderNER(
                prompt_start=(
                    "Your task is to identify all disability mentions in the text. "
                    "For each mention, please output the disability present (if any) in that mention."
                    "Your answer must be a list of strings, where each string is the explicit mention of a disability."
                )
            ),
            SYSTEM_PROMPT: "You are an expert at identifying disability mentions in text.",
            PROMPT_SYNTAX: CustomSyntax(
                prompt_guide="Here are some examples of disabilities to guide you:",
                prompt_end="Now, extract the disability mentions from the following sequence:",
            ),
            TEXT_KEY: "tokens",
            TRANSFORM: lambda row: " ".join(row["tokens"]),
            USE_BIO: False,
        },
    ],
    "dipromats_2023_t1": [
        {
            CLASS_BUILDER: DipromatsT1PromptBuilder(
                prompt_start="This task (Propaganda Identification) consists on determining whether in a tweet propaganda techniques are used or not. "
                "This is a classification task and the labels are 'true' or 'false'.",
                prompt_guide="Here are some examples to guide you:",
                prompt_end="Now, classify the following text:",
            ),
            SYSTEM_PROMPT: "You are an expert linguist in English and Spanish. You are an expert at solving binary classification problems.",
            PROMPT_SYNTAX: BasicSyntax(),
            TEXT_KEY: "text",
            EVALUATION: {},
        }
    ],
    "dipromats_2023_t2": [
        {
            CLASS_BUILDER: DipromatsT2PromptBuilder(
                prompt_start="Coarse propaganda characterisation) seeks to classify the tweet into five multi-labels classes of propaganda techniques:"
                "'1 appeal to commonality', '2 discrediting the opponent', '3 loaded language', '4 appeal to authority', and 'false'."
                "The 'false' class is mutually exclusive with the other classes.",
                prompt_guide="Here are some examples to guide you:",
                prompt_end="Now, classify the following text:",
            ),
            SYSTEM_PROMPT: "You are an expert linguist in English and Spanish. You are an expert at solving multi-label classification problems.",
            PROMPT_SYNTAX: BasicSyntax(),
            TEXT_KEY: "text",
            EVALUATION: {PyEvALLUtils.PARAM_HIERARCHY: DIPROMATS_TASK2},
        }
    ],
    "dipromats_2023_t3": [
        {
            CLASS_BUILDER: DipromatsT3PromptBuilder(
                prompt_start="This task (Fine - grained propaganda characterisation) consists on categorising propagandistic tweets into 16 multi-labels propaganda techniques:"
                "'1 appeal to commonality - ad populum',"
                "'1 appeal to commonality - flag waving',"
                "'2 discrediting the opponent - absurdity appeal',"
                "'2 discrediting the opponent - demonization',"
                "'2 discrediting the opponent - doubt',"
                "'2 discrediting the opponent - fear appeals (destructive)',"
                "'2 discrediting the opponent - name calling',"
                "'2 discrediting the opponent - propaganda slinging',"
                "'2 discrediting the opponent - scapegoating',"
                "'2 discrediting the opponent - personal attacks',"
                "'2 discrediting the opponent - undiplomatic assertiveness/whataboutism',"
                "'2 discrediting the opponent - reductio ad hitlerum',"
                "'3 loaded language',"
                "'4 appeal to authority - appeal to false authority',"
                "'4 appeal to authority - bandwagoning',"
                "and 'false'."
                "The 'false' class is mutually exclusive with the other classes.",
                prompt_guide="Here are some examples to guide you:",
                prompt_end="Now, classify the following text:",
            ),
            SYSTEM_PROMPT: "You are an expert linguist in English and Spanish. You are an expert at solving multi-label classification problems.",
            PROMPT_SYNTAX: BasicSyntax(),
            TEXT_KEY: "text",
            EVALUATION: {PyEvALLUtils.PARAM_HIERARCHY: DIPROMATS_TASK3},
        }
    ],
    "exist_2022_t1": [
        {
            CLASS_BUILDER: Exist2022T1PromptBuilder(
                prompt_start="This task  (Sexism Identification) consists on a binary classification task where systems have to decide whether or not a given tweet is sexist."
                "The labels are 'sexist' and 'non-sexist'.",
                prompt_guide="Here are some examples to guide you:",
                prompt_end="Now, classify the following text:",
            ),
            SYSTEM_PROMPT: "You are an expert linguist in English and Spanish. You are an expert at solving binary classification problems.",
            PROMPT_SYNTAX: BasicSyntax(),
            TEXT_KEY: "text",
        }
    ],
    "exist_2022_t2": [
        {
            CLASS_BUILDER: Exist2022T2PromptBuilder(
                prompt_start="This task  (Sexism Categorization) consists on a six-class, mono-label classification task where each sexist tweet must be classified according to the type of sexism."
                "The labels are 'sexual-violence', 'stereotyping-dominance', 'non-sexist', 'misogyny-non-sexual-violence', 'objectification', and 'ideological-inequality'.",
                prompt_guide="Here are some examples to guide you:",
                prompt_end="Now, classify the following text:",
            ),
            SYSTEM_PROMPT: "You are an expert linguist in English and Spanish. You are an expert at solving mono-label classification problems.",
            PROMPT_SYNTAX: BasicSyntax(),
            TEXT_KEY: "text",
        }
    ],
    "exist_2023_t1": [
        {
            CLASS_BUILDER: Exist2023T1PromptBuilder(
                prompt_start="This task (Sexism Identification) is a binary classification task where systems have to decide whether or not a given tweet is sexist."
                "The labels are 'YES' or 'NO'."
                "The prediction for each instance is the set of probabilities of the possible labels, i.e., 'YES' and 'NO'.",
                prompt_guide="Here are some examples to guide you:",
                prompt_end="Now, classify the following tweet:",
            ),
            SYSTEM_PROMPT: "You are an expert linguist in English and Spanish. You are an expert at solving binary classification problems.",
            PROMPT_SYNTAX: BasicSyntax(),
            TEXT_KEY: "text",
        }
    ],
    "exist_2023_t2": [
        {
            CLASS_BUILDER: Exist2023T2PromptBuilder(
                prompt_start="This task (Source Intention) is a four-class, mono-label classification task where each sexist tweet must be classified according to the intention of the person who wrote it."
                "The labels are 'DIRECT', 'JUDGEMENTAL', 'REPORTED' or 'NO'."
                "The prediction for each instance is the set of probabilities of the possible labels, i.e., 'DIRECT', 'JUDGEMENTAL', 'REPORTED', and 'NO'.",
                prompt_guide="Here are some examples to guide you:",
                prompt_end="Now, classify the following tweet:",
            ),
            SYSTEM_PROMPT: "You are an expert linguist in English and Spanish. You are an expert at solving mono-label classification problems.",
            PROMPT_SYNTAX: BasicSyntax(),
            TEXT_KEY: "text",
        }
    ],
    "exist_2023_t3": [
        {
            CLASS_BUILDER: Exist2023T3PromptBuilder(
                prompt_start="This task (Sexism Categorization) is a six-class, multi-label classification task where each sexist tweet must be classified according to the type of sexism."
                "The labels are 'IDEOLOGICAL-INEQUALITY', 'MISOGYNY-NON-SEXUAL-VIOLENCE', 'OBJECTIFICATION', 'SEXUAL-VIOLENCE', 'STEREOTYPING-DOMINANCE', and 'NO'."
                "The prediction for each instance is the set of probabilities of the possible labels, i.e., 'IDEOLOGICAL-INEQUALITY', 'MISOGYNY-NON-SEXUAL-VIOLENCE', 'OBJECTIFICATION', 'SEXUAL-VIOLENCE', 'STEREOTYPING-DOMINANCE', and 'NO'.",
                prompt_guide="Here are some examples to guide you:",
                prompt_end="Now, classify the following tweet:",
            ),
            SYSTEM_PROMPT: "You are an expert linguist in English and Spanish. You are an expert at solving multi-label classification problems.",
            PROMPT_SYNTAX: BasicSyntax(),
            TEXT_KEY: "text",
        }
    ],
    "sqac_squad_2024_t1": [
        {
            CLASS_BUILDER: SqacSquad2024PromptBuilder(
                prompt_start="For this task, find the shortest span needed to answer the question."
                "The texts are academic news from CSIC (for Spanish) and Cambridge University (for English)."
                "In all cases, the answers are fragments of the text and all questions can be answered from the text.",
                prompt_guide="Here are some examples to guide you:",
                prompt_end="Now, find the shortest span that answer the following question:",
            ),
            SYSTEM_PROMPT: "You are an expert linguist in English and Spanish. You are an expert at answering questions in an extractive manner.",
            PROMPT_SYNTAX: BasicSyntax(),
            TEXT_KEY: "question",
        },
    ],
}

BATCH_SIZE = {
    "diann_2023_t1": 8,
    "dipromats_2023_t1": 128,
    "dipromats_2023_t2": 128,
    "dipromats_2023_t3": 128,
    "exist_2023_t1": 128,
    "exist_2023_t2": 128,
    "exist_2023_t3": 128,
    "exist_2022_t1": 128,
    "exist_2022_t2": 128,
    "sqac_squad_2024_t1": 2,
}

RELATIVE_BATCH_SIZE = {
    "diann_2023_t1": [{0: 8, 5: 8}, {0: 8, 5: 8}, {0: 8, 5: 8}, {0: 8, 5: 8}],
    "dipromats_2023_t1": [{0: 64, 5: 8}],
    "dipromats_2023_t2": [{0: 64, 5: 8}],
    "dipromats_2023_t3": [{0: 64, 5: 8}],
    "exist_2023_t1": [{0: 64, 5: 8}],
    "exist_2023_t2": [{0: 64, 5: 8}],
    "exist_2023_t3": [{0: 64, 5: 8}],
    "exist_2022_t1": [{0: 64, 5: 8}],
    "exist_2022_t2": [{0: 64, 5: 8}],
    "sqac_squad_2024_t1": [{0: 4, 5: 1}],
}

CLASSES_DIPROMATS_2023_T1 = ["true", "false"]
CLASSES_DIPROMATS_2023_T2 = [
    "1 appeal to commonality",
    "2 discrediting the opponent",
    "3 loaded language",
    "4 appeal to authority",
    "false",
]
CLASSES_DIPROMATS_2023_T3 = [
    "1 appeal to commonality - ad populum",
    "1 appeal to commonality - flag waving",
    "2 discrediting the opponent - absurdity appeal",
    "2 discrediting the opponent - demonization",
    "2 discrediting the opponent - doubt",
    "2 discrediting the opponent - fear appeals (destructive)",
    "2 discrediting the opponent - name calling",
    "2 discrediting the opponent - propaganda slinging",
    "2 discrediting the opponent - scapegoating",
    "2 discrediting the opponent - personal attacks",
    "2 discrediting the opponent - undiplomatic assertiveness/whataboutism",
    "2 discrediting the opponent - reductio ad hitlerum",
    "3 loaded language",
    "4 appeal to authority - appeal to false authority",
    "4 appeal to authority - bandwagoning",
    "false",
]

CLASSES_EXIST_2022_T1 = ["sexist", "non-sexist"]
CLASSES_EXIST_2022_T2 = [
    "sexual-violence",
    "stereotyping-dominance",
    "non-sexist",
    "misogyny-non-sexual-violence",
    "objectification",
    "ideological-inequality",
]
CLASSES_EXIST_2023_T1 = ["YES", "NO"]
CLASSES_EXIST_2023_T2 = ["DIRECT", "JUDGEMENTAL", "REPORTED", "NO"]
CLASSES_EXIST_2023_T3 = [
    "IDEOLOGICAL-INEQUALITY",
    "MISOGYNY-NON-SEXUAL-VIOLENCE",
    "OBJECTIFICATION",
    "SEXUAL-VIOLENCE",
    "STEREOTYPING-DOMINANCE",
    "NO",
]

CLASSES_DIANN_2023_T3 = ["O", "B-DIS", "I-DIS"]
