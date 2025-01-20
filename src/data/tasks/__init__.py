TASK_CONFIG = {
    "dipromats_t1": {
        "simple": "This task (Propaganda Identification) consists on determining whether in a tweet propaganda techniques are used or not. This is a classification task and the labels are 'true' or 'false'.",
    },
    "dipromats_t2": {
        "simple": "This task (Coarse propaganda characterisation) seeks to classify the tweet into five multi-labels classes of propaganda techniques: '1 appeal to commonality', '2 discrediting the opponent', '3 loaded language', '4 appeal to authority', and 'false'. The 'false' class is mutually exclusive with the other classes.",
        "detail": "",
    },
    "dipromats_t3": {
        "simple": "This task (Fine - grained propaganda characterisation) consists on categorising propagandistic tweets into 16 multi-labels propaganda techniques: '1 appeal to commonality - ad populum', '1 appeal to commonality - flag waving', '2 discrediting the opponent - absurdity appeal', '2 discrediting the opponent - demonization', '2 discrediting the opponent - doubt', '2 discrediting the opponent - fear appeals (destructive)', '2 discrediting the opponent - name calling', '2 discrediting the opponent - propaganda slinging', '2 discrediting the opponent - scapegoating', '2 discrediting the opponent - personal attacks', '2 discrediting the opponent - undiplomatic assertiveness/whataboutism', '2 discrediting the opponent - reductio ad hitlerum', '3 loaded language', '4 appeal to authority - appeal to false authority', '4 appeal to authority - bandwagoning', and 'false'. The 'false' class is mutually exclusive with the other classes.",
        "detail": "",
    },
    "exist_2022_t1": {
        "simple": "This task  (Sexism Identification) consists on a binary classification task where systems have to decide whether or not a given tweet is sexist. The labels are 'sexist' and 'non-sexist'."
    },
    "exist_2022_t2": {
        "simple": "This task  (Sexism Categorization) consists on a six-class, mono-label classification task where each sexist tweet must be classified according to the type of sexism. The labels are 'sexual-violence', 'stereotyping-dominance', 'non-sexist', 'misogyny-non-sexual-violence', 'objectification', and 'ideological-inequality'."
    },
    "exist_2023_t1": {
        "simple": "This task (Sexism Identification) is a binary classification task where systems have to decide whether or not a given tweet is sexist. The labels are 'YES' or 'NO'. The prediction for each instance is the set of probabilities of the possible labels, i.e., 'YES' and 'NO'.",
    },
    "exist_2023_t2": {
        "simple": "This task (Source Intention) is a four-class, mono-label classification task where each sexist tweet must be classified according to the intention of the person who wrote it. The labels are 'DIRECT', 'JUDGEMENTAL', 'REPORTED' or 'NO'. The prediction for each instance is the set of probabilities of the possible labels, i.e., 'DIRECT', 'JUDGEMENTAL', 'REPORTED', and 'NO'.",
    },
    "exist_2023_t3": {
        "simple": "This task (Sexism Categorization) is a six-class, multi-label classification task where each sexist tweet must be classified according to the type of sexism. The labels are 'IDEOLOGICAL-INEQUALITY', 'MISOGYNY-NON-SEXUAL-VIOLENCE', 'OBJECTIFICATION', 'SEXUAL-VIOLENCE', 'STEREOTYPING-DOMINANCE', and 'NO'. The prediction for each instance is the set of probabilities of the possible labels, i.e., 'IDEOLOGICAL-INEQUALITY', 'MISOGYNY-NON-SEXUAL-VIOLENCE', 'OBJECTIFICATION', 'SEXUAL-VIOLENCE', 'STEREOTYPING-DOMINANCE', and 'NO'.",
    },
}
