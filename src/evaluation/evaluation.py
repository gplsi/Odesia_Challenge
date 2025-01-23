
from pyevall.evaluation import PyEvALLEvaluation
from pyevall.utils.utils import PyEvALLUtils
from pyevall.metrics.metricfactory import MetricFactory
import json

#PyEvALL para DIPROMATS 2023, EXIST 2022 and EXIST 2023
#Tareas de clasificación

#DIPROMPATS_2023
#Metric: ICMNorm


def evaluate_dipromats_2023(predictions_file, gold_file):
    test = PyEvALLEvaluation()
    metrics=[MetricFactory.ICMNorm.value]
    params= dict()
    report = test.evaluate(predictions_file, gold_file, metrics, **params)
    report.print_report()
    return


def save_to_file(data, filename):
    """Save a list of predictions or gold data in a JSONL file."""
    with open(filename, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

if __name__ == "__main__":        
    predictions = [
        {"test_case": "EXIST2023", "id": "I1", "value": "A"},
        {"test_case": "EXIST2023", "id": "I2", "value": "B"},
        {"test_case": "EXIST2023", "id": "I3", "value": "C"}
    ]
    gold = [
        {"test_case": "EXIST2023", "id": "I1", "value": "B"},
        {"test_case": "EXIST2023", "id": "I2", "value": "B"},
        {"test_case": "EXIST2023", "id": "I3", "value": "C"}
    ]

    # Guardar las listas en archivos temporales
    predictions_file = 'predictions.jsonl'
    gold_file = 'gold.jsonl'
    save_to_file(predictions, predictions_file)
    save_to_file(gold, gold_file)
    # predictions = "test/resources/metric/test/classification/predictions/SYS5.txt"
    # gold = "test/resources/metric/test/classification/gold/GOLD5.txt"
    test = PyEvALLEvaluation()
    metrics=[MetricFactory.FMeasure.value]

    params= dict()
    report = test.evaluate(predictions_file, gold_file, metrics, **params)
    report.print_report()


#SQUAD METRIC para SQUAD-SQAC 2024

from evaluate import load
squad_metric = load("squad")
predictions = [{'prediction_text': '1976', 'id': '56e10a3be3433e1400422b22'}]
references = [{'answers': {'answer_start': [97], 'text': ['1976']}, 'id': '56e10a3be3433e1400422b22'}]
results = squad_metric.compute(predictions=predictions, references=references)
results
{'exact_match': 100.0, 'f1': 100.0}


#HuggingFace Evaluate para DIANN 2023
#BIO
from evaluate import evaluator
from datasets import load_dataset

# task_evaluator = evaluator("token-classification")
task_evaluator = evaluator("question-answering")

data = load_dataset("squad", split="validation[:1000]")
eval_results = task_evaluator.compute(
    data=data,
    metric="squad", #macrof1
)