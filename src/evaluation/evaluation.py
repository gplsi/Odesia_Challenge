
from pyevall.evaluation import PyEvALLEvaluation
from pyevall.metrics.metricfactory import MetricFactory
from pyevall.utils.utils import PyEvALLUtils
import json
from evaluate import load
import csv

from src.data.config import CLASSES_DIANN_2023_T3


# Función para recolectar métricas y escribir en un CSV
def write_metrics_to_csv(dataset_name, metric_results, partition):
    # Crear archivo si no existe y escribir encabezados
    output_file=f"./data/{partition}_metrics.csv"
    try:
        with open(output_file, mode='x', newline='') as file:
            writer = csv.writer(file)
            headers = ["Dataset"] + list(metric_results.keys())
            writer.writerow(headers)
    except FileExistsError:
        pass

    # Escribir los resultados
    with open(output_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        row = [dataset_name] + list(metric_results.values())
        writer.writerow(row)

# Funciones de evaluación con recolectores de métricas
def evaluate_dipromats_2023(predictions_file, gold_file, dataset_name, params, partition):
    test = PyEvALLEvaluation()
    metrics = [MetricFactory.ICMNorm.value]
    report = test.evaluate(predictions_file, gold_file, metrics, **params).report
    try:
        print("report: ",report)
        icm_norm_metric = report["metrics"]["ICMNorm"]["results"]["average_per_test_case"]
        print(f"ICM-Norm Result: {icm_norm_metric}")
        metric_results = {"ICMNorm": icm_norm_metric}
        write_metrics_to_csv(dataset_name, metric_results, partition)
    except AttributeError as e:
        print(f"Error: Unable to access ICM-Norm metric. {e}")

def evaluate_exist_2022_t1(predictions_file, gold_file, dataset_name, partition):
    test = PyEvALLEvaluation()
    metrics = [MetricFactory.Accuracy.value]
    params = dict()
    report = test.evaluate(predictions_file, gold_file, metrics, **params).report
    
    try:
        print("report: ",report)
        accuracy = report["metrics"]["Accuracy"]["results"]["average_per_test_case"]
        print(f"Accuracy Result: {accuracy}")
        metric_results = {"Accuracy": accuracy}
        write_metrics_to_csv(dataset_name, metric_results, partition)
    except AttributeError as e:
        print(f"Error: Unable to access Accuracy metric. {e}")

def evaluate_exist_2022_t2(predictions_file, gold_file, dataset_name, partition):
    test = PyEvALLEvaluation()
    metrics = [MetricFactory.FMeasure.value]
    params = dict()
    report = test.evaluate(predictions_file, gold_file, metrics, **params).report

    try:
        print("report: ",report)
        fmeasure = report["metrics"]["FMeasure"]["results"]["average_per_test_case"]
        print(f"FMeasure Result: {fmeasure}")
        metric_results = {"FMeasure": fmeasure}
        write_metrics_to_csv(dataset_name, metric_results, partition)
    except AttributeError as e:
        print(f"Error: Unable to access F-Measure metric. {e}")

def evaluate_exist_2023(predictions_file, gold_file, dataset_name, partition):
    test = PyEvALLEvaluation()
    metrics = [MetricFactory.ICMSoftNorm.value]
    params = dict()
    report = test.evaluate(predictions_file, gold_file, metrics, **params).report

    try:
        print("report: ",report)
        icmSoftNorm = report["metrics"]["ICMSoftNorm"]["results"]["average_per_test_case"]
        print(f"ICMSoftNorm Result: {icmSoftNorm}")
        metric_results = {"ICMSoftNorm": icmSoftNorm}
        write_metrics_to_csv(dataset_name, metric_results, partition)
    except AttributeError as e:
        print(f"Error: Unable to access ICMSoftNorm metric. {e}")

def evaluate_sqac_squad_2024(predictions_file, gold_file, dataset_name, partition):
    squad_metric = load("squad")
    with open(predictions_file) as f:
        predictions = json.load(f)
    with open(gold_file) as f:
        golds = json.load(f)
    predictions = [{'prediction_text': prediction["value"], 'id': prediction["id"]} for prediction in predictions]
    golds = [{'answers': {'text': [gold["value"]], 'answer_start': [gold["context"].find(gold["value"])]}, 'id': gold["id"]} for gold in golds]
    print(len(golds))
    print(len(predictions))
    results = squad_metric.compute(predictions=predictions, references=golds)
    print(results)

    metric_results = {"ExactMatch": results["exact_match"], "F1": results["f1"]}
    write_metrics_to_csv(dataset_name, metric_results, partition)

def evaluate_diann_2023(predictions_file, gold_file, dataset_name, partition):
    f1_metric = load("f1")
    
    with open(predictions_file) as f:
        predictions = json.load(f)
        
    with open(gold_file) as f:
        golds = json.load(f)
    
    # Create dicts keyed by ID
    gold_dict = {item["id"]: item["value"] for item in golds}
    
    # turn list CLASSES_DIANN_2023_T3 to dictionary with value being index
    classes_dict = {CLASSES_DIANN_2023_T3[i]: i for i in range(len(CLASSES_DIANN_2023_T3))}
    
    gold_cls = []
    pred_cls = []
    for i in range(len(predictions)):
        gold_tokens = gold_dict[predictions[i]["id"]]
        pred_tokens = predictions[i]["value"]
        
        gold_cls += [classes_dict[item] for item in gold_tokens]
        pred_cls += [classes_dict[item] for item in pred_tokens]
        
    results = f1_metric.compute(references=gold_cls, predictions=pred_cls, average="macro")
    metric_results = {"MacroF1": results["f1"]}
    write_metrics_to_csv(dataset_name, metric_results, partition)


#EJEMPLOS DE FORMATO
# def save_to_file(data, filename):
#     """Save a list of predictions or gold data in a JSONL file."""
#     with open(filename, 'w') as f:
#         for item in data:
#             f.write(json.dumps(item) + '\n')

# if __name__ == "__main__":
#     # DIPROMATS 2023, EXIST 2022 and EXIST 2023
#     # EJEMPLO DEL FORMATO EN EL QUE DEBEN ESTAR LOS JSONS        
#     predictions = [
#         {"test_case": "EXIST2023", "id": "I1", "value": "A"},
#         {"test_case": "EXIST2023", "id": "I2", "value": "B"},
#         {"test_case": "EXIST2023", "id": "I3", "value": "C"}
#     ]
#     gold = [
#         {"test_case": "EXIST2023", "id": "I1", "value": "B"},
#         {"test_case": "EXIST2023", "id": "I2", "value": "B"},
#         {"test_case": "EXIST2023", "id": "I3", "value": "C"}
#     ]

#     # Guardar las listas en archivos temporales
#     predictions_file = 'predictions.jsonl'
#     gold_file = 'gold.jsonl'
#     save_to_file(predictions, predictions_file)
#     save_to_file(gold, gold_file)
#     # predictions = "test/resources/metric/test/classification/predictions/SYS5.txt"
#     # gold = "test/resources/metric/test/classification/gold/GOLD5.txt"
#     test = PyEvALLEvaluation()
#     metrics=[MetricFactory.FMeasure.value]

#     params= dict()
#     report = test.evaluate(predictions_file, gold_file, metrics, **params)
#     report.print_report()


#     #SQUAD METRIC para SQUAD-SQAC 2024

#     from evaluate import load
#     squad_metric = load("squad")
#     predictions = [{'prediction_text': '1976', 'id': '56e10a3be3433e1400422b22'}]
#     references = [{'answers': {'answer_start': [97], 'text': ['1976']}, 'id': '56e10a3be3433e1400422b22'}]
#     results = squad_metric.compute(predictions=predictions, references=references)
#     print(results)


#     #HuggingFace Evaluate para DIANN 2023
#     #BIO
#     from evaluate import load
#     f1_metric = load("f1")
#     predictions = [0, 0, 1, 1, 0]
#     references = [0, 1, 0, 1, 0]
#     results = f1_metric.compute(references=references, predictions=predictions, average="macro")
#     print(results)
