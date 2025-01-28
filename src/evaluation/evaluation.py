
from pyevall.evaluation import PyEvALLEvaluation
from pyevall.metrics.metricfactory import MetricFactory
from pyevall.utils.utils import PyEvALLUtils
import json
from evaluate import load
import csv


# Función para recolectar métricas y escribir en un CSV
def write_metrics_to_csv(dataset_name, metric_results, output_file="evaluation_metrics.csv"):
    # Crear archivo si no existe y escribir encabezados
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
def evaluate_dipromats_2023(predictions_file, gold_file, dataset_name):
    test = PyEvALLEvaluation()
    metrics = [MetricFactory.ICMNorm.value]
    # params = dict() #TASK 1
    DIPROMATS_TASK3={"True":{
        "1 appeal to commonality":["1 appeal to commonality - ad populum", "1 appeal to commonality - flag waving"],
        "2 discrediting the opponent":["2 discrediting the opponent - absurdity appeal","2 discrediting the opponent - demonization", "2 discrediting the opponent - doubt", "2 discrediting the opponent - fear appeals (destructive)", "2 discrediting the opponent - name calling", "2 discrediting the opponent - propaganda slinging", "2 discrediting the opponent - scapegoating", "2 discrediting the opponent - undiplomatic assertiveness/whataboutism"],
        "3 loaded language":[], 
        "4 appeal to authority":["4 appeal to authority - appeal to false authority", "4 appeal to authority - bandwagoning"]}, 
        "False":[]}
    params = {PyEvALLUtils.PARAM_HIERARCHY: DIPROMATS_TASK3}
    report = test.evaluate(predictions_file, gold_file, metrics, **params).report

    # DIPROMATS_TASK2={"True":["1 appeal to commonality", "2 discrediting the opponent", "3 loaded language", "4 appeal to authority"],"False":[]}

    # params[PyEvALLUtils.PARAM_HIERARCHY]= DIPROMATS_TASK2

    try:
        print("report: ",report)
        icm_norm_metric = report["metrics"]["ICMNorm"]["results"]["average_per_test_case"]
        print(f"ICM-Norm Result: {icm_norm_metric}")
        metric_results = {"ICMNorm": icm_norm_metric}
        write_metrics_to_csv(dataset_name, metric_results)
    except AttributeError as e:
        print(f"Error: Unable to access ICM-Norm metric. {e}")

def evaluate_exist_2022_t1(predictions_file, gold_file, dataset_name):
    test = PyEvALLEvaluation()
    metrics = [MetricFactory.Accuracy.value]
    params = dict()
    report = test.evaluate(predictions_file, gold_file, metrics, **params).report
    
    try:
        print("report: ",report)
        accuracy = report["metrics"]["Accuracy"]["results"]["average_per_test_case"]
        print(f"Accuracy Result: {accuracy}")
        metric_results = {"Accuracy": accuracy}
        write_metrics_to_csv(dataset_name, metric_results)
    except AttributeError as e:
        print(f"Error: Unable to access Accuracy metric. {e}")

def evaluate_exist_2022_t2(predictions_file, gold_file, dataset_name):
    test = PyEvALLEvaluation()
    metrics = [MetricFactory.FMeasure.value]
    params = dict()
    report = test.evaluate(predictions_file, gold_file, metrics, **params).report

    try:
        print("report: ",report)
        fmeasure = report["metrics"]["FMeasure"]["results"]["average_per_test_case"]
        print(f"FMeasure Result: {fmeasure}")
        metric_results = {"FMeasure": fmeasure}
        write_metrics_to_csv(dataset_name, metric_results)
    except AttributeError as e:
        print(f"Error: Unable to access F-Measure metric. {e}")

def evaluate_exist_2023(predictions_file, gold_file, dataset_name):
    test = PyEvALLEvaluation()
    metrics = [MetricFactory.ICMSoftNorm.value]
    params = dict()
    report = test.evaluate(predictions_file, gold_file, metrics, **params).report

    try:
        print("report: ",report)
        icmSoftNorm = report["metrics"]["ICMSoftNorm"]["results"]["average_per_test_case"]
        print(f"ICMSoftNorm Result: {icmSoftNorm}")
        metric_results = {"ICMSoftNorm": icmSoftNorm}
        write_metrics_to_csv(dataset_name, metric_results)
    except AttributeError as e:
        print(f"Error: Unable to access ICMSoftNorm metric. {e}")

def evaluate_sqac_squad_2024(predictions_file, gold_file, dataset_name):
    squad_metric = load("squad")
    results = squad_metric.compute(predictions=predictions_file, references=gold_file)
    print(results)

    metric_results = {"ExactMatch": results["exact_match"], "F1": results["f1"]}
    write_metrics_to_csv(dataset_name, metric_results)

def evaluate_diann_2023(predictions_file, gold_file, dataset_name):
    f1_metric = load("f1")
    results = f1_metric.compute(references=gold_file, predictions=predictions_file, average="macro")
    print(results)

    metric_results = {"MacroF1": results["f1"]}
    write_metrics_to_csv(dataset_name, metric_results)






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
