import json
import os

def calculate_value(file_path, output_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    for entry in data:
        reported_count = entry['value'].count("REPORTED")
        direct_count = entry['value'].count("DIRECT")
        no_count = entry['value'].count("-")
        judgemental_count = entry['value'].count("JUDGEMENTAL")
        total = reported_count + direct_count + no_count + judgemental_count

        entry['value_probability'] = {
            "REPORTED": round((reported_count / total), 4) if total > 0 else 0,
            "DIRECT": round((direct_count / total), 4) if total > 0 else 0,
            "NO": round((no_count / total), 4) if total > 0 else 0,
            "JUDGEMENTAL": round((judgemental_count / total), 4) if total > 0 else 0
        }

    with open(output_path, 'w', encoding='utf-8') as output_file:
        json.dump(data, output_file, ensure_ascii=False, indent=4)

# Obtener el directorio donde se encuentra el script
script_dir = os.path.dirname(os.path.realpath(__file__))

# Directorios de entrada y salida basados en el directorio del script
input_dir = os.path.join(script_dir, '../data/exist_2023')
output_dir = os.path.join(input_dir, 'processed')
os.makedirs(output_dir, exist_ok=True)

# Archivos de entrada
input_files = {
    "train_t2_es.json": os.path.join(input_dir, "train_t2_es.json"),
    "train_t2_en.json": os.path.join(input_dir, "train_t2_en.json"),
    "val_t2_es.json": os.path.join(input_dir, "val_t2_es.json"),
    "val_t2_en.json": os.path.join(input_dir, "val_t2_en.json")
}

# Procesar cada archivo
for file_name, file_path in input_files.items():
    output_path = os.path.join(output_dir, file_name)
    calculate_value(file_path, output_path)

print(f"Procesamiento completado. Archivos guardados en {output_dir}")


