import json
import os

def calculate_value_binary(file_path, output_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    for entry in data:
        yes_count = entry['value'].count("YES")
        no_count = entry['value'].count("NO")
        total = yes_count + no_count

        entry['value_binary'] = {
            "YES": round((yes_count / total) * 100, 4) if total > 0 else 0,
            "NO": round((no_count / total) * 100, 4) if total > 0 else 0
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
    "train_t1_es.json": os.path.join(input_dir, "train_t1_es.json"),
    "train_t1_en.json": os.path.join(input_dir, "train_t1_en.json"),
    "val_t1_es.json": os.path.join(input_dir, "val_t1_es.json"),
    "val_t1_en.json": os.path.join(input_dir, "val_t1_en.json")
}

# Procesar cada archivo
for file_name, file_path in input_files.items():
    output_path = os.path.join(output_dir, file_name)
    calculate_value_binary(file_path, output_path)

print(f"Procesamiento completado. Archivos guardados en {output_dir}")


