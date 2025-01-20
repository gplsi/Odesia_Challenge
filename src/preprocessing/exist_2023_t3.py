import json
import os

def calculate_value(file_path, output_path):
    import json

    # Load the data
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Iterate through each entry in the data
    for entry in data:
        # Flatten the nested lists in "value"
        flattened_values = [item for sublist in entry['value'] for item in sublist]

        # Count occurrences of each category
        misogyny_count = flattened_values.count("MISOGYNY-NON-SEXUAL-VIOLENCE")
        ideological_count = flattened_values.count("IDEOLOGICAL-INEQUALITY")
        no_count = flattened_values.count("-")
        stereotyping_count = flattened_values.count("STEREOTYPING-DOMINANCE")
        sexual_violence_count = flattened_values.count("SEXUAL-VIOLENCE")
        objectification_count = flattened_values.count("OBJECTIFICATION")

        # Calculate the total occurrences
        total = 6

        # Calculate probabilities for each category
        entry['value_probability'] = {
            "MISOGYNY-NON-SEXUAL-VIOLENCE": round((misogyny_count / total), 4) if total > 0 else 0,
            "IDEOLOGICAL-INEQUALITY": round((ideological_count / total), 4) if total > 0 else 0,
            "NO": round((no_count / total), 4) if total > 0 else 0,
            "STEREOTYPING-DOMINANCE": round((stereotyping_count / total), 4) if total > 0 else 0,
            "SEXUAL-VIOLENCE": round((sexual_violence_count / total), 4) if total > 0 else 0,
            "OBJECTIFICATION": round((objectification_count / total), 4) if total > 0 else 0
        }

    # Save the updated data to the output file
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
    "train_t3_es.json": os.path.join(input_dir, "train_t3_es.json"),
    "train_t3_en.json": os.path.join(input_dir, "train_t3_en.json"),
    "val_t3_es.json": os.path.join(input_dir, "val_t3_es.json"),
    "val_t3_en.json": os.path.join(input_dir, "val_t3_en.json")
}

# Procesar cada archivo
for file_name, file_path in input_files.items():
    output_path = os.path.join(output_dir, file_name)
    calculate_value(file_path, output_path)

print(f"Procesamiento completado. Archivos guardados en {output_dir}")


