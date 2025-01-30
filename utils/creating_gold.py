import os
import json


GOLDEN_FOLDERS = ["diann", "dipromats", "exist", "sqac"]

# Carpeta base donde están las carpetas con los archivos
script_dir = os.path.dirname(os.path.realpath(__file__))
base_folder = os.path.join(script_dir, "./../data")
output_folder = os.path.join(script_dir, "./../data_gold")
print(f"Base folder: {base_folder}")

# Crear la carpeta de salida si no existe
os.makedirs(output_folder, exist_ok=True)

# Recorrer las carpetas y archivos dentro de la carpeta base
for root, _, files in os.walk(base_folder):
    for file_name in files:
        # Ignorar archivos que contengan "test" en el nombre
        if "test" in file_name:
            continue

        # Construir la ruta completa del archivo
        file_path = os.path.join(root, file_name)

        # Procesar solo archivos JSON
        if file_name.endswith(".json"):
            # Leer el archivo JSON
            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)

            # Filtrar los campos deseados
            data_filtered = [
                {"test_case": item["test_case"], "id": item["id"], "value": item["value"]}
                for item in data
            ]

            # Crear el nuevo nombre de archivo y su ruta
            relative_path = os.path.relpath(root, base_folder)
            new_folder_path = os.path.join(output_folder, relative_path)
            os.makedirs(new_folder_path, exist_ok=True)

            new_file_name = file_name.replace(".json", "_gold.json")
            new_file_path = os.path.join(new_folder_path, new_file_name)

            # Guardar los datos filtrados en el nuevo archivo
            with open(new_file_path, "w", encoding="utf-8") as new_file:
                json.dump(data_filtered, new_file, indent=4, ensure_ascii=False)

print(f"Archivos procesados y guardados en la carpeta: {output_folder}")

script_dir = os.path.dirname(os.path.realpath(__file__))
base_folder = "/workspace/data"

def create_gold(partition, base_folder):
    
    output_folder = f"{base_folder}/data_gold{partition}"

    # Crear la carpeta de salida si no existe
    os.makedirs(output_folder, exist_ok=True)
    print(f"Output folder: {output_folder}")

    # Recorrer las carpetas y archivos dentro de la carpeta base
    folder_files = os.listdir(base_folder)
    
    for root, _, files in os.walk(base_folder):
        for file_name in files:
            # Ignorar archivos que contengan "test" en el nombre
            if partition in file_name:
                continue

            # Construir la ruta completa del archivo
            file_path = os.path.join(root, file_name)

            # Procesar solo archivos JSON
            if file_name.endswith(".json"):
                # Leer el archivo JSON
                with open(file_path, "r", encoding="utf-8") as file:
                    data = json.load(file)

                # Filtrar los campos deseados
                data_filtered = [
                    {"test_case": item["test_case"], "id": item["id"], "value": item["value"]}
                    for item in data
                ]

                # Crear el nuevo nombre de archivo y su ruta
                relative_path = os.path.relpath(root, base_folder)
                new_folder_path = os.path.join(output_folder, relative_path)
                os.makedirs(new_folder_path, exist_ok=True)

                new_file_name = file_name.replace(".json", "_gold.json")
                new_file_path = os.path.join(new_folder_path, new_file_name)

                # Guardar los datos filtrados en el nuevo archivo
                with open(new_file_path, "w", encoding="utf-8") as new_file:
                    json.dump(data_filtered, new_file, indent=4, ensure_ascii=False)

    print(f"Archivos procesados y guardados en la carpeta: {output_folder}")