import os
import json
import datetime
def post_processing_error_handler(e, text, ids, task_name, language, partition):
    print(f"Error al procesar el texto: {e}")
    error_entry = {
        "id": str(ids),
        "text": text,
        "error": str(e)
    }
    
    # Define the error file name dynamically based on task_name
    current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    error_folder = './error_files'
    if not os.path.exists(error_folder):
        os.makedirs(error_folder)
    error_file = f"{error_folder}/error_file_{task_name}_{language}_{partition}_{current_datetime}.json"
    
    # Check if the file exists and has valid content
    if os.path.exists(error_file) and os.path.getsize(error_file) > 0:
        with open(error_file, "r") as f:
            try:
                errors = json.load(f)  # Load existing data
            except json.JSONDecodeError:
                errors = []  # If the file is invalid, start fresh
    else:
        errors = []

    # Append the new error entry
    errors.append(error_entry)

    # Write the updated errors list back to the file
    with open(error_file, "w") as f:
        json.dump(errors, f, indent=4)
                
                
def evaluation_error_handler(e, task_name, partition, language):
                print(f"Error al evaluar el texto: {e}")
                error_entry = {
                    "task": task_name,
                    "partition": partition,
                    "language": language,
                    "error": str(e)
                }
                
                # Check if the file exists and has content
                if os.path.exists("error_file.json") and os.path.getsize("error_file.json") > 0:
                    with open("error_file.json", "r") as f:
                        try:
                            errors = json.load(f)  # Load existing data
                        except json.JSONDecodeError:
                            errors = []  # If the file is invalid, start