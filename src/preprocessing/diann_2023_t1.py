def extract_bio_tokens(data):
    # Asegúrate de que las claves 'tokens' y 'labels' están en el diccionario
    if 'tokens' not in data or 'value' not in data:
        raise ValueError("El diccionario debe contener las claves 'tokens' y 'value'")

    tokens = data['tokens']
    labels = data['value']

    # Validar que ambos arreglos tienen la misma longitud
    if len(tokens) != len(labels):
        raise ValueError("'tokens' y 'value' deben tener la misma longitud")

    # Extraer los tokens con etiquetas B o I
    bio_tokens = [token for token, label in zip(tokens, labels) if label in {'B-DIS', 'I-DIS'}]

    # Convertirlos a una cadena separada por comas
    return str(bio_tokens)