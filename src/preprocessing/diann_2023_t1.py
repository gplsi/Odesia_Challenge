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

def extract_bio_spans(data):
    # Asegúrate de que las claves 'tokens' y 'labels' están en el diccionario
    if 'tokens' not in data or 'value' not in data:
        raise ValueError("El diccionario debe contener las claves 'tokens' y 'value'")

    tokens = data['tokens']
    labels = data['value']

    # Validar que ambos arreglos tienen la misma longitud
    if len(tokens) != len(labels):
        raise ValueError("'tokens' y 'labels' deben tener la misma longitud")

    # Extraer los tokens con etiquetas B o I
    spans = []
    current_span = []
    for i in range(0, len(tokens)):
        token = tokens[i]
        label = labels[i]
        if label == 'B-DIS':
            if current_span:  # Si había un span previo, guardarlo antes de iniciar uno nuevo
                spans.append(' '.join(current_span))
                current_span = []
            current_span.append(token)
        elif label == 'I-DIS':
            if current_span:  # Solo añadir si hay un span en curso
                current_span.append(token)
        else:  # 'O' u otra etiqueta
            if current_span:  # Guardar span antes de resetear
                spans.append(' '.join(current_span))
                current_span = []
    
    if current_span:
        spans.append(' '.join(current_span))

    # Convertirlos a una cadena separada por comas
    return str(spans)