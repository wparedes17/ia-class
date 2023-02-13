import re
import unicodedata
import numpy as np

def hello_function():
    print("Hello World!")

def create_vocabulary(list_of_documents):
    i = 0
    vocabulary = {}
    # Para cada documento en la lista de documentos
    for document_no, document in enumerate(list_of_documents):
        # Separa el documento en palabras
        words = set(document.split(' '))
        # Para cada palabra en las palabras
        for word in words:
            # Si la palabra no está en el vocabulario
            # agregamos el id y el documento en el que aparece
            if word not in vocabulary:
                vocabulary[word] = {'id':i, 'docto':[document_no]}
                i += 1
            # en otro caso solo agregamos el documento en el que aparece
            else:
                vocabulary[word]['docto'].append(document_no)

    # Calculamos el df (document frequency) para cada palabra
    for word in vocabulary:
        vocabulary[word]['df'] = len(vocabulary[word]['docto'])

    return vocabulary

def create_vectorized_document(document, vocabulary, n_docs):
    # Separa el documento en palabras
    words = document.split(' ')
    # Crea un vector de ceros
    vector = np.zeros(len(vocabulary))
    ivector = np.zeros(len(vocabulary))
    # Para cada palabra en las palabras
    for word in words:
        # Si la palabra está en el vocabulario
        if word in vocabulary:
            # Incrementa el valor del vector en la posición de la palabra
            vector[vocabulary[word]['id']] += 1
            ivector[vocabulary[word]['id']] = n_docs/vocabulary[word]['df']
    
    return (np.log(vector+1))*ivector

def create_matrix_documents(list_of_documents, vocabulary):
    # Crea una matriz de ceros
    matrix = np.zeros((len(list_of_documents), len(vocabulary)))
    # Para cada documento en la lista de documentos
    for i in range(len(list_of_documents)):
        # Crea un vector para el documento
        vector = create_vectorized_document(list_of_documents[i], vocabulary, len(list_of_documents))
        # Agrega el vector a la matriz
        matrix[i] = vector
    return matrix

def calculate_cosine_similarity(vector1, vector2):
    # Multiplica los vectores
    dot_product = np.dot(vector1, vector2)
    # Calcula la norma de los vectores
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)
    # Calcula la similitud coseno
    similarity = dot_product / (norm_vector1 * norm_vector2)
    return similarity

def calculate_similarity_for_query(query, matrix_documents, vocabulary):
    # Crea un vector para la consulta
    vector_query = create_vectorized_document(query, vocabulary, 10)
    # Crea una lista para las similitudes
    similarities = {}
    # Para cada documento en la matriz de documentos
    for i in range(len(matrix_documents)):
        # Calcula la similitud coseno
        similarity = calculate_cosine_similarity(vector_query, matrix_documents[i])
        # Agrega la similitud a la lista de similitudes
        similarities[i] = similarity
    return sort_dictionary_by_value(similarities)

def sort_dictionary_by_value(dictionary):
    # Ordena el diccionario
    sorted_dictionary = {k: v for k,v in sorted(dictionary.items(), key=lambda x: x[1], reverse=True)}
    return sorted_dictionary

def remove_accents(input_str):
    if isinstance(input_str, str):
        nkfd_form = unicodedata.normalize('NFKD', input_str.lower())
        return u"".join([c for c in nkfd_form if not unicodedata.combining(c)])
    return str(input_str)

def remove_punctuation(input_str):
    if isinstance(input_str, str):
        return re.sub(r'[^\w\s]', ' ', input_str)
    return str(input_str)
