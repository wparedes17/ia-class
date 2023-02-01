import numpy as np

def hello_function():
    print("Hello World!")

def create_vocabulary(list_of_documents):
    i = 0
    vocabulary = {}
    # Para cada documento en la lista de documentos
    for document in list_of_documents:
        # Separa el documento en palabras
        words = document.split(' ')
        # Para cada palabra en las palabras
        for word in words:
            # Si la palabra no está en el vocabulario
            if word not in vocabulary:
                vocabulary[word] = i
                i += 1
    return vocabulary

def create_vectorized_document(document, vocabulary):
    # Separa el documento en palabras
    words = document.split(' ')
    # Crea un vector de ceros
    vector = np.zeros(len(vocabulary))
    # Para cada palabra en las palabras
    for word in words:
        # Si la palabra está en el vocabulario
        if word in vocabulary:
            # Incrementa el valor del vector en la posición de la palabra
            vector[vocabulary[word]] += 1
    return vector

def create_matrix_documents(list_of_documents, vocabulary):
    # Crea una matriz de ceros
    matrix = np.zeros((len(list_of_documents), len(vocabulary)))
    # Para cada documento en la lista de documentos
    for i in range(len(list_of_documents)):
        # Crea un vector para el documento
        vector = create_vectorized_document(list_of_documents[i], vocabulary)
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
    vector_query = create_vectorized_document(query, vocabulary)
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


# DF-IDF = (1 + log(tf)) * log(N/df)