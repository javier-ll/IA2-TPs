import numpy as np

def dividir_entrenamiento_prueba(x, t, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)

    num_examples = x.shape[0]
    num_test_examples = int(num_examples * test_size)
    indices = np.random.permutation(num_examples)

    x_prueba = x[indices[:num_test_examples]]
    t_prueba = t[indices[:num_test_examples]]
    x_entrenamiento = x[indices[num_test_examples:]]
    t_entrenamiento = t[indices[num_test_examples:]]

    return x_entrenamiento, x_prueba, t_entrenamiento, t_prueba

def dividir_entrenamiento_validacion_prueba(x, t, test_size=0.2, validacion_size=0.2, random_state=42):
    if random_state is not None:
        np.random.seed(random_state)

    num_examples = x.shape[0]
    num_test_examples = int(num_examples * test_size)
    num_validation_examples = int(num_examples * validacion_size)

    indices = np.random.permutation(num_examples)

    x_prueba = x[indices[:num_test_examples]]
    t_prueba = t[indices[:num_test_examples]]

    remaining_indices = indices[num_test_examples:]
    x_validacion = x[remaining_indices[:num_validation_examples]]
    t_validacion = t[remaining_indices[:num_validation_examples]]
    x_entrenamiento = x[remaining_indices[num_validation_examples:]]
    t_entrenamiento = t[remaining_indices[num_validation_examples:]]

    return x_entrenamiento, x_validacion, x_prueba, t_entrenamiento, t_validacion, t_prueba