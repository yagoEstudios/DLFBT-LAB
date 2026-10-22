# Lo que hay que hacer — Proyecto final

Clasificar `Activity` (6 clases) del dataset UCI HAR comparando 4 arquitecturas: **DNN, CNN, LSTM, RNN**.

## Datos
- `train.csv` / `test.csv` en `data/final_project/`.
- Entrada: las 561 columnas (todas menos `subject` y `Activity`). Son estadísticos de ventanas de 2,56 s, no una señal temporal.
- Etiqueta: `Activity` -> `LabelEncoder` (0..5).
- `subject` solo para separar validación por sujeto (`GroupShuffleSplit`), nunca como feature.
- `StandardScaler` ajustado solo con train.
- Test solo para la evaluación final (nunca para elegir modelo/hiperparámetros).

## Estructura obligatoria
```text
src/
├── __init__.py
├── models.py       # DNNClassifier, CNNClassifier, LSTMClassifier, RNNClassifier
├── training.py     # bucle de entrenamiento reutilizable + early stopping
└── evaluation.py   # métricas reutilizables
lab3_2026.ipynb     # experimentos, gráficas y análisis (importa de src.models)
```
- Mismo framework para todo (PyTorch o Keras).
- Los modelos NO se definen en el notebook.

## Comparación
- Mismos train / val / test y mismas métricas para los 4 modelos.
- Hiperparámetros: los elijo yo, con validación. Configuración comparable; si algo difiere (p. ej. LR o grad clipping en RNN/LSTM), justificarlo.

## Registrar por modelo
- Por época: train/val loss y train/val accuracy.
- Tiempo total de entrenamiento y nº de épocas completadas.

## Métricas por modelo
- Accuracy, macro precision, macro recall, macro F1, weighted F1.
- Por clase: precision, recall, F1, support.
- Matriz de confusión.
- Explicar macro vs weighted y por qué importa en este dataset.

## Rendimiento computacional
- Tiempo total, tiempo medio por época, tiempo de inferencia, nº de parámetros entrenables.
- Opcional: muestras/s, memoria CPU/GPU, tamaño del modelo.
- Indicar el hardware.

## Gráficas
- Loss (train vs val) y accuracy (train vs val) por época para cada modelo.
- Discutir: convergencia, overfitting, underfitting, estabilidad, generalización.

## Semillas
- Experimento principal con ≥ 3 semillas -> media ± std de las métricas principales (p. ej. macro F1).

## Experimento adicional
- Variar un único factor (hidden size, capas, dropout, LR, batch size, kernel CNN, nº filtros...) manteniendo el resto fijo.

## Análisis de matrices de confusión
- Clases más fáciles y más confundidas (probablemente SITTING vs STANDING).
- ¿Son las mismas para todas las arquitecturas? ¿Alguna trata clases de forma distinta?
- Qué características de los datos lo explican.

## Pregunta central a responder
¿Cómo difieren DNN, CNN, LSTM y RNN en rendimiento, eficiencia, comportamiento de entrenamiento, robustez y complejidad? Conclusiones basadas en los experimentos.
- Punto a comentar: las 561 features no son una secuencia temporal real, así que la localidad (CNN) y la memoria temporal (RNN/LSTM) se aprovechan solo parcialmente.
