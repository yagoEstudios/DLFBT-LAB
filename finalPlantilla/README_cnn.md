# CNNClassifier: decisiones de arquitectura

Arquitectura implementada en `Final_project/src/models.py`. No se busca la máxima accuracy, sino el esquema más estándar y sencillo que funcione de forma razonable.

## Estructura

```
(B, 561) -> adapt_input(x, "cnn") -> (B, 1, 561)
-> [Conv1d -> BatchNorm -> ReLU -> MaxPool(2)] x 3   (32, 64, 128 filtros)
-> AdaptiveAvgPool1d(1) -> (B, 128)
-> Dropout -> Linear -> (B, 6) logits
```

## Por qué

- **Bloques Conv → BatchNorm → ReLU → MaxPool**: es el patrón básico de cualquier CNN, fácil de justificar y paralelo al DNN.
- **Filtros que se duplican (32 → 64 → 128) mientras el MaxPool reduce la longitud a la mitad**: es la regla habitual. Cada capa ve un contexto más amplio con más canales, y el coste se mantiene controlado.
- **`kernel_size=5`**: un valor pequeño y común. Como las features vienen agrupadas por ejes y por tipo, un kernel corto abarca features relacionadas (p. ej. X, Y, Z de la misma medida).
- **Pooling global en lugar de aplanar**: ~53 000 parámetros en vez de cientos de miles, menos sobreajuste y un entrenamiento rápido.

## Expectativa

En este dataset las CNN 1D suelen quedarse cerca del DNN, porque el orden de las columnas no es aleatorio: las features relacionadas están juntas. Probablemente quede algo por debajo del DNN, lo cual ya es un resultado interesante para el informe.

## Hiperparámetros configurables

`channels`, `kernel_size`, `dropout` y `batch_norm` son argumentos del constructor, así que sirven para el experimento adicional (tamaño del kernel o número de filtros).
