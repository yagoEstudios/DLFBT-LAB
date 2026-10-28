# Cambios del profesor y split por sujetos

## Últimos commits del profesor (upstream, 20/10/2026)

**Importante:** la entrega del proyecto final se retrasa del viernes 23 al **viernes 30 de octubre de 2026 a las 13:00**.

1. `97924f3`, README: cambio de fecha de entrega. El resto es formato de tablas.
2. `2472d2a`, notebook: solo ejecutó la celda de carga de datos, que ahora muestra su salida (`Training shape: (5551, 561)`, validación `(1801, 561)`, test `(2947, 561)`, 16 sujetos de train, 5 de validación, solapamiento 0). No cambia ninguna regla.
3. `19b939f`, cambio grande del notebook:
   - El split de validación pasa de `train_test_split` estratificado a `GroupShuffleSplit` por `subject`, con un `assert` que comprueba que train y validación no comparten sujetos.
   - Añade las versiones `X_*_rnn`, `X_*_lstm` y `X_*_cnn` con `np.newaxis` (los modelos secuenciales leen los 561 valores como 561 pasos de un canal).
   - Quita la celda de "Visualize Features".
   - Vacía los `src/*.py` para dejarlos como plantilla.

Mi notebook ya va en línea con esto: usa `GroupShuffleSplit` por sujeto, salen las mismas formas `(5551, 561)` / `(1801, 561)` / `(2947, 561)`, y los modelos secuenciales leen la entrada como `(batch, 561, 1)`.

Ojo: el texto del enunciado sigue hablando de entradas `(batch, 128, 9)`, pero los CSV solo traen las 561 features. Mi notebook ya explica esa discrepancia; contarlo también en la memoria.

## Por qué se separa por sujetos

- UCI HAR tiene 30 voluntarios, y cada uno (`subject`) aporta cientos de filas: ventanas de 2,56 s de esa persona andando, sentada, etc.
- Las ventanas se solapan un 50 %, así que dos filas seguidas de la misma persona son casi iguales.
- Cada persona se mueve a su manera (forma de andar, cómo lleva el móvil...).

Si se reparten filas al azar, la misma persona acaba en train y en validación. El modelo aprende "así anda Juan" y en validación vuelve a ver a Juan, con datos casi iguales. La accuracy de validación sale inflada (*data leakage*, fuga de datos). El test son personas nuevas, así que la validación también tiene que tenerlas. Si no, el early stopping y la elección de hiperparámetros se hacen con una estimación demasiado optimista.

**Regla:** cada persona está en un solo grupo (train, val o test).

## Qué sujetos hay en cada conjunto

- El test ya viene fijo en el dataset: `test.csv` tiene 9 personas que no aparecen en `train.csv`, que tiene 21.
- Las 21 personas de `train.csv` se dividen en train y val con `GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)`.

| Conjunto | Personas | Sujetos | Para qué |
|---|---|---|---|
| train | 16 | 5, 6, 7, 8, 11, 14, 16, 17, 19, 21, 22, 23, 26, 28, 29, 30 | entrenar los pesos |
| val | 5 | 1, 3, 15, 25, 27 | early stopping y elegir hiperparámetros |
| test | 9 | 2, 4, 9, 10, 12, 13, 18, 20, 24 | solo la evaluación final |

Si cambio `random_state` o `test_size`, salen otros sujetos en train y val, y no pasa nada mientras no se repitan. El test no se toca.

## La columna `subject`

No entra en el modelo: está en `DROP_COLUMNS` junto con `Activity`, así que no forma parte de las features `X`. Solo se usa como `groups=` en `GroupShuffleSplit`. El modelo tiene que reconocer la actividad, no la persona, y en test las personas son otras.
