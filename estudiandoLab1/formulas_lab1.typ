#import "neuronas.typ": neurona, grafo, capa, red
#set page(paper: "a4", margin: 2cm)
#set text(lang: "es", size: 10.5pt)
#set heading(numbering: "1.")
#set math.equation(numbering: none)
#show raw.where(block: true): it => block(fill: luma(245), inset: 6pt, radius: 3pt, width: 100%, it)

// ---------- Diagramas de matrices ----------
#let U = 0.85em
#let cX = rgb("#cfe2ff")
#let cW = rgb("#ffe0b3")
#let cB = rgb("#d4f4d4")
#let cY = rgb("#e7d4f7")
#let cD = rgb("#ffd0d0")

// M(etiqueta, dims, ancho, alto, color)
#let M(l, d, w, h, fill) = (l: l, d: d, w: w * U, h: h * U, fill: fill)

#let diag(..items) = {
  let its = items.pos()
  align(center, block(inset: (y: 6pt), grid(
    columns: its.len(), column-gutter: 9pt, row-gutter: 5pt, align: center + horizon,
    ..its.map(it => if type(it) == dictionary { context {
      let lbl = text(9pt, it.l)
      let s = measure(lbl)
      rect(
        width: calc.max(it.w.to-absolute(), s.width + 6pt),
        height: calc.max(it.h.to-absolute(), s.height + 6pt),
        fill: it.fill, stroke: 0.7pt, inset: 1pt, align(center + horizon, lbl),
      )
    }} else { text(13pt, it) }),
    ..its.map(it => if type(it) == dictionary { text(8pt, fill: luma(90), it.d) } else { [] }),
  )))
}

// Fórmula de partida, copiada tal cual de los apuntes
#let apuntes(src, body) = block(
  width: 100%, breakable: false, fill: rgb("#fff8e1"), stroke: (left: 3pt + rgb("#f0b429")), inset: 8pt,
)[#text(8pt, weight: "bold", fill: rgb("#8a6d00"))[APUNTES · #src] #body]

// Pasos intermedios
#let pasos(body) = block(width: 100%, inset: (left: 8pt, y: 2pt), stroke: (left: 1pt + luma(190)), body)

// Fórmula final que implementa el TO-DO
#let formula(body) = align(center, block(
  fill: rgb("#f3f7ff"), stroke: 0.6pt + rgb("#9bb5e0"), inset: 10pt, radius: 4pt,
)[#text(8pt, weight: "bold", fill: rgb("#3d5f99"))[LAB] #body])

#align(center, text(16pt, weight: "bold")[DLFBT — Lab 1: de los apuntes a cada TO-DO])
#align(center, text(9pt, fill: luma(80))[
  Cada apartado parte de la fórmula de los apuntes (#box(fill: rgb("#fff8e1"), inset: 2pt)[amarillo]), la transforma paso a paso y llega a la del lab (#box(fill: rgb("#f3f7ff"), inset: 2pt)[azul]). \
  Los apuntes escriben la traspuesta como $bold(w)^t$; aquí uso $bold(w)^top$ para no confundirla con el target $t$. \
  Índices: $i = 1, dots, N$ patrones · $k = 1, dots, d$ (o $n_(l-1)$) entradas · $j = 1, dots, n_l$ neuronas de la capa $l$.   $N$ patrones, $d$ dimensión, $eta$ learning rate. Ej. 1, 2, 5: datos por *filas* $(N, d)$. Ej. 6, 7: datos por *columnas* $(d, N)$.
])

#block(breakable: false, width: 100%)[
= Regresión lineal (NumPy) — `LinearRegressionModel`

== `predict`
```python
y = np.dot(x, self.w) + self.b
```
#apuntes[Linear Regression · arbitrary dimension][$ y_i = bold(w)^t bold(x)_i + b $]
#pasos[
  Apilo los $N$ patrones como *filas* de una matriz. La fila $i$ de $bold(X) bold(w)$ es $bold(x)_i^top bold(w) = bold(w)^top bold(x)_i$, así que las $N$ predicciones salen de golpe:
  $ bold(X) = mat(bold(x)_1^top; dots.v; bold(x)_N^top) quad ==> quad
    mat(y_1; dots.v; y_N) = mat(bold(x)_1^top bold(w) + b; dots.v; bold(x)_N^top bold(w) + b) = bold(X) bold(w) + b $
]
#formula[$ bold(y) = bold(X) bold(w) + b $]
#neurona(act: $"id"$, nota: [regresión lineal = una neurona con activación identidad])
#diag(M($bold(X)$, $N times d$, 2, 6, cX), $dot$, M($bold(w)$, $d times 1$, 1, 2, cW), $+$, M($b$, $1 times 1$, 1, 1, cB), $=$, M($bold(y)$, $N times 1$, 1, 6, cY))
El bias $b$ $(1 times 1)$ se suma a las $N$ filas por _broadcasting_.
]

#block(breakable: false, width: 100%)[
== `compute_gradients`
```python
delta = (y - t) / x.shape[0]
db = np.sum(delta, axis=0, keepdims=True)
dw = np.dot(x.T, delta)
```
#apuntes[Linear Regression · arbitrary dimension][$
  E = sum_(i=1)^N 1/2 (y_i - t_i)^2, quad
  nabla_bold(w) E = sum_(i=1)^N (y_i - t_i) bold(x)_i, quad
  (partial E) / (partial b) = sum_(i=1)^N (y_i - t_i)
$]
#pasos[
  *1. Suma → media.* El lab usa la media (`0.5 * np.mean(...)`), no la suma: $L = E \/ N$, así que todo gradiente se divide entre $N$:
  $ L = 1/(2N) sum_(i=1)^N (y_i - t_i)^2 ==> nabla_bold(w) L = 1/N sum_(i=1)^N (y_i - t_i) bold(x)_i, quad (partial L) / (partial b) = 1/N sum_(i=1)^N (y_i - t_i) $
  *2. Agrupo el error.* Llamo $delta_i = (y_i - t_i) \/ N$ (es $partial L \/ partial y_i$):
  $ nabla_bold(w) L = sum_(i=1)^N delta_i bold(x)_i, quad (partial L) / (partial b) = sum_(i=1)^N delta_i $
  *3. Suma → producto matricial.* Las columnas de $bold(X)^top$ son los $bold(x)_i$, y $bold(X)^top bold(delta)$ es la combinación de esas columnas con pesos $delta_i$:
  $ sum_(i=1)^N delta_i bold(x)_i = mat(bold(x)_1, dots, bold(x)_N) mat(delta_1; dots.v; delta_N) = bold(X)^top bold(delta) $
]
#formula[$
  bold(delta) = (bold(y) - bold(t)) / N, quad
  (partial L) / (partial b) = sum_(i=1)^N delta_i, quad
  (partial L) / (partial bold(w)) = bold(X)^top bold(delta)
$]
#neurona(act: $"id"$, mode: "back")
#grid(columns: (1fr, auto), align: horizon,
  diag(M($bold(X)^top$, $d times N$, 6, 2, cX), $dot$, M($bold(delta)$, $N times 1$, 1, 6, cD), $=$, M($partial L \/ partial bold(w)$, $d times 1$, 3.2, 2, cW)),
  diag(M($bold(delta)$, $N times 1$, 1, 6, cD), $arrow.r^(sum_(i=1)^N)$, M($partial L \/ partial b$, $1 times 1$, 3.2, 1, cB)))
]

#block(breakable: false, width: 100%)[
== `gradient_step`
```python
self.b = self.b - eta * db
self.w = self.w - eta * dw
```
#apuntes[Linear Regression · arbitrary dimension][$
  bold(w)_(t+1) = bold(w)_t - eta nabla_bold(w) E, quad
  b_(t+1) = b_t - eta (partial E) / (partial b)
$]
#pasos[Idéntico, cambiando $E$ por $L$ (los gradientes del apartado anterior).]
#formula[$ b arrow.l b - eta (partial L) / (partial b), quad bold(w) arrow.l bold(w) - eta (partial L) / (partial bold(w)) $]
#neurona(act: $"id"$, mode: "upd")
]

#block(breakable: false, width: 100%)[
= Regresión logística — `LogisticRegressionModel`

== `predict` (sobrescrito)
```python
def predict(self, x):
    z = np.dot(x, self.w) + self.b
    return LogisticRegressionModel.sigmoid(z)
```
#apuntes[Logistic Regression · arbitrary dimension][$
  y_i = sigma(bold(w)^t bold(x)_i + b), quad sigma(z) = e^z / (e^z + 1) = 1 / (1 + e^(-z))
$]
#pasos[
  Es la regresión lineal seguida de $sigma$ elemento a elemento. Con el mismo apilado por filas que en el ej. 1:
  $ bold(z) = bold(X) bold(w) + b ==> bold(y) = sigma(bold(z)) $
]
#formula[$ bold(y) = sigma(bold(X) bold(w) + b) $]
#neurona(act: $sigma$, nota: [regresión logística = una neurona con activación sigmoide])
#diag(M($bold(X)$, $N times d$, 2, 6, cX), $dot$, M($bold(w)$, $d times 1$, 1, 2, cW), $+$, M($b$, $1 times 1$, 1, 1, cB), $arrow.r^sigma$, M($bold(y)$, $N times 1$, 1, 6, cY))
]

#block(breakable: false, width: 100%)[
== Gradientes (heredados, no hace falta sobrescribirlos)
```python
# Heredado de LinearRegressionModel (no es un TO-DO de esta clase):
delta = (y - t) / x.shape[0]
db = np.sum(delta, axis=0, keepdims=True)
dw = np.dot(x.T, delta)
```
#apuntes[Logistic Regression · cross-entropy / arbitrary dimension][$
  C = -sum_(i=1)^N [t_i log y_i + (1 - t_i) log(1 - y_i)], quad
  nabla_bold(w) C = sum_(i=1)^N (y_i - t_i) bold(x)_i, quad
  (partial C) / (partial b) = sum_(i=1)^N (y_i - t_i), quad
  sigma'(z) = sigma(z)(1 - sigma(z))
$]
#pasos[
  Por qué sale lo mismo que en lineal: derivando respecto a la preactivación $z_i$ con la regla de la cadena,
  $ (partial C) / (partial y_i) = -t_i / y_i + (1 - t_i) / (1 - y_i) = (y_i - t_i) / (y_i (1 - y_i)), quad
    (partial y_i) / (partial z_i) = sigma'(z_i) = y_i (1 - y_i) $
  $ ==> (partial C) / (partial z_i) = y_i - t_i $
  El lab usa la media ($L = C \/ N$, `-np.mean(...)`), así que $partial L \/ partial z_i = (y_i - t_i) \/ N = delta_i$: exactamente la $bold(delta)$ del ej. 1.
]
#formula[$ bold(delta) = (bold(y) - bold(t)) / N, quad (partial L) / (partial b) = sum_(i=1)^N delta_i, quad (partial L) / (partial bold(w)) = bold(X)^top bold(delta) $]
#neurona(act: $sigma$, mode: "back")
`compute_gradients` y `gradient_step` se heredan de `LinearRegressionModel` sin cambios; sólo cambian `predict` y `get_loss`.
]

#block(breakable: false, width: 100%)[
= TensorFlow básico — `BasicTF`

== `differentiate`
```python
with tf.GradientTape() as tape:
    y = f(x)
dy_dx = tape.gradient(y, x).numpy()
```
#apuntes[Linear Regression · gradient descent][$ (partial E) / (partial w) $ — los apuntes calculan las derivadas a mano; aquí las calcula TensorFlow.]
#pasos[
  `f` se aplica elemento a elemento: $y_m = f(x_m)$, con $m = 1, dots, M$ recorriendo los $M$ elementos de `x`. `tape.gradient(y, x)` con `y` no escalar deriva la *suma* $sum_(m=1)^M y_m$; como $y_m$ sólo depende de $x_m$:
  $ (partial) / (partial x_m) sum_(m'=1)^M f(x_(m')) = f'(x_m) $
]
#formula[$ (d y_m) / (d x_m) = f'(x_m), quad m = 1, dots, M $]
#grafo()
`.numpy()` convierte el tensor resultado en array.
]

#block(breakable: false, width: 100%)[
== `gradient_descent`
```python
with tf.GradientTape() as tape:
    y = f(x)
dy_dx = tape.gradient(y, x)
x.assign(x - eta * dy_dx)
```
#apuntes[Linear Regression · gradient descent][$ w_1 = w_0 - eta (partial E) / (partial w) $]
#pasos[La variable a optimizar es $x$ y la función a minimizar es $f$: $w -> x$, $E -> f$; el subíndice es la iteración, que llamo $tau$.]
#formula[$ x_(tau+1) = x_tau - eta f'(x_tau) quad (tau = "iteración") $]
#grafo(mode: "gd")
]

#block(breakable: false, width: 100%)[
= Regresión lineal (TensorFlow) — `LinearRegressionModel_TF`

== `predict`
```python
y = tf.matmul(x, self.w) + self.b
```
#apuntes[Linear Regression · arbitrary dimension][$ y_i = bold(w)^t bold(x)_i + b $]
#pasos[Mismo apilado por filas que en el ej. 1; `tf.matmul` hace el papel de `np.dot`.]
#formula[$ bold(y) = bold(X) bold(w) + b $]
#neurona(act: $"id"$)
#diag(M($bold(X)$, $N times d$, 2, 6, cX), $dot$, M($bold(w)$, $d times 1$, 1, 2, cW), $+$, M($b$, $1 times 1$, 1, 1, cB), $=$, M($bold(y)$, $N times 1$, 1, 6, cY))
]

#block(breakable: false, width: 100%)[
== `compute_gradients` (autodiferenciación)
```python
with tf.GradientTape() as tape:
    loss = self.get_loss(x, t)
db, dw = tape.gradient(loss, [self.b, self.w])
```
#apuntes[Linear Regression · gradient descent][$ E = sum_(i=1)^N 1/2 (y_i - t_i)^2 $]
#pasos[
  Con la media en vez de la suma (`tf.reduce_mean(0.5 * (y - t) * (y - t))`):
  $ L = E / N = 1/(2N) sum_(i=1)^N (y_i - t_i)^2 $
  No derivo a mano: la cinta graba las operaciones y aplica la regla de la cadena. Da lo mismo que el ej. 1: $bold(X)^top bold(delta)$ y $sum_(i=1)^N delta_i$.
]
#formula[$ ((partial L)/(partial b), (partial L)/(partial bold(w))) = "tape.gradient"(L, [b, bold(w)]) $]
#neurona(act: $"id"$, mode: "back")
]

#block(breakable: false, width: 100%)[
== `gradient_step`
```python
self.b.assign_sub(eta * db)
self.w.assign_sub(eta * dw)
```
#apuntes[Linear Regression · arbitrary dimension][$ bold(w)_(t+1) = bold(w)_t - eta nabla_bold(w) E, quad b_(t+1) = b_t - eta (partial E) / (partial b) $]
#formula[$ b arrow.l b - eta (partial L) / (partial b), quad bold(w) arrow.l bold(w) - eta (partial L) / (partial bold(w)) $]
#neurona(act: $"id"$, mode: "upd")
`assign_sub(v)` hace `var = var - v` in-place sobre el `tf.Variable`.
]

#block(breakable: false, width: 100%)[
= Red neuronal (NumPy) — `NeuralNetwork`

Capas $l = 1, dots, L$ con $n_l$ neuronas; $bold(y)^((0)) = bold(x)$. En el código la capa $l$ es el índice `l-1` de las listas.

== `predict` (forward)
```python
yl = x
for l in range(self.nlayers):
    zl = np.dot(self.W[l], yl) + self.b[l]
    yl = self.a[l](zl)
    z.append(zl)
    y.append(yl)
```
#apuntes[Non-linear models · artificial neuron, ec. (1) y (3) — índice $i -> k$ para no chocar con los patrones][$
  z = sum_(k=1)^d w_k x_k + b, quad y = f(bold(w)^t bold(x) + b)
$]
#pasos[
  *1. Una neurona $j$ de la capa $l$.* Su entrada es la salida de la capa anterior $bold(y)^((l-1))$ y tiene sus propios pesos $bold(w)_j$ y bias $b_j$:
  $ z_j = bold(w)_j^top bold(y)^((l-1)) + b_j $
  *2. Toda la capa.* Apilo los $bold(w)_j^top$ como filas de $bold(W)^((l))$ ($n_l times n_(l-1)$) y los $b_j$ en $bold(b)^((l))$:
  $ mat(z_1; dots.v; z_(n_l)) = mat(bold(w)_1^top; dots.v; bold(w)_(n_l)^top) bold(y)^((l-1)) + mat(b_1; dots.v; b_(n_l))
    quad ==> quad bold(z)^((l)) = bold(W)^((l)) bold(y)^((l-1)) + bold(b)^((l)) $
  *3. Todo el batch.* Aquí los patrones van como *columnas* ($d times N$), así que cada columna se procesa igual y $bold(b)^((l))$ se suma a las $N$ columnas por broadcasting. La activación $f = a^((l))$ se aplica elemento a elemento.
]
#formula[$ bold(z)^((l)) = bold(W)^((l)) bold(y)^((l-1)) + bold(b)^((l)), quad bold(y)^((l)) = a^((l))(bold(z)^((l))) $]
#grid(columns: 2, align: horizon, column-gutter: 4pt, scale(62%, reflow: true, red()), scale(62%, reflow: true, capa()))
#diag(M($bold(W)^((l))$, $n_l times n_(l-1)$, 3, 2, cW), $dot$, M($bold(y)^((l-1))$, $n_(l-1) times N$, 7, 3, cX), $+$, M($bold(b)^((l))$, $n_l times 1$, 1.3, 2, cB), $=$, M($bold(z)^((l))$, $n_l times N$, 7, 2, cY))
]

#block(breakable: false, width: 100%)[
== `compute_gradients` (backpropagation)
```python
delta = dy
for l in range(self.nlayers - 1, -1, -1):
    y_prev = y[l-1] if l > 0 else x
    dW.insert(0, np.dot(delta, y_prev.T))
    db.insert(0, np.sum(delta, axis=1, keepdims=True))
    if l > 0:
        delta = np.dot(self.W[l].T, delta) * self.da[l-1](z[l-1])
```

=== Error en la última capa
#apuntes[Linear / Logistic Regression][$
  (partial E) / (partial b) = sum_(i=1)^N (y_i - t_i), quad (partial C) / (partial b) = sum_(i=1)^N (y_i - t_i)
$]
#pasos[
  Como $partial z_i \/ partial b = 1$, ambas fórmulas dicen que la derivada respecto a la preactivación es $y_i - t_i$ (lineal + MSE y sigmoide + cross-entropy; ver ej. 2). Dividiendo entre $N$ por usar la media:
]
#formula[$ bold(delta)^((L)) = (partial L) / (partial bold(z)^((L))) = (bold(y)^((L)) - bold(t)) / N $]
#red(mode: "back")
]

#block(breakable: false, width: 100%)[
=== Gradientes de $bold(W)$ y $bold(b)$
#apuntes[Linear Regression · arbitrary dimension][$ nabla_bold(w) E = sum_(i=1)^N (y_i - t_i) bold(x)_i $ #h(1em) (error de la neurona × su entrada)]
#pasos[
  *1. Neurona $j$ de la capa $l$.* Su "entrada" es $bold(y)^((l-1))$ y su error $delta_j^((l))$:
  $ nabla_(bold(w)_j) L = sum_(i=1)^N delta_(j,i)^((l)) bold(y)_i^((l-1)), quad (partial L) / (partial b_j) = sum_(i=1)^N delta_(j,i)^((l)) $
  *2. Toda la capa.* Apilando por filas ($bold(w)_j^top$ es la fila $j$ de $bold(W)^((l))$), la fila $j$ del resultado es $sum_(i=1)^N delta_(j,i) (bold(y)_i^((l-1)))^top$; eso es un producto matriz × matriz traspuesta:
  $ (partial L) / (partial bold(W)^((l))) = sum_(i=1)^N bold(delta)_i^((l)) (bold(y)_i^((l-1)))^top = bold(delta)^((l)) (bold(y)^((l-1)))^top $
]
#formula[$
  (partial L) / (partial bold(W)^((l))) = bold(delta)^((l)) (bold(y)^((l-1)))^top, quad
  (partial L) / (partial bold(b)^((l))) = sum_(i=1)^N bold(delta)^((l))_(:, i)
$]
#capa(mode: "grad")
#diag(M($bold(delta)^((l))$, $n_l times N$, 7, 2, cD), $dot$, M($(bold(y)^((l-1)))^top$, $N times n_(l-1)$, 3, 7, cX), $=$, M($partial L \/ partial bold(W)^((l))$, $n_l times n_(l-1)$, 5, 2, cW))
#diag(M($bold(delta)^((l))$, $n_l times N$, 7, 2, cD), $arrow.r^(sum_(i=1)^N)$, M($partial L \/ partial bold(b)^((l))$, $n_l times 1$, 5, 2, cB))
]

#block(breakable: false, width: 100%)[
=== Propagar el error a la capa anterior
#apuntes[Logistic Regression · gradient descent][$ sigma'(z) = sigma(z)(1 - sigma(z)) $ — los apuntes sólo tienen una neurona; esto es la regla de la cadena.]
#pasos[
  $z_k^((l-1))$ afecta a todas las neuronas $j$ de la capa $l$ a través de $y_k^((l-1)) = f(z_k^((l-1)))$ y del peso $W_(j k)$:
  $ delta_k^((l-1)) = sum_(j=1)^(n_l) (partial L) / (partial z_j^((l))) (partial z_j^((l))) / (partial y_k^((l-1))) (partial y_k^((l-1))) / (partial z_k^((l-1)))
    = f'(z_k^((l-1))) sum_(j=1)^(n_l) W_(j k) delta_j^((l)) $
  $sum_(j=1)^(n_l) W_(j k) delta_j^((l))$ es la componente $k$ de $bold(W)^top bold(delta)$; el factor $f'$ va elemento a elemento:
]
#formula[$ bold(delta)^((l-1)) = ((bold(W)^((l)))^top bold(delta)^((l))) dot.o a'^((l-1))(bold(z)^((l-1))) $]
#capa(mode: "back")
#diag(M($(bold(W)^((l)))^top$, $n_(l-1) times n_l$, 2.4, 3, cW), $dot$, M($bold(delta)^((l))$, $n_l times N$, 7, 2, cD), $dot.o$, M($a'(bold(z)^((l-1)))$, $n_(l-1) times N$, 7, 3, cY), $=$, M($bold(delta)^((l-1))$, $n_(l-1) times N$, 7, 3, cD))
$dot.o$ es el producto elemento a elemento (`*` en NumPy).
]

#block(breakable: false, width: 100%)[
== `gradient_step`
```python
for l in range(self.nlayers):
    self.W[l] = self.W[l] - (eta * dW[l])
    self.b[l] = self.b[l] - (eta * db[l])
```
#apuntes[Linear Regression · arbitrary dimension][$ bold(w)_(t+1) = bold(w)_t - eta nabla_bold(w) E, quad b_(t+1) = b_t - eta (partial E) / (partial b) $]
#pasos[La misma regla, aplicada a los parámetros de cada capa.]
#formula[$ bold(W)^((l)) arrow.l bold(W)^((l)) - eta (partial L) / (partial bold(W)^((l))), quad bold(b)^((l)) arrow.l bold(b)^((l)) - eta (partial L) / (partial bold(b)^((l))) quad forall l $]
#capa(mode: "upd")
]

#block(breakable: false, width: 100%)[
= Red neuronal (TensorFlow) — `NeuralNetwork_TF`

== `predict`
```python
y = x
for l in range(self.nlayers):
    z = tf.matmul(self.W[l], y) + self.b[l]
    y = self.a[l](z)
```
#apuntes[Non-linear models · artificial neuron, ec. (3)][$ y = f(bold(w)^t bold(x) + b) $]
#pasos[Igual que en el ej. 6 (apilado por capas y batch por columnas), encadenando las capas; sólo se devuelve la última activación.]
#formula[$ bold(y)^((0)) = bold(x), quad bold(y)^((l)) = a^((l))(bold(W)^((l)) bold(y)^((l-1)) + bold(b)^((l))), quad "salida" = bold(y)^((L)) $]
#red()
#diag(M($bold(W)^((l))$, $n_l times n_(l-1)$, 3, 2, cW), $dot$, M($bold(y)^((l-1))$, $n_(l-1) times N$, 7, 3, cX), $+$, M($bold(b)^((l))$, $n_l times 1$, 1.3, 2, cB), $arrow.r^(a^((l)))$, M($bold(y)^((l))$, $n_l times N$, 7, 2, cY))
]

#block(breakable: false, width: 100%)[
== `compute_gradients`
```python
with tf.GradientTape() as tape:
    loss = self.get_loss(x, t, loss_function)
grads = tape.gradient(loss, self.b + self.W)
db = grads[:self.nlayers]   # dL/db^(1..L)
dW = grads[self.nlayers:]   # dL/dW^(1..L)
```
#pasos[La cinta hace el backprop del ej. 6 automáticamente, para todos los parámetros a la vez.]
#formula[$ "grads" = "tape.gradient"(L, [bold(b)^((1)), dots, bold(b)^((L)), bold(W)^((1)), dots, bold(W)^((L))]) $]
#red(mode: "back")
`self.b + self.W` concatena las dos listas de Python (no suma tensores), así que los gradientes vuelven en ese mismo orden: los primeros `nlayers` son los de los bias y el resto los de los pesos.
]

#block(breakable: false, width: 100%)[
== `gradient_step`
```python
for l in range(self.nlayers):
    self.b[l].assign_sub(eta * dB[l])
    self.W[l].assign_sub(eta * dW[l])
```
#apuntes[Linear Regression · arbitrary dimension][$ bold(w)_(t+1) = bold(w)_t - eta nabla_bold(w) E, quad b_(t+1) = b_t - eta (partial E) / (partial b) $]
#formula[$ bold(W)^((l)) arrow.l bold(W)^((l)) - eta (partial L) / (partial bold(W)^((l))), quad bold(b)^((l)) arrow.l bold(b)^((l)) - eta (partial L) / (partial bold(b)^((l))) quad forall l $]
#red(mode: "upd")
]
