// Diagramas de neuronas para formulas_lab1.typ
#import "@preview/cetz:0.4.2": canvas, draw

#let cRed = rgb("#c0392b")
#let cOr = rgb("#d68910")
#let cGr = rgb("#1e8449")
#let cNode = rgb("#eef3fb")

// Flecha de a a b recortada ra al inicio y rb al final (para no pisar los círculos)
#let flecha(a, b, ra: 0, rb: 0, stroke: 0.6pt + black, mark: true) = {
  let (dx, dy) = (b.at(0) - a.at(0), b.at(1) - a.at(1))
  let n = calc.sqrt(dx * dx + dy * dy)
  let (ux, uy) = (dx / n, dy / n)
  draw.line(
    (a.at(0) + ux * ra, a.at(1) + uy * ra), (b.at(0) - ux * rb, b.at(1) - uy * rb),
    stroke: stroke,
    mark: if mark { (end: "stealth", fill: stroke.paint, scale: 0.55) } else { none },
  )
}
#let etiqueta(p, body, col: black) = draw.content(p, box(fill: white, inset: 1.2pt, text(8.5pt, fill: col, body)))
#let nodo(p, body, r: 0.32, fill: cNode) = {
  draw.circle(p, radius: r, fill: fill, stroke: 0.6pt)
  draw.content(p, text(if r < 0.35 { 7pt } else { 8pt }, body))
}

// Una neurona: y = act(w^T x + b). mode: "fwd" | "back" | "upd"
#let neurona(act: $f$, mode: "fwd", nota: none) = align(center, canvas(length: 1.05cm, {
  import draw: *
  let S = (3.4, 0)
  let A = (5.1, 0)
  let Y = (6.8, 0)
  let ins = (($x_1$, 1.8, $w_1$), ($x_2$, 0.9, $w_2$), ($x_d$, -0.9, $w_d$))
  let wcol = if mode == "upd" { cOr } else if mode == "back" { cRed } else { black }
  let wst = if mode == "fwd" { 0.6pt + black } else { 1.1pt + wcol }
  for (l, y, w) in ins {
    nodo((0, y), l)
    flecha((0, y), S, ra: 0.32, rb: 0.55, stroke: wst)
    etiqueta((0.95, y - (y / 3.4) * 0.95), w, col: wcol)
  }
  content((0, 0.05), $dots.v$)
  nodo((0, -2.0), $1$, fill: rgb("#e3f5e3"))
  flecha((0, -2.0), S, ra: 0.32, rb: 0.55, stroke: if mode == "fwd" { 0.6pt + cGr } else { 1.1pt + wcol })
  etiqueta((0.95, -2.0 + (2.0 / 3.4) * 0.95), $b$, col: if mode == "fwd" { cGr } else { wcol })
  circle(S, radius: 0.55, fill: cNode, stroke: 0.8pt)
  content(S, text(13pt, $sum$))
  content((S.at(0), 0.85), text(8.5pt)[$z = bold(w)^top bold(x) + b$])
  rect((A.at(0) - 0.45, -0.4), (A.at(0) + 0.45, 0.4), fill: rgb("#f4ecfb"), stroke: 0.8pt, radius: 3pt)
  content(A, text(9pt, act))
  flecha(S, A, ra: 0.55, rb: 0.45)
  content(((S.at(0) + A.at(0)) / 2, 0.22), text(8.5pt, $z$))
  flecha(A, Y, ra: 0.45, rb: 0.0)
  content((Y.at(0) + 0.1, 0), anchor: "west", text(9pt, $y$))
  if mode == "back" {
    flecha((Y.at(0) + 0.2, -0.7), (S.at(0) + 0.2, -0.7), stroke: 1pt + cRed)
    content(((Y.at(0) + S.at(0)) / 2 + 0.2, -1.1), text(8.5pt, fill: cRed, $delta_i = (partial L) / (partial z_i) = (y_i - t_i) / N$))
    content((3.6, -2.3), anchor: "west", text(8.5pt, fill: cRed,
      $(partial L) / (partial w_k) = sum_(i=1)^N delta_i x_(i k), quad (partial L) / (partial b) = sum_(i=1)^N delta_i$))
  }
  if mode == "upd" {
    content((3.6, -2.0), anchor: "west", text(8.5pt, fill: cOr,
      $w_k arrow.l w_k - eta (partial L) / (partial w_k), quad b arrow.l b - eta (partial L) / (partial b)$))
  }
  if nota != none { content((3.4, 2.7), text(8.5pt, fill: luma(80), nota)) }
}))

// Grafo x -> f -> y (BasicTF). mode: "diff" | "gd"
#let grafo(mode: "diff") = align(center, canvas(length: 1.05cm, {
  import draw: *
  nodo((0, 0), $x$, r: 0.4)
  rect((1.8, -0.4), (2.8, 0.4), fill: rgb("#f4ecfb"), stroke: 0.8pt, radius: 3pt)
  content((2.3, 0), $f$)
  nodo((4.6, 0), $y$, r: 0.4)
  flecha((0, 0), (1.8, 0), ra: 0.4)
  flecha((2.8, 0), (4.6, 0), rb: 0.4)
  flecha((4.6, -0.7), (0, -0.7), stroke: 1pt + cRed)
  content((2.3, -1.05), text(8.5pt, fill: cRed, $(d y) / (d x) = f'(x)$))
  if mode == "gd" {
    arc((-0.35, 0.3), start: -20deg, stop: 200deg, radius: 0.5, stroke: 1pt + cOr, mark: (end: "stealth", fill: cOr, scale: 0.55))
    content((0, 1.35), text(8.5pt, fill: cOr, $x arrow.l x - eta f'(x)$))
  }
}))

// Una capa densa l-1 -> l. mode: "fwd" | "grad" | "back" | "upd"
#let capa(mode: "fwd") = align(center, canvas(length: 1.05cm, {
  import draw: *
  let L0 = ((1.8, $1$), (0.2, $k$), (-1.8, $n_(l-1)$))
  let L1 = ((1.5, $1$), (0, $j$), (-1.5, $n_l$))
  let xl = 0
  let xr = 4.2
  let hl = if mode == "fwd" or mode == "upd" { cOr } else { cRed }
  for (ya, _) in L0 { for (yb, _) in L1 {
    flecha((xl, ya), (xr, yb), ra: 0.32, rb: 0.32, stroke: 0.4pt + luma(170), mark: false)
  } }
  flecha((xl, 0.2), (xr, 0), ra: 0.32, rb: 0.32, stroke: 1.3pt + hl, mark: mode != "back")
  if mode == "back" { flecha((xr, -0.25), (xl, -0.05), ra: 0.32, rb: 0.32, stroke: 1.3pt + cRed) }
  for (y, l) in L0 { nodo((xl, y), l) }
  for (y, l) in L1 { nodo((xr, y), l) }
  content((xl, -0.8), $dots.v$)
  content((xr, -0.75), $dots.v$)
  content((xl, 1.0), $dots.v$)
  content((xr, 0.75), $dots.v$)
  content((xl, 2.6), text(8pt)[capa $l-1$])
  content((xr, 2.6), text(8pt)[capa $l$])
  // bias de la neurona j
  nodo((xr - 1.3, -2.6), $1$, r: 0.25, fill: rgb("#e3f5e3"))
  flecha((xr - 1.3, -2.6), (xr, 0), ra: 0.25, rb: 0.32, stroke: 0.6pt + cGr)
  etiqueta((xr - 0.55, -1.4), $b_j$, col: cGr)
  let lbl = if mode == "fwd" { $W_(j k)$ } else if mode == "upd" { $W_(j k) arrow.l W_(j k) - eta (partial L)/(partial W_(j k))$ } else if mode == "grad" { $(partial L) / (partial W_(j k))$ } else { $W_(j k)$ }
  etiqueta((2.1, 0.55), lbl, col: hl)
  if mode == "fwd" or mode == "upd" {
    content((xl - 0.45, 0.2), anchor: "east", text(8.5pt, $y_k^((l-1))$))
    content((xr + 0.45, 0), anchor: "west", text(8.5pt,
      $z_j^((l)) = sum_(k=1)^(n_(l-1)) W_(j k) y_k^((l-1)) + b_j, quad y_j^((l)) = a(z_j^((l)))$))
  }
  if mode == "grad" {
    content((xl - 0.45, 0.2), anchor: "east", text(8.5pt, $y_(k,i)^((l-1))$))
    content((xr + 0.45, 0), anchor: "west", text(8.5pt, fill: cRed, $delta_(j,i)^((l))$))
    content((2.1, -3.4), text(8.5pt, fill: cRed,
      $(partial L) / (partial W_(j k)) = sum_(i=1)^N delta_(j,i)^((l)) y_(k,i)^((l-1)), quad (partial L) / (partial b_j) = sum_(i=1)^N delta_(j,i)^((l))$))
  }
  if mode == "back" {
    content((xr + 0.45, 0), anchor: "west", text(8.5pt, fill: cRed, $delta_j^((l))$))
    content((xl - 0.45, 0.2), anchor: "east", text(8.5pt, fill: cRed, $delta_k^((l-1))$))
    content((2.1, -3.4), text(8.5pt, fill: cRed,
      $delta_k^((l-1)) = a'(z_k^((l-1))) sum_(j=1)^(n_l) W_(j k) delta_j^((l))$))
  }
}))

// Red completa (entrada, oculta, salida). mode: "fwd" | "back" | "upd"
#let red(mode: "fwd") = align(center, canvas(length: 1.05cm, {
  import draw: *
  let cols = ((0, (1.5, 0.5, -0.5, -1.5)), (3, (2.0, 1.0, 0, -1.0, -2.0)), (6, (1.0, 0, -1.0)), (9, (0,)))
  let ec = if mode == "back" { cRed } else if mode == "upd" { cOr } else { luma(150) }
  for c in range(cols.len() - 1) {
    let (xa, ya) = cols.at(c)
    let (xb, yb) = cols.at(c + 1)
    for a in ya { for b in yb {
      flecha((xa, a), (xb, b), ra: 0.3, rb: 0.3, stroke: 0.45pt + ec, mark: false)
    } }
    let lbl = if mode == "upd" { $bold(W)^((#(c + 1))), bold(b)^((#(c + 1)))$ } else { $bold(W)^((#(c + 1)))$ }
    content(((xa + xb) / 2, 2.65), text(8pt, fill: if mode == "fwd" { black } else { ec }, lbl))
  }
  for (x, ys) in cols { for y in ys { nodo((x, y), [], r: 0.3) } }
  content((0, -2.6), text(8.5pt)[$bold(y)^((0)) = bold(x)$])
  content((3, -2.8), text(8.5pt)[$bold(y)^((1))$])
  content((6, -1.8), text(8.5pt)[$bold(y)^((2))$])
  content((9, -0.7), text(8.5pt)[$bold(y)^((L))$])
  if mode == "fwd" { flecha((1.5, -3.4), (7.5, -3.4)); content((4.5, -3.75), text(8.5pt)[forward: $bold(z)^((l)) = bold(W)^((l)) bold(y)^((l-1)) + bold(b)^((l))$]) }
  if mode == "back" {
    flecha((7.5, -3.4), (1.5, -3.4), stroke: 1pt + cRed)
    content((4.5, -3.75), text(8.5pt, fill: cRed)[backward: $bold(delta)^((L)) = (bold(y)^((L)) - bold(t)) \/ N ->  bold(delta)^((l-1)) = (bold(W)^((l)top) bold(delta)^((l))) dot.o a'(bold(z)^((l-1)))$])
    content((9.5, 0), anchor: "west", text(8.5pt, fill: cRed, $bold(delta)^((L))$))
  }
  if mode == "upd" { content((4.5, -3.6), text(8.5pt, fill: cOr)[todas las $bold(W)^((l)), bold(b)^((l))$ se actualizan a la vez con su gradiente]) }
}))
