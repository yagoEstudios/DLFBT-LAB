#set page(paper: "a4", margin: 2cm)
#set text(lang: "en", size: 10pt, font: ("Lato", "Arial"))
#set par(justify: true)
#show figure.caption: set text(size: 9pt)

#let fig(name, cap, w: 100%) = figure(image("figures/" + name + ".png", width: w), caption: cap)
#let grid4(prefix, cap) = figure(
  grid(
    columns: 2,
    gutter: 6pt,
    ..("dnn", "rnn", "cnn", "lstm").map(m => image("figures/" + m + "_" + prefix + ".png", width: 100%)),
  ),
  caption: cap,
)

= Results

This document summarises the comparison of four deep-learning architectures for multiclass human activity
recognition: a fully connected network (DNN), a vanilla recurrent network (RNN), a 1D convolutional network (CNN)
and a Long Short-Term Memory network (LSTM). Every model receives the same 561 engineered features per sample and
predicts one of six activities: LAYING, SITTING, STANDING, WALKING, WALKING_DOWNSTAIRS and WALKING_UPSTAIRS.

All models were trained with the same configuration: Adam with learning rate $10^(-3)$, batch size 64, at most
50 epochs and early stopping with a patience of 8 epochs, using seed 42. Training ran on an NVIDIA RTX 3050 Laptop
GPU. The metrics below are computed on the held-out test set (2947 samples).

#let data = csv("metrics/model_comparison.csv")
#let header = data.at(0)
#let rows = data.slice(1)
#let f(x, n) = {
  let s = str(calc.round(float(x), digits: n))
  if not s.contains(".") { s += "." }
  s + "0" * (n - s.split(".").at(1).len())
}

#show table: set text(size: 7.5pt)
#figure(
  table(
    columns: header.len(),
    align: (left,) + (right,) * (header.len() - 1),
    stroke: none,
    table.hline(),
    ..header.map(h => [*#h*]),
    table.hline(stroke: 0.5pt),
    ..rows
      .map(r => (
        r.at(0),
        ..r.slice(1, 6).map(x => f(x, 4)),
        r.at(6),
        f(r.at(7), 2) + " s",
        f(r.at(8), 4) + " s",
      ))
      .flatten(),
    table.hline(),
  ),
  caption: [Test metrics, trainable parameters, total training time and inference time on the test set.],
)

== Predictive performance

The *DNN* is clearly the best model, with an accuracy of 0.952 and a macro F1 of 0.952. The *CNN* comes second
(macro F1 0.845), followed by the *LSTM* (0.787) and the *RNN* (0.756). The gap between the DNN and the other
models is large, around 10 points of macro F1 with respect to the CNN.

For every model, macro and weighted F1 are very close, and precision and recall are similar. This is consistent
with the test set being roughly balanced (between 420 and 537 samples per class), so no single class dominates
the averages.

#fig("predictive_performance", [Predictive metrics per model.], w: 85%)

The ranking suggests that, for these features, treating the 561 values as a single vector works better than
treating them as a sequence. The features are engineered statistics (means, energies, frequency coefficients…),
so their order does not carry the temporal meaning that recurrent and convolutional layers are designed to exploit.

== Computational cost

The DNN has by far the most parameters (178,310, against 52,806 for the CNN, 17,542 for the LSTM and only 4,678
for the RNN), yet it is the fastest model both to train (6.3 s) and to run at inference (0.004 s for the whole
test set). A fully connected layer processes the full input in a single matrix product, which is very efficient on
a GPU. The CNN and LSTM are about 10 times slower at inference than the DNN.

#grid(
  columns: 2,
  gutter: 8pt,
  fig("macro_f1_vs_training_time", [Macro F1 vs. training time.]),
  fig("macro_f1_vs_model_size", [Macro F1 vs. number of trainable parameters.]),
  fig("trainable_parameters", [Trainable parameters.]),
  fig("total_training_time", [Total training time.]),
)

Total training time depends on both the cost per epoch and the number of epochs before early stopping. The RNN
has the cheapest epochs (0.28 s) but needed 34 epochs, while the DNN and the CNN stopped after 20 and the LSTM
after 27. No model reached the limit of 50 epochs.

#fig("inference_time", [Inference time on the test set.], w: 60%)

In this experiment, a larger model is not a more expensive one: the DNN offers the best accuracy at the lowest
time cost, while the smaller recurrent models are both slower and less accurate.

== Training curves

The DNN converges in a few epochs, and its validation loss stays low and close to the training loss. The RNN and
the LSTM also learn steadily, although they converge more slowly and their validation loss is noisier.

The CNN shows the most unstable behaviour. Its training loss keeps decreasing, but the validation loss oscillates
strongly, with peaks of up to about 1.7. This points to sensitivity to the learning rate or some overfitting, and
it explains why its test performance is below what its training accuracy would suggest.

#grid4("loss", [Training and validation loss (DNN, RNN, CNN, LSTM).])
#grid4("accuracy", [Training and validation accuracy (DNN, RNN, CNN, LSTM).])

== Confusion matrices

LAYING is recognised almost perfectly by all models (the DNN classifies all 537 samples correctly). The errors
concentrate in two groups of similar activities:

- *Static activities (SITTING vs. STANDING).* The DNN still confuses 56 SITTING samples as STANDING. The LSTM
  confuses 114 in the same direction, and the CNN fails the other way, labelling 133 STANDING samples as SITTING.
- *Dynamic activities (WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS).* This is where the recurrent models lose
  most of their accuracy: the RNN reaches an F1 of only 0.57 on WALKING_DOWNSTAIRS, and the LSTM labels 139
  WALKING_UPSTAIRS samples as WALKING. The CNN also struggles with WALKING_UPSTAIRS (recall 0.66).

Errors between static and dynamic activities are much less frequent than errors within each group (the DNN makes
almost none), so all models learn this coarse separation; the differences lie mainly in distinguishing activities
within each group.

#grid4("confusion_matrix", [Test confusion matrices (DNN, RNN, CNN, LSTM).])
