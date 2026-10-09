# ===============================================================================
# DLFBT 2026/2027
# Lab assignment 2
# Authors:
#   Álvaro Sáiz López   e487478
#   Yago Clerigo Ruiz   e483273
# ===============================================================================

import numpy as np
import tensorflow as tf
from time import perf_counter
from dataclasses import dataclass
from typing import Dict, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42


def set_reproducible(seed):
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def load_phoneme():
    """
    Load the phoneme dataset from the follwing directory:
    "../data/lab2/phoneme.csv"

    The CSV contains:
    - several feature columns;
    - one final binary target column.

    Returns
    -------
    X:
        Feature matrix of shape (n_samples, n_features).
    y:
        Binary target vector of shape (n_samples,).

    TODO
    ----
    Load the dataset and split the last column from the remaining columns.
    """

    # TODO: load the CSV.
    # The file is purely numeric with no header, so np.loadtxt is enough and
    # returns a single 2-D float array of shape (n_samples, n_columns).
    # The relative path assumes the notebook is run from inside Lab2/.
    path = "../data/lab2/phoneme.csv"
    dataset = np.loadtxt(path, delimiter=",")

    # TODO: perform a shape / validity check.
    # Fail early with a clear message if the file is not a 2-D table with at
    # least one feature column plus the target column.
    if dataset.ndim != 2 or dataset.shape[1] < 2:
        raise ValueError(f"Unexpected dataset shape: {dataset.shape}")
    
    # TODO:
    # Every column except the last is a feature; the last column is the 0/1
    # label. Slicing keeps X 2-D (n, 5) and makes y 1-D (n,), which is what
    # Keras expects for a single sigmoid output with binary cross-entropy.
    X = dataset[:, :-1].astype(np.float64)
    y = dataset[:, -1].astype(np.float64)

    return X, y    


def dataset_overview(X, y):
    labels, counts = np.unique(y, return_counts=True)
    return {
        "samples": int(X.shape[0]),
        "features": int(X.shape[1]),
        "class_distribution": {int(k): int(v) for k, v in zip(labels, counts)},
        "feature_mean": np.mean(X, axis=0),
        "feature_std": np.std(X, axis=0),
    }


@dataclass
class DataSplit:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray

    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray

    scaler: Optional[StandardScaler]


def prepare_data(
    X,
    y,
    test_size=0.20,
    val_size=0.20,
    normalize=True,
    random_state=RANDOM_STATE,
):
    """
    Create stratified training, validation and test partitions.

    Important note:

        FIT THE SCALER USING ONLY THE TRAINING DATA.

    Note that;
    ----
    If information from validation or test samples is used to compute
    preprocessing statistics, information leaks into the training pipeline.
    That makes the reported evaluation too optimistic.

    Parameters
    ----------
    X, y:
        Full dataset.
    test_size:
        Fraction of the full dataset assigned to the test partition.
    val_size:
        Fraction of the full dataset assigned to validation.
    normalize:
        If True, standardize features using StandardScaler.
    random_state:
        Seed passed to train_test_split.

    Returns
    -------
    DataSplit
        Object containing the three partitions and the fitted scaler.

    TODO
    ----
    Implement the full preprocessing pipeline.
    """

    # TODO: validate test_size and val_size.
    # Both are fractions of the FULL dataset, so each must lie in [0, 1] and
    # together they must leave a non-negative fraction for training.
    if test_size < 0.0 or test_size > 1.0:
        raise ValueError(f"Unexpected split ratio test_size : {test_size}")
    if val_size < 0.0 or val_size > 1.0:
        raise ValueError(f"Unexpected split ratio val_size : {val_size}")
    train_size = 1 - val_size - test_size
    if train_size < 0.0 or train_size > 1.0:
        raise ValueError(f"Unexpected split ratio train_size : {train_size}")
        
    # TODO: first split -> train+validation and test.
    # Hold out the test set first so it never takes part in any later
    # decision. stratify=y keeps the ~70/30 class ratio in both parts.
    X_trval, X_test, y_trval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )
    
    # TODO: compute the validation fraction relative to train+validation.
    # train_test_split works on what is left after removing the test set, so
    # val_size has to be rescaled: 0.20 / (1 - 0.20) = 0.25 of the remaining
    # 80% gives 20% of the full dataset, i.e. a 60/20/20 split.
    val_fraction = val_size / (1.0 - test_size)

    # TODO: second split -> train and validation.
    # Stratify again (on y_trval) so validation has the same class balance.
    X_train, X_val, y_train, y_val = train_test_split(
        X_trval, y_trval, test_size=val_fraction, random_state=random_state, stratify=y_trval,
    )

    # TODO: optionally fit StandardScaler ONLY on X_train.
    # The mean and std are learned from the training set alone and then
    # applied to validation and test with transform(). Fitting on all the data
    # would leak validation/test statistics into training (data leakage) and
    # make the evaluation optimistic. After this, X_train has mean 0, std 1.
    scaler = None
    if normalize:
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
                
    # TODO: return DataSplit(...)
    # The scaler is returned too (None if normalize=False) so new data can be
    # transformed with the same training statistics.

    return DataSplit(X_train, X_val, X_test, y_train, y_val, y_test, scaler)


def build_baseline_model(input_dim, hidden_units=8):
    """
    Build the small baseline multilayer perceptron used in Part 1.

    Architecture
    ------------
    Input
      -> Dense(hidden_units, activation="relu")
      -> Dense(1, activation="sigmoid")

    Important
    ---------
    This function should BUILD the model, but it should NOT compile it.

    TODO
    ----
    Create and return the Keras Sequential model.
    """

    # TODO:
    # One hidden ReLU layer and a single sigmoid unit whose output is read as
    # P(y = 1 | x). With input_dim=5 and 8 hidden units this has
    # 5*8 + 8 + 8*1 + 1 = 57 trainable parameters. Not compiled here, so the
    # caller can choose the optimizer.
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(hidden_units, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    return model


def compile_binary_model(
    model,
    optimizer="adam",
    learning_rate=None,
):
    """
    Compile a Keras model for binary classification.

    Required settings
    -----------------
    loss:
        "binary_crossentropy"

    metric:
        "accuracy"

    optimizer:
        May be either:
        - a ready-created Keras optimizer object; or
        - one of the supported strings:
              "sgd"
              "adagrad"
              "rmsprop"
              "adam"

    TODO
    ----
    Compile the model and return it.
    """

    # TODO:
        # If optimizer is a string AND learning_rate is provided,
        # create the appropriate tf.keras.optimizers.* object.

    if isinstance(optimizer, str) and learning_rate is not None:
        optimizer = optimizer_from_name(optimizer, learning_rate=learning_rate)

        # TODO:
        # Otherwise pass `optimizer` through unchanged: Keras accepts either
        # an optimizer object or a string name (which uses its default
        # learning rate).

    # Binary cross-entropy is the negative log-likelihood of a Bernoulli
    # output, the natural loss for a sigmoid unit; accuracy uses a 0.5
    # threshold on the predicted probability.
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


def make_early_stopping(patience=20):
    return tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
    )


def train_model(
    model,
    split,
    *,
    epochs=250,
    batch_size=32,
    patience=20,
    verbose=0,
):
    """
    Train a model using the explicit training and validation partitions.

    ------------------
    Use:

        split.X_train
        split.y_train

    for fitting and:

        split.X_val
        split.y_val

    as validation_data.

    Also include the EarlyStopping callback returned by
    `make_early_stopping(patience)`.

    Returns
    -------
    history:
        The Keras History object returned by `model.fit`.

    TODO
    ----
    Call `model.fit(...)` with the appropriate arguments.
    """

    # TODO:
    # Fit on the training partition only and pass the validation partition as
    # validation_data, so val_loss is computed at the end of every epoch.
    # EarlyStopping watches val_loss: after `patience` epochs without
    # improvement it stops and restores the weights of the best epoch, so
    # `epochs` is only an upper bound. The test set is not touched here.
    callback = make_early_stopping(patience=patience)
    history = model.fit(
        x=split.X_train,
        y=split.y_train,
        batch_size = batch_size,
        epochs = epochs,
        verbose = verbose,
        callbacks = callback,
        validation_data=(split.X_val,split.y_val)
        
    )

    return history


def evaluate_model(model, split):
    """
    Evaluate the trained model ONCE on the held-out test set.

    The returned dictionary must contain:

        "test_loss"
        "test_accuracy"

    Use:
        split.X_test
        split.y_test

    TODO
    ----
    Evaluate the model and return the metrics as normal Python floats.
    """

    # TODO:
    # loss, accuracy = model.evaluate(...)
    # evaluate() returns [loss, metric] in the order given to compile(). It
    # runs on the test partition, which played no part in fitting, scaling or
    # early stopping, so this is the unbiased estimate of generalization.
    # verbose=0 suppresses the progress bar printed on every evaluation.
    loss, accuracy = model.evaluate(
        x=split.X_test,
        y=split.y_test,
        verbose=0,
    )

    results = {
        "test_loss": float(loss),
        "test_accuracy": float(accuracy),
    }

    return results

    #raise NotImplementedError("TODO: implement evaluate_model")


def build_improved_model(input_dim):
    """
    Build a modestly improved network.

    Required architecture
    ---------------------
    Input
      -> Dense(32, relu)
      -> Dropout(0.15)
      -> Dense(16, relu)
      -> Dense(1, sigmoid)

    Compile it with Adam and a learning rate of 1e-3.

    The intention is NOT to build a huge model. We want a small, controlled
    change that can be compared with the baseline.

    TODO
    ----
    Build, compile and return the model.
    """

    # TODO: build the architecture.
    # Two hidden layers (32 -> 16) add capacity over the 8-unit baseline,
    # and light dropout (15% of the first layer's activations zeroed at each
    # training step) stops that extra capacity from overfitting.
    # 737 parameters in total.
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.15),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1,activation="sigmoid")
    ])
    

    # TODO: compile it, preferably by reusing compile_binary_model(...).
    # Reusing the helper keeps loss, metric and optimizer identical to the
    # baseline, so the only difference between the two is the architecture.

    compile_binary_model(model,optimizer="adam",learning_rate=1e-3)

    return model

    raise NotImplementedError("TODO: implement build_improved_model")


def build_regularized_model(input_dim, l2_strength=1e-4):
    """
    Build an MLP using L2 kernel regularization.

    Required architecture
    ---------------------
    Dense(32, relu, kernel_regularizer=L2)
      -> Dense(16, relu, kernel_regularizer=L2)
      -> Dense(1, sigmoid)

    Important
    ---------
    This function only builds the model. The notebook will compile it later.

    TODO
    ----
    Create the L2 regularizer and apply it to both hidden Dense layers.
    """

    # TODO:
    # L2 adds l2_strength * sum(w^2) of each hidden kernel to the loss, which
    # pulls the weights towards zero (weight decay) and favours smoother
    # decision boundaries. Only the kernels are penalized, not the biases.
    # The same architecture as the dropout model, so the two are comparable.
    reg = tf.keras.regularizers.l2(l2_strength)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(32, activation="relu",kernel_regularizer=reg),
        tf.keras.layers.Dense(16, activation="relu",kernel_regularizer=reg),
        tf.keras.layers.Dense(1,activation="sigmoid")
    ])

    return model

    # TODO: build and return the Sequential model.

    raise NotImplementedError("TODO: implement build_regularized_model")


def build_dropout_model(input_dim, rate=0.25):
    """
    Build an MLP that uses dropout.

    Required architecture
    ---------------------
    Dense(32, relu)
      -> Dropout(rate)
      -> Dense(16, relu)
      -> Dense(1, sigmoid)

    Important
    ---------
    This function only builds the model. The notebook will compile it later.

    TODO
    ----
    Create and return the model.

    """

    # Dropout randomly zeroes a fraction `rate` of the first hidden layer's
    # outputs at each training step (scaling the rest by 1 / (1 - rate)), so
    # units cannot co-adapt. Keras disables it automatically at inference.
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(rate),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1,activation="sigmoid")
    ])

    return model

    # TODO: build and return the Sequential model.

    raise NotImplementedError("TODO: implement build_dropout_model")


def optimizer_from_name(name, learning_rate=1e-3):
    """
    Construct one of the optimizers studied in Part 2.

    Supported names
    ---------------
    "sgd"
        Standard stochastic gradient descent.

    "momentum"
        SGD with momentum=0.9.

    "nesterov"
        SGD with momentum=0.9 and nesterov=True.

    "adagrad"
        AdaGrad.

    "rmsprop"
        RMSprop.

    "adam"
        Adam.

    All optimizers must use the provided learning rate.

    Hints
    -----
    You may normalize the input string using:

        name.lower()
        .replace(" ", "")
        .replace("-", "")

    Raise ValueError for an unknown optimizer.

    TODO
    ----
    Implement the optimizer selection logic.
    """

    # TODO: return the matching tf.keras.optimizers optimizer.
    # sgd:      w <- w - lr * g
    # momentum: SGD plus a velocity term (beta = 0.9) that accumulates past
    #           gradients, speeding up consistent directions.
    # nesterov: momentum with the gradient taken at the look-ahead point.
    # adagrad:  per-parameter rate lr / sqrt(sum of past g^2); it only shrinks.
    # rmsprop:  like Adagrad but with a moving average of g^2, so it does not
    #           decay to zero.
    # adam:     RMSprop-style scaling plus momentum, with bias correction.
    # All of them receive the same learning_rate so only the update rule
    # changes between experiments.
    # Normalize the name so variants such as "Adam" or "RMS-prop" also match.
    key = name.lower().replace(" ", "").replace("-", "")
    match key:
        case "sgd":
                return tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.0)
        case "momentum":
                return tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        case "nesterov":
            return tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)
        case "adagrad":
            return tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
        case "rmsprop":
            return tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        case "adam":
            return tf.keras.optimizers.Adam(learning_rate=learning_rate)
        case _:
            # Fail loudly instead of returning None, which would only break
            # later inside model.compile().
            raise ValueError(f"Unknown optimizer: {name!r}")
               


def run_optimizer_experiment(
    split,
    optimizer_name,
    epochs=150,
    batch_size=32,
    learning_rate=1e-3,
    patience=15,
    seed=RANDOM_STATE,
):
    """
    Train ONE optimizer under controlled conditions.

    Fair-comparison requirements
    ----------------------------
    Keep these fixed:
    - data split;
    - architecture;
    - seed;
    - batch size;
    - stopping rule.

    Only the optimizer should change.


    The notebook expects:

        "optimizer"
        "epochs_run"
        "seconds"
        "test_loss"
        "test_accuracy"
        "history"
        "model"

    TODO
    ----
    Implement one complete experiment.
    """

    # TODO: make the run reproducible.
    # Reset the seeds before building the model so every optimizer starts
    # from the same initial weights and sees the same batch order.
    set_reproducible(seed)
    
    # TODO: build model.
    # Always the same baseline architecture: the optimizer is the only
    # variable in this experiment.
    model = build_baseline_model(split.X_train.shape[1])
    
    # TODO: construct optimizer.
    # A fresh optimizer object per run, so no state (momentum, accumulators)
    # carries over from a previous experiment.
    optimizer = optimizer_from_name(optimizer_name, learning_rate=learning_rate)
    
    # TODO: compile model.
    compile_binary_model(model=model, optimizer=optimizer)
    
    # TODO: record start time.
    # perf_counter is a monotonic high-resolution clock meant for measuring
    # intervals (it must be called: perf_counter(), not perf_counter).
    start = perf_counter()
    
    # TODO: train.
    # Same epochs, batch size and early-stopping rule for every optimizer.
    history = train_model(
        model=model,
        split=split, 
        epochs=epochs, 
        batch_size=batch_size, 
        patience=patience, 
        verbose=0,
    )

    # TODO: compute elapsed time.
    # Only the fit call is timed; model building and evaluation are excluded.
    time = perf_counter() - start # Time in seconds
    
    # TODO: evaluate.
    # Held-out test metrics, computed once after training.
    metrics = evaluate_model(model=model, split=split)

    # TODO: return the result dictionary.
    # epochs_run is the length of the loss history, i.e. how many epochs ran
    # before early stopping fired (or the epochs limit was reached).

    return {
        "optimizer": optimizer_name,
        "epochs_run": len(history.history["loss"]),
        "seconds": time,
        "test_loss": metrics["test_loss"],
        "test_accuracy": metrics["test_accuracy"],
        "history": history,
        "model": model,
    }

    raise NotImplementedError("TODO: implement run_optimizer_experiment")


def compare_optimizers(
    split,
    names=(
        "sgd",
        "momentum",
        "nesterov",
        "adagrad",
        "rmsprop",
        "adam",
    ),
    **kwargs,
):
    """
    Run `run_optimizer_experiment` for every requested optimizer.

    Parameters
    ----------
    split:
        The common DataSplit used for every experiment.
    names:
        Iterable containing optimizer names.
    **kwargs:
        Additional keyword arguments forwarded to run_optimizer_experiment.

    Returns
    -------
    list
        One result dictionary per optimizer.

    TODO
    ----
    A compact loop or list comprehension is sufficient.
    """

    # TODO: call run_optimizer_experiment once per optimizer name.
    # The same split and the same kwargs (epochs, batch_size, learning_rate,
    # patience, seed) are forwarded to every run, which is what makes the
    # comparison fair.


    return [
        run_optimizer_experiment(split, name, **kwargs)
        for name in names
    ]

    raise NotImplementedError("TODO: implement compare_optimizers")


def available_devices():
    return {
        "CPU": [d.name for d in tf.config.list_physical_devices("CPU")],
        "GPU": [d.name for d in tf.config.list_physical_devices("GPU")],
    }
