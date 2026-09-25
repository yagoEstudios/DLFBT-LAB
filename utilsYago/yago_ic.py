# yago_ic.py
import numpy as np
from icecream import ic, argumentToString


def _formato(x):
    if isinstance(x, np.ndarray):
        return f"shape={x.shape} dtype={x.dtype}\n{x}"
    if type(x).__module__.startswith("tensorflow") and hasattr(x, "numpy"):
        return f"shape={tuple(x.shape)} dtype={x.dtype.name}\n{x.numpy()}"
    return argumentToString(x)   # el resto, como siempre


ic.configureOutput(argToStringFunction=_formato)


