# ===============================================================================
# DLFBT 2026/2027
# Lab assignment 1
# Authors:
#   Álvaro Sáiz López   e487478
#   Yago Clerigo Ruiz   e483273
# ===============================================================================



import numpy as np
import tensorflow as tf


#
# LEGEND used in the comments:
#   [MATMUL]       matrix product (np.dot / tf.matmul / @): (a,k)·(k,b) -> (a,b), sums over k
#   [ELEMENT-WISE] same position <-> same position (*, +, -, /, **, exp, log, sigmoid)
#   [BROADCAST]    a size-1 dimension (or a scalar) is virtually copied to match
#   [REDUCTION]    a dimension is collapsed by sum/mean (np.sum, np.mean, tf.reduce_mean)
#   Q:             a question they may ask you, with the answer right below it
#
# DATA LAYOUTS (very important for shapes):
#   Ex 1-5 -> one sample per ROW:    x (N, d),  w (d, 1),  b (1, 1),  y, t (N, 1)
#   Ex 6-7 -> one sample per COLUMN: x (d, N),  W[l] (n_l, n_{l-1}),  b[l] (n_l, 1),  y, t (1, N)
#
# GOLDEN RULE: a gradient always has EXACTLY the same shape as the parameter it updates.
# ===============================================================================

# ===============================================================================
# EXERCISE 1 - Linear regression in NumPy, gradients derived by hand
#
#   Model:  y = x·w + b
#   Loss:   L = (1/2N) * sum_n (y_n - t_n)^2        (MSE, the 0.5 simplifies the derivative)
#   Grads:  dL/db = sum_n (y_n - t_n)/N
#           dL/dw = x^T · (y - t)/N
# ===============================================================================
class LinearRegressionModel(object):
    """
    Linear regression model for exercise 1

    Parameters
    ----------
    d : int
        Dimension

    Attributes
    ----------
    w : array
        Weight vector of shape (d, 1)
    b : array
        Bias term of shape (1, 1)
    """

    def __init__(self, d=2):
        # Random normal init. NumPy arrays are float64 by default.
        self.w = np.random.randn(d, 1)   # (d, 1): one weight per input feature
        self.b = np.random.randn(1, 1)   # (1, 1): kept 2-D so it broadcasts cleanly

    def predict(self, x):
        """
        Predicts output y for input batch x

        Parameters
        ----------
        x : array
            Input batch of shape (N, d), where N is the number of patterns and
            d is the dimension

        Returns
        -------
        y : array
            Ouput batch of shape (N, 1) which is the result of applying the
            linear regression model to the input x
        """

        # --- TO-DO block: Compute the model output y -- DONE
        # [MATMUL]    (N, d)·(d, 1) = (N, 1): each row of x (one sample) times w
        # [BROADCAST] b is (1, 1) -> copied to all N rows, so every sample gets the same bias
        # Q: Why np.dot and not x * self.w?
        #    x * self.w is element-wise: (N,d)*(d,1) only works if d == N or d == 1, and even
        #    then it does not SUM over the features. The model needs sum_j x_nj * w_j -> matmul.
        y = np.dot(x,self.w) + self.b
        # --- End of TO-DO block

        return y

    def compute_gradients(self, x, t):
        """
        Calculates the gradients of the loss function with respect to the model
        parameters b and w, for an input batch (x, t)

        Parameters
        ----------
        x : array
            Input batch of shape (N, d), where N is the number of patterns and
            d is the dimension
        t : array
            Array of shape (N, 1) with the target values for each x

        Returns
        -------
        db : array
             Gradient of the loss with respect to the bias, shape (1, 1)
        dw : array
             Gradient of the loss with respect to the weights, shape (d, 1)
        """
        y = self.predict(x)   # forward pass, (N, 1)

        # --- TO-DO block: Compute the gradients db and dw -- DONE
        # delta = dL/dy for each sample.  [ELEMENT-WISE] (N,1) - (N,1), then / scalar N
        # x.shape[0] = N because samples are ROWS in this exercise.
        # Q: Where does (y - t)/N come from?
        #    L = (1/2N) sum (y-t)^2  ->  dL/dy = 2*(1/2N)*(y-t) = (y-t)/N  (the 0.5 cancels the 2)
        delta = (y - t) / x.shape[0]

        # [REDUCTION] sum over the samples (axis=0 = rows disappear): (N,1) -> (1,1)
        # dL/db = sum_n delta_n * dy_n/db = sum_n delta_n   (because dy_n/db = 1)
        # Q: Why keepdims=True?
        #    So db stays (1,1), the same shape as b. Without it the result is 1-D, shape (1,).
        #    Here (1,1) - (1,) still works by luck, but in Ex 6 with more neurons it breaks.
        db = np.sum(delta, axis = 0, keepdims=True)

        # [MATMUL] (d, N)·(N, 1) = (d, 1) = shape of w  (golden rule)
        # dL/dw_j = sum_n delta_n * x_nj   -> that sum over samples is exactly x^T · delta
        # Q: Why x.T and not x?  np.dot(x, delta) = (N,d)·(N,1) -> inner dims d != N -> error.
        #    The transpose puts the sample dimension N in the middle so it gets summed.
        dw = np.dot(x.T, delta)
        # --- End of TO-DO block
        

        return db, dw

    def gradient_step(self, x, t, eta):
        """
        Updates the model parameters with an input batch (x, t)

        Parameters
        ----------
        x : array
            Input batch of shape (N, d), where N is the number of patterns and
            d is the dimension
        t : array
            Array of shape (N, 1) with the target values for each x
        eta : float
            Learning rate
        """
        db, dw = self.compute_gradients(x, t)

        # --- TO-DO block: Update the model parameters b and w -- DONE
        # Gradient descent: move AGAINST the gradient (minus sign), step size eta.
        # [BROADCAST] scalar eta times every entry, [ELEMENT-WISE] subtraction.
        # Shapes match exactly: (1,1)-(1,1) and (d,1)-(d,1).
        # Rebinding with "=" is fine in NumPy (in TF we must use assign, see Ex 4-5).
        self.b = self.b - eta * db
        self.w = self.w - eta * dw                
        # --- End of TO-DO block

    def fit(self, x, t, eta, num_iters):
        """
        Trains the model for a fixed number of iterations

        Parameters
        ----------
        x : array
            Input batch of shape (N, d), where N is the number of patterns and
            d is the dimension
        t : array
            Array of shape (N, 1) with the target values for each x
        eta : float
            Learning rate
        num_iters : int
            Number of training iterations

        Returns
        -------
        loss : array
             Array of shape (num_iters,) with the loss after each training
             iteration
        """
        # Full-batch gradient descent: every iteration uses ALL the data (no mini-batches).
        loss = np.zeros(num_iters)
        for i in range(num_iters):
            self.gradient_step(x, t, eta)
            loss[i] = self.get_loss(x, t)
        return loss

    def get_loss(self, x, t):
        """
        Calculates the MSE loss for an input batch (x, t)

        Parameters
        ----------
        x : array
            Input batch of shape (N, d), where N is the number of patterns and
            d is the dimension
        t : array
            Array of shape (N, 1) with the target values for each x

        Returns
        -------
        loss : float
             MSE loss
        """
        y = self.predict(x)
        # [ELEMENT-WISE] square (** is NOT a matrix power), then [REDUCTION] mean over
        # everything -> a scalar. No axis, so keepdims is irrelevant here.
        loss = 0.5 * np.mean((y - t) ** 2)
        return loss


# ===============================================================================
# EXERCISE 2 - Logistic regression (binary classification)
#
#   Model:  y = sigmoid(x·w + b)          output in (0, 1) = probability of class 1
#   Loss:   cross-entropy  L = -(1/N) sum [ t log y + (1-t) log(1-y) ]
#
# Q: Why do we only override predict (and get_loss) and inherit compute_gradients?
#    With sigmoid + cross-entropy, dL/dz = (y - t)/N, EXACTLY the same expression as
#    linear regression + MSE:
#       dL/dy = (1/N)(y - t) / (y(1-y))    and    sigmoid'(z) = y(1-y)
#       dL/dz = dL/dy * sigmoid'(z) = (y - t)/N      <- the y(1-y) cancels
#    So compute_gradients, gradient_step and fit from the parent class are still correct;
#    compute_gradients calls self.predict, which now includes the sigmoid.
# ===============================================================================
class LogisticRegressionModel(LinearRegressionModel):
    """
    Logistic regression model for exercise 2

    Parameters
    ----------
    d : int
        Dimension

    Attributes
    ----------
    w : array
        Weight vector of shape (d, 1)
    b : array
        Bias term of shape (1, 1)
    """

    def __init__(self, d=2):
        # Same parameters as linear regression: reuse the parent constructor.
        LinearRegressionModel.__init__(self, d)

    # Q: This method has no "self" and no @staticmethod. Does it work?
    #    Yes, but ONLY when called through the class: LogisticRegressionModel.sigmoid(z).
    #    In Python 3 a function stored in a class is a plain function when accessed via the
    #    class. self.sigmoid(z) would FAIL: Python would pass self as z plus our z ->
    #    TypeError (2 positional arguments given, 1 expected).
    def sigmoid(z):
        """
        Calculates the sigmoid function on input z, element-wise

        Parameters
        ----------
        z : array or float
            Input value or values, of arbitrary shape

        Returns
        -------
        _ : array or float
            Sigmoid function evaluated on z, same shape as z
        """
        # [ELEMENT-WISE] np.exp acts on every entry; output has the same shape as z.
        return 1.0 / (1.0 + np.exp(-z))

    # --- TO-DO block: Overwrite the methods of the LinearRegressionModel class
    def predict(self, x):
        # Same linear part as Ex 1: [MATMUL] (N,d)·(d,1)=(N,1) + [BROADCAST] bias
        z = np.dot(x, self.w) + self.b
        # [ELEMENT-WISE] sigmoid squashes each z_n into (0, 1) -> probability, (N, 1)
        return LogisticRegressionModel.sigmoid(z)
    # --- End of TO-DO block

    def get_loss(self, x, t):
        """
        Calculates the cross-entropy loss for an input batch (x, t)

        Parameters
        ----------
        x : array
            Input batch of shape (N, d), where N is the number of patterns and
            d is the dimension
        t : array
            Array of shape (N, 1) with the target labels for each x

        Returns
        -------
        loss : float
             Cross-entropy loss
        """
        y = self.predict(x)
        # All ELEMENT-WISE (t * log y is NOT a matrix product), then [REDUCTION] mean -> scalar.
        # Q: Numerical risk? If y becomes exactly 0 or 1, log(0) = -inf -> loss = nan.
        #    Real frameworks clip y or compute the loss directly from the logits z.
        loss = -np.mean(t * np.log(y) + (1.0 - t) * np.log(1.0 - y))
        return loss


# ===============================================================================
# EXERCISES 3 and 4 - Automatic differentiation with tf.GradientTape
#
# Key ideas:
#   - tf.Variable = mutable tensor (parameters). Trainable Variables are WATCHED
#     automatically by the tape. A plain tf.Tensor/constant needs tape.watch(t).
#   - Inside "with tf.GradientTape() as tape:" every op is RECORDED.
#   - tape.gradient(target, source) replays the recording backwards (reverse-mode autodiff).
#   - A normal (non-persistent) tape can only call gradient() ONCE.
# ===============================================================================
class BasicTF:
    """
    Static methods for exercises 3 and 4
    """

    @staticmethod
    def differentiate(f, x):
        """
        Calculates the derivative of the funcion f for each point in x

        Parameters
        ----------
        f : function
            A function of one variable with auto-differentiation, such as tf.cos
            or tf.exp
        x : array
            Input array of arbitrary shape with the points where the derivative of
            the function f must be evaluated

        Returns
        -------
        dy_dx : array
            Array with the same shape as x with the derivative
        """
        # Convert to a Variable so the tape watches it automatically.
        # Must be a FLOAT array: integer variables are not differentiable (gradient = None).
        x = tf.Variable(x)

        # --- TO-DO block: Define the computational graph within a gradient tape and
        # --- compute the gradient
        with tf.GradientTape() as tape:
            y = f(x)   # forward pass is recorded; f is element-wise (tf.cos, tf.exp, ...)

        # Q: y is an ARRAY, not a scalar. What does tape.gradient(y, x) return?
        #    When the target is not a scalar, TF differentiates sum(y).
        #    d(sum_j y_j)/dx_i = dy_i/dx_i = f'(x_i), because f is element-wise
        #    (y_i only depends on x_i). So we get the derivative at EVERY point,
        #    with the same shape as x (a gradient always has the shape of its source).
        # .numpy() converts the resulting EagerTensor into a NumPy array.
        dy_dx = tape.gradient(y, x).numpy()

        # --- End of TO-DO block

        return dy_dx

    @staticmethod
    def gradient_descent(f, x0, eta, niters):
        """
        Minimizes a function using gradient descent

        Parameters
        ----------
        f : function
            A function of one variable x
        x : float
            The initial value of x
        eta : float
            The learning rate
        niters : int
            The number of iterations

        Returns
        -------
        xvals : array
            An array with shape (niters,) with the value of x after each iteration
        """
        x = tf.Variable(x0)   # x0 must be a float (e.g. 2.0, not 2)

        x_history = []
        for i in range(niters):
            # --- TO-DO block: Define the computational graph within a gradient tape and
            # --- compute the gradient
            
            # Q: Why create a NEW tape inside the loop?
            #    A non-persistent tape is released after one gradient() call, and each
            #    iteration evaluates f at a different x, so a new recording is needed.
            with tf.GradientTape() as tape:
                y = f(x)
            dy_dx = tape.gradient(y, x)   # f'(x) at the current point
            # --- End of TO-DO block

            # --- TO-DO block: Update the value of x using the tf.Variable assign method
            # x <- x - eta * f'(x), done IN PLACE.
            # Q: Why not "x = x - eta * dy_dx"?
            #    That creates a plain tf.Tensor and rebinds the name x to it. x would no
            #    longer be a Variable, the tape would not watch it in the next iteration,
            #    and tape.gradient would return None -> crash. assign keeps it a Variable.
            #    (x.assign_sub(eta * dy_dx) would be equivalent.)
            x.assign(x - eta * dy_dx)
            # --- End of TO-DO block

            x_history.append(x.numpy())   # store a NumPy copy of the current value

        xvals = np.array(x_history)

        return xvals


# ===============================================================================
# EXERCISE 5 - Linear regression again, but with TensorFlow
#   Same model and shapes as Ex 1, but the gradients come from the GradientTape
#   instead of being derived by hand. Result should match Ex 1.
# ===============================================================================
class LinearRegressionModel_TF(object):
    """
    Linear regression model using TensorFlow for exercise 5

    Parameters
    ----------
    d : int
        Dimension

    Attributes
    ----------
    w : tf.Variable
        Weight vector of shape (d, 1)
    b : tf.Variable
        Bias term of shape (1, 1)
    """

    def __init__(self, d=2):
        # Q: Why dtype=float64?
        #    tf.random.normal is float32 by default, but our NumPy data is float64.
        #    TF does NOT auto-cast: tf.matmul(float64, float32) raises an error.
        self.w = tf.Variable(tf.random.normal(shape=[d, 1], dtype=tf.dtypes.float64))
        self.b = tf.Variable(tf.random.normal(shape=[1, 1], dtype=tf.dtypes.float64))

    def predict(self, x):
        """
        Predicts output y for input batch x

        Parameters
        ----------
        x : array or tensor
            Input batch of shape (N, d), where N is the number of patterns and
            d is the dimension

        Returns
        -------
        y : tensor
            Ouput batch of shape (N, 1) which is the result of applying the
            linear regression model to the input x

        """
        # --- TO-DO block: Compute the model output y
        # [MATMUL] (N,d)·(d,1)=(N,1) + [BROADCAST] bias. Same as Ex 1 with tf.matmul.
        # tf.matmul accepts a NumPy x directly and converts it to a tensor.
        y = tf.matmul(x, self.w) + self.b
        # --- End of TO-DO block

        return y

    def compute_gradients(self, x, t):
        """
        Calculates the gradients of the loss function with respect to the model
        parameters b and w, for an input batch (x, t)

        Parameters
        ----------
        x : array or tensor
            Input batch of shape (N, d), where N is the number of patterns and
            d is the dimension
        t : array or tensor
            Array of shape (N, 1) with the target values for each x

        Returns
        -------
        db : tensor
             Gradient of the loss with respect to the bias, shape (1, 1)
        dw : tensor
             Gradient of the loss with respect to the weights, shape (d, 1)
        """
        # --- TO-DO block: Compute the gradients db and dw of the loss function
        # The whole forward pass (predict + loss) must happen INSIDE the tape,
        # otherwise nothing is recorded and the gradients are None.
        # self.w and self.b are trainable Variables -> watched automatically.
        with tf.GradientTape() as tape:
            loss = self.get_loss(x, t)

        # Sources given as a LIST -> gradients returned as a list IN THE SAME ORDER,
        # so we can unpack them directly: first b, then w.
        # Each gradient has the shape of its Variable: db (1,1), dw (d,1).
        # Q: Do we need keepdims here? No: the tape already returns the correct shapes.
        db, dw = tape.gradient(loss, [self.b, self.w])
    
        # --- End of TO-DO block

        return db, dw

    def gradient_step(self, x, t, eta):
        """
        Updates the model parameters with an input batch (x, t)

        Parameters
        ----------
        x : array or tensor
            Input batch of shape (N, d), where N is the number of patterns and
            d is the dimension
        t : array or tensor
            Array of shape (N, 1) with the target values for each x
        eta : float
            Learning rate
        """
        db, dw = self.compute_gradients(x, t)

        # --- TO-DO block: Update the model parameters b and w
        # assign_sub(v) means "self.b = self.b - v" IN PLACE, keeping it a tf.Variable.
        # (Using "=" would turn it into a Tensor and break the tape next iteration.)
        self.b.assign_sub(eta * db)
        self.w.assign_sub(eta * dw)
        # --- End of TO-DO block

    def fit(self, x, t, eta, num_iters):
        """
        Trains the model for a fixed number of iterations

        Parameters
        ----------
        x : array or tensor
            Input batch of shape (N, d), where N is the number of patterns and
            d is the dimension
        t : array or tensor
            Array of shape (N, 1) with the target values for each x
        eta : float
            Learning rate
        num_iters : int
            Number of training iterations

        Returns
        -------
        loss : array
             Array of shape (num_iters,) with the loss after each training
             iteration
        """
        loss = np.zeros(num_iters)
        for i in range(num_iters):
            self.gradient_step(x, t, eta)
            # get_loss returns a scalar Tensor -> .numpy() to store it in a NumPy array
            loss[i] = self.get_loss(x, t).numpy()
        return loss

    def get_loss(self, x, t):
        """
        Calculates the MSE loss for an input batch (x, t)

        Parameters
        ----------
        x : array or tensor
            Input batch of shape (N, d), where N is the number of patterns and
            d is the dimension
        t : array or tensor
            Array of shape (N, 1) with the target values for each x

        Returns
        -------
        loss : float
             MSE loss
        """
        y = self.predict(x)
        # (y - t) * (y - t) is [ELEMENT-WISE] (a square), NOT a matrix product.
        # tf.reduce_mean = np.mean -> [REDUCTION] to a scalar.
        loss = tf.reduce_mean(0.5 * (y - t) * (y - t))
        return loss


# ===============================================================================
# EXERCISE 6 - Dense feed-forward neural network with MANUAL backpropagation (NumPy)
#
# LAYOUT CHANGE: samples are now COLUMNS.
#   x (d, N),  W[l] (n_l, n_{l-1}),  b[l] (n_l, 1),  z[l], y[l] (n_l, N),  t (1, N)
#
# Backprop equations:
#   delta_L     = dL/dz_L = (y_L - t)/N
#   dW_l        = delta_l · y_{l-1}^T                      [MATMUL]
#   db_l        = sum over samples of delta_l              [REDUCTION] axis=1
#   delta_{l-1} = (W_l^T · delta_l) * a'_{l-1}(z_{l-1})    [MATMUL] then [ELEMENT-WISE]
#
# Q: Why delta·y_prev^T here but x^T·delta in Ex 1?
#    Same math (sum over samples of input x error), transposed layout. The order is
#    whatever gives the shape of the parameter: (n_l,N)·(N,n_{l-1}) = (n_l,n_{l-1}) = W[l].
# ===============================================================================
class NeuralNetwork(object):
    """
    Dense feedforward neural network model for exercise 6

    Parameters
    ----------
    d : list of tuples
        List of tuples that define the neural network architecture. There is one
        tuple of the form (n, a) for each layer, where n is the number of layer
        neurons and a is the type of layer. The following types are allowed:

        - 'sigmoid': a layer of sigmoid units
        - 'linear': a layer of linear units
        - 'input': a special layer for the input

        The minimum is two layers (input and output).
        The first layer must be always of type 'input'.
        Only one 'input' layer must be present, and it must be the first layer.
        The default value is layers=[(2, 'input'), (1, 'sigmoid')].
        The output layer must have only one neuron.

    Attributes
    ----------
    nlayers : int
        Number of layers, excluding the input layer
    W : list of arrays
        List containing the weight matrix for each processing layer
    b : list of arrays
        List containing the bias vector for each processing layer
    a : list of functions
        List containing the activation function for each processing layer
    da : list of functions
        List containing the derivative of the activation function for each
        processing layer
    """

    def __init__(self, layers=[(2, "input"), (1, "sigmoid")]):
        self.nlayers = len(layers) - 1   # the input "layer" has no parameters
        self.W = []
        self.b = []
        self.a = []
        self.da = []

        # Set the weights, biases and activation functions:
        # zip(layers[:-1], layers[1:]) walks consecutive pairs (previous layer, current layer).
        for l0, l1 in zip(layers[:-1], layers[1:]):
            # W: (neurons of this layer, neurons of previous layer) so that W·y_prev works
            self.W.append(np.random.randn(l1[0], l0[0]))
            # b: one bias per neuron, a COLUMN (n_l, 1) -> broadcasts over the N columns
            self.b.append(np.random.randn(l1[0], 1))
            # store the activation AND its derivative (needed for backprop)
            if l1[1] == "sigmoid":
                self.a.append(NeuralNetwork.sigmoid)
                self.da.append(NeuralNetwork.dsigmoid)
            elif l1[1] == "linear":
                self.a.append(NeuralNetwork.identity)
                self.da.append(NeuralNetwork.didentity)

    @staticmethod
    def sigmoid(z):
        """
        Sigmoid function
        """
        return 1.0 / (1.0 + np.exp(-z))   # [ELEMENT-WISE]

    @staticmethod
    def dsigmoid(z):
        """
        Derivative of the sigmoid function
        """
        # sigma'(z) = sigma(z) * (1 - sigma(z)), [ELEMENT-WISE] product
        return NeuralNetwork.sigmoid(z) * (1.0 - NeuralNetwork.sigmoid(z))

    @staticmethod
    def identity(z):
        """
        Identity function
        """
        return z

    @staticmethod
    def didentity(z):
        """
        Derivative of the identity function
        """
        # derivative of z is 1 everywhere; ones_like keeps the shape of z
        return np.ones_like(z)

    @staticmethod
    def mse_loss(y, t):
        """
        MSE loss
        """
        loss = 0.5 * np.mean((y - t) ** 2)
        return loss

    @staticmethod
    def cross_entropy_loss(y, t):
        """
        Cross-entropy loss
        """
        loss = -np.mean(t * np.log(y) + (1.0 - t) * np.log(1.0 - y))
        return loss

    def predict(self, x):
        """
        Predicts (forward pass) output y for input batch x

        Parameters
        ----------
        x : array
            Input batch of shape (d, N), where N is the number of patterns and
            d is the dimension

        Returns
        -------
        z : list of arrays
            List containing the preactivations of all layers for input batch x.
            z[i] is an array of shape (ni, N), with ni the number of neurons in
            layer i and N the number of patterns
        y : list of arrays
            List containing the activations of all layers for input batch x.
            y[i] is an array of shape (ni, N), with ni the number of neurons in
            layer i and N the number of patterns
        """
        z = []
        y = []
        # --- TO-DO block: loop in the network layers computing both the pre-
        # --- activation and the activation and appending them to lists z and y.
        yl = x   # the "activation" feeding the first layer is the input itself
        for l in range(self.nlayers):
            # [MATMUL]    (n_l, n_{l-1})·(n_{l-1}, N) = (n_l, N)
            # [BROADCAST] b[l] (n_l, 1) is copied across the N columns (samples)
            zl = np.dot(self.W[l], yl) + self.b[l]
            # [ELEMENT-WISE] activation on every neuron and every sample
            yl = self.a[l](zl)
            # Q: Why store z and y for every layer?
            #    Backprop needs them: y[l-1] for dW[l], and z[l-1] for the activation
            #    derivative da(z). NOTE: the list y does NOT include the input x.
            z.append(zl)
            y.append(yl)
        # --- End of TO-DO block

        return z, y

    def compute_gradients(self, x, t):
        """
        Implements the backward pass, calculating the gradients of the loss
        function with respect to all the weights and biases for an input batch
        (x, t)

        Parameters
        ----------
        x : array
            Input batch of shape (d, N), where N is the number of patterns and
            d is the dimension
        t : array
            Array of shape (1, N) with the target values for each x

        Returns
        -------
        db : list of arrays
            List containing the gradients of the loss function with respect to
            the biases of each layer, for input batch x.
        dW : list of arrays
            List containing the gradients of the loss function with respect to
            the weights of each layer, for input batch x.
        """
        # N = number of samples. shape[1] because samples are COLUMNS here (Ex 1 used shape[0]).
        n = x.shape[1]

        dW = []
        db = []

        # Call the predict method (forward pass). The preactivations z and the
        # activations y in each layer are needed for the backward pass:
        z, y = self.predict(x)

        # Derivative of the loss with respect to z in the last layer. This is
        # valid both for a regression problem with linear output and MSE loss,
        # and for a classification problem with sigmoid output and cross-entropy
        # loss:
        # Q: The variable is called dy, but what is it really?
        #    dL/dz of the OUTPUT layer (not dL/dy). Same cancellation as in Ex 2:
        #    linear+MSE and sigmoid+CE both give (y - t)/N. Shape (1, N).
        dy = (y[-1] - t) / n

        # --- TO-DO block: loop in the network layers computing the gradients with
        # --- respect to W and b. Note that the gradients must be computed starting
        # --- by the last layer, it may be useful to traverse the lists backwards.
        delta = dy   # delta = dL/dz of the current layer, starts at the output layer
        # range(nlayers-1, -1, -1) -> L-1, L-2, ..., 0   (backwards: that's BACKprop)
        for l in range(self.nlayers - 1, -1, -1):
            # Input that fed layer l. y has no entry for x, so y[l-1] is the input of layer l.
            # Q: Why the "else x"?
            #    For l = 0, y[l-1] = y[-1] would wrap around to the OUTPUT layer (Python
            #    negative indexing) -> wrong. The first layer's input is x itself.
            y_prev = y[l-1] if l > 0 else x

            # [MATMUL] (n_l, N)·(N, n_{l-1}) = (n_l, n_{l-1}) = shape of W[l]
            # The product sums over the N samples: dW = sum_n delta_n * y_prev_n^T
            # insert(0, ...) because we go backwards: at the end the list is in forward order.
            dW.insert(0, np.dot(delta, y_prev.T))

            # [REDUCTION] sum over samples = over COLUMNS -> axis=1 -> (n_l, 1) = shape of b[l]
            # Q: What if keepdims=False?  db would be (n_l,) (1-D). Then b - eta*db would
            #    broadcast (n_l,1) - (1,n_l) -> (n_l, n_l): the bias silently becomes a
            #    matrix. It only shows up in hidden layers with more than one neuron.
            db.insert(0, np.sum(delta, axis=1, keepdims=True))

            # Propagate the error one layer back (not needed after the first layer).
            if l > 0:
                # [MATMUL] W[l]^T (n_{l-1}, n_l) · delta (n_l, N) = (n_{l-1}, N):
                #          sends the error back through the weights
                # [ELEMENT-WISE] * da(z) of the PREVIOUS layer (l-1): chain rule through
                #          its activation, per neuron and per sample
                # Q: Why W[l] but da[l-1]? We cross layer l's weights backwards, then
                #    layer l-1's activation. Both are needed to reach dL/dz_{l-1}.
                delta = np.dot(self.W[l].T, delta) * self.da[l-1](z[l-1])
        # --- End of TO-DO block

        # NOTE: the docstring says (db, dW), but we return (dW, db).
        # It is consistent because gradient_step unpacks "dW, db".
        return dW, db

    def gradient_step(self, x, t, eta):
        """
        Updates the model parameters with an input batch (x, t)

        Parameters
        ----------
        x : array
            Input batch of shape (d, N), where N is the number of patterns and
            d is the dimension
        t : array
            Array of shape (1, N) with the target values for each x
        eta : float
            Learning rate
        """
        dW, db = self.compute_gradients(x, t)

        # --- TO-DO block: Loop in layers updating the model parameters b and w
        # Gradient descent on every layer. Shapes match exactly thanks to keepdims
        # (golden rule): W[l] and dW[l] (n_l, n_{l-1}), b[l] and db[l] (n_l, 1).
        for l in range(self.nlayers):
            self.W[l] = self.W[l] - (eta * dW[l])
            self.b[l] = self.b[l] - (eta * db[l])
        # --- End of TO-DO block

    def fit(self, x, t, eta, num_epochs, batch_size, loss_function):
        """
        Trains the model for a fixed number of epochs

        Parameters
        ----------
        x : array
            Input batch of shape (d, N), where N is the number of patterns and
            d is the dimension
        t : array
            Array of shape (1, N) with the target values for each x
        eta : float
            Learning rate
        num_epochs : int
            Number of training epochs
        batch_size : int
            Size of mini-batches
        loss_function : function
            Loss function, either one of mse_loss or cross_entropy_loss

        Returns
        -------
        loss : array
             Array of shape (num_epochs,) with the loss after each training
             epoch
        """
        dim, n = x.shape
        # ceil(n / batch_size): the extra +1 (True) adds a last, smaller batch if n is
        # not a multiple of batch_size.
        num_batches = (n // batch_size) + ((n % batch_size) != 0)

        loss = np.zeros(num_epochs)
        # Q: Epoch vs iteration? An epoch = one pass over ALL the data = num_batches updates.
        for i in range(num_epochs):
            # Shuffle data and generate batches:
            # Q: Why shuffle every epoch? So the mini-batches differ each epoch and the
            #    order of the data does not bias the updates.
            ix = np.random.permutation(n)
            for j in range(num_batches):
                imin = j * batch_size
                imax = np.minimum((j + 1) * batch_size, n)   # last batch may be smaller

                ibatch = ix[imin:imax]
                # x[:, ibatch] selects COLUMNS (samples are columns in this layout)
                batch_x = x[:, ibatch]
                batch_t = t[:, ibatch]
                # Mini-batch gradient descent: one update per batch
                self.gradient_step(batch_x, batch_t, eta)

            # At the end of each epoch, compute the model loss on all data:
            loss[i] = self.get_loss(x, t, loss_function)
        return loss

    def get_loss(self, x, t, loss_function):
        """
        Calculates the loss for an input batch (x, t)

        Parameters
        ----------
        x : array
            Input batch of shape (d, N), where N is the number of patterns and
            d is the dimension
        t : array
            Array of shape (1, N) with the target values for each x
        loss_function : function
            Loss function, either one of mse_loss or cross_entropy_loss

        Returns
        -------
        loss : float
             Calculated loss
        """
        _, y = self.predict(x)            # we only need the activations
        return loss_function(y[-1], t)    # y[-1] = output of the last layer, (1, N)


# ===============================================================================
# EXERCISE 7 - Same neural network, but gradients from the GradientTape
#   No hand-written backprop, no need to store z and y: the tape records the forward
#   pass and computes all the gradients.
# ===============================================================================
class NeuralNetwork_TF(object):
    """
    Dense feedforward neural network model for exercise 7, using TensorFlow

    Parameters
    ----------
    d : list of tuples
        List of tuples that define the neural network architecture. There is one
        tuple of the form (n, a) for each layer, where n is the number of layer
        neurons and a is the activation. Any activation function in TensroFlow is
        valid. In particular:

        - tf.sigmoid: a layer of sigmoid units
        - tf.identity: a layer of linear units

        The activation function is ignored for the input layer.
        The minimum is two layers (input and output).
        The default value is layers=[(2, None), (1, tf.sigmoid)].
        The output layer must have only one neuron.

    Attributes
    ----------
    nlayers : int
        Number of layers, excluding the input layer
    W : list of tensors
        List containing the weight matrix for each processing layer
    b : list of tensors
        List containing the bias vector for each processing layer
    a : list of functions
        List containing the activation function for each processing layer
    """

    def __init__(self, layers=[(2, None), (1, tf.sigmoid)]):
        self.nlayers = len(layers) - 1
        self.W = []
        self.b = []
        self.a = []
        for l0, l1 in zip(layers[:-1], layers[1:]):
            # Same shapes as Ex 6, but as tf.Variables (float64 to match the NumPy data).
            self.W.append(
                tf.Variable(
                    tf.random.normal(shape=[l1[0], l0[0]], dtype=tf.dtypes.float64)
                )
            )
            self.b.append(
                tf.Variable(tf.random.normal(shape=[l1[0], 1], dtype=tf.dtypes.float64))
            )
            # Only the activation: no derivative needed, the tape differentiates it.
            self.a.append(l1[1])

    @staticmethod
    def mse_loss(y, t):
        """
        MSE loss
        """
        # tf.pow is [ELEMENT-WISE]; tf.reduce_mean is the [REDUCTION] to a scalar.
        loss = 0.5 * tf.reduce_mean(tf.pow(y - t, 2.0))
        return loss

    @staticmethod
    def cross_entropy_loss(y, t):
        """
        Cross-entropy loss
        """
        # TF version of the same formula: tf.math.log instead of np.log.
        loss = -tf.reduce_mean(t * tf.math.log(y) + (1.0 - t) * tf.math.log(1.0 - y))
        return loss

    def predict(self, x):
        """
        Predicts (forward pass) output y for input batch x

        Parameters
        ----------
        x : tensor
            Input batch of shape (d, N), where N is the number of patterns and
            d is the dimension

        Returns
        -------
        y : tensor
            Tensor of shape (1, N) with the activation in the last layer
        """
        # --- TO-DO block: loop in the network layers computing the activations.
        # --- The activation of the last layer should be stored at variable y to
        # --- be returned.
        # Q: Why don't we store z and y in lists like in Ex 6?
        #    Because we don't write the backward pass: the tape records every
        #    intermediate value it needs by itself.
        y = x
        for l in range(self.nlayers):
            # [MATMUL] (n_l, n_{l-1})·(n_{l-1}, N) = (n_l, N)  + [BROADCAST] b (n_l, 1)
            z = tf.matmul(self.W[l], y) + self.b[l]
            # [ELEMENT-WISE] activation (tf.sigmoid, tf.identity, ...)
            y = self.a[l](z)
        # --- End of TO-DO block

        return y

    def compute_gradients(self, x, t, loss_function):
        """
        Implements the backward pass, calculating the gradients of the loss
        function with respect to all the weights and biases for an input batch
        (x, t)

        Parameters
        ----------
        x : tensor
            Input batch of shape (d, N), where N is the number of patterns and
            d is the dimension
        t : tensor
            Tensor of shape (1, N) with the target values for each x
        loss_function : function
            Loss function, either one of mse_loss or cross_entropy_loss

        Returns
        -------
        db : list of tensors
            List containing the gradients of the loss function with respect to
            the biases of each layer, for input batch x.
        dW : list of tensors
            List containing the gradients of the loss function with respect to
            the weights of each layer, for input batch x.
        """
        # --- TO-DO block: compute the gradients db, dW using the gradient tape
        # Forward pass + loss INSIDE the tape so it is recorded.
        with tf.GradientTape() as tape:
            loss = self.get_loss(x, t, loss_function)

        # Q: What is self.b + self.W?
        #    PYTHON LIST CONCATENATION, not tensor addition:
        #    [b[0], ..., b[L-1], W[0], ..., W[L-1]]  -> a list of 2L Variables.
        #    tape.gradient returns the gradients in exactly that same order.
        grads = tape.gradient(loss, self.b + self.W)

        # So we slice the list back into its two halves:
        db = grads[: self.nlayers]  # first L entries -> gradients w.r.t. the biases
        dW = grads[self.nlayers :]  # last  L entries -> gradients w.r.t. the weights
        # Each gradient already has the shape of its Variable (no keepdims needed).
        # --- End of TO-DO block

        # NOTE: here we return (db, dW), the OPPOSITE order to Ex 6 (dW, db).
        # gradient_step below unpacks "dB, dW", so it is consistent.
        return db, dW

    # ---------------------------------------------------------------------------
    # Gradient step:
    # ---------------------------------------------------------------------------
    def gradient_step(self, x, t, eta, loss_function):
        """
        Updates the model parameters with an input batch (x, t)

        Parameters
        ----------
        x : tensor
            Input batch of shape (d, N), where N is the number of patterns and
            d is the dimension
        t : tensor
            Tensor of shape (1, N) with the target values for each x
        eta : float
            Learning rate
        loss_function : function
            Loss function, either one of mse_loss or cross_entropy_loss
        """
        dB, dW = self.compute_gradients(x, t, loss_function)

        # --- TO-DO block: Loop in layers updating the model parameters b and w
        # In-place updates so they stay tf.Variables (see Ex 4 for why not "=").
        for l in range(self.nlayers):
            self.b[l].assign_sub(eta * dB[l])
            self.W[l].assign_sub(eta * dW[l])
        # --- End of TO-DO block

    def fit(self, x, t, eta, num_epochs, batch_size, loss_function):
        """
        Trains the model for a fixed number of epochs

        Parameters
        ----------
        x : array
            Input batch of shape (d, N), where N is the number of patterns and
            d is the dimension
        t : array
            Array of shape (1, N) with the target values for each x
        eta : float
            Learning rate
        num_epochs : int
            Number of training epochs
        batch_size : int
            Size of mini-batches
        loss_function : function
            Loss function, either one of mse_loss or cross_entropy_loss

        Returns
        -------
        loss : array
             Array of shape (num_epochs,) with the loss after each training
             epoch
        """
        # Identical mini-batch loop to Ex 6 (shuffle, split into columns, one step per batch).
        dim, n = x.shape
        num_batches = (n // batch_size) + ((n % batch_size) != 0)

        loss = np.zeros(num_epochs)
        for i in range(num_epochs):
            # Shuffle data and generate batches:
            ix = np.random.permutation(n)
            for j in range(num_batches):
                imin = j * batch_size
                imax = np.minimum((j + 1) * batch_size, n)

                ibatch = ix[imin:imax]
                batch_x = x[:, ibatch]
                batch_t = t[:, ibatch]
                self.gradient_step(batch_x, batch_t, eta, loss_function)

            # Calculo el loss de la epoca con todos los datos:
            # (the loss is a scalar Tensor -> .numpy() to store it)
            loss[i] = self.get_loss(x, t, loss_function).numpy()
        return loss

    def get_loss(self, x, t, loss_function):
        """
        Calculates the loss for an input batch (x, t)

        Parameters
        ----------
        x : array
            Input batch of shape (d, N), where N is the number of patterns and
            d is the dimension
        t : array
            Array of shape (1, N) with the target values for each x
        loss_function : function
            Loss function, either one of mse_loss or cross_entropy_loss

        Returns
        -------
        loss : float
             Calculated loss
        """
        y = self.predict(x)          # only the final output here (no lists)
        return loss_function(y, t)