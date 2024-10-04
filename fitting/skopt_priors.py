import numpy as np
from skopt.space import Dimension, Identity, Pipeline, Normalize
from scipy.stats import expon, beta
from skopt.space.space import _uniform_inclusive, Real


class Exponential(Dimension):
    """Search space dimension that takes on values from an exponential distribution."""

    def __init__(self, scale,  name=None, dtype=float):
        self.scale = scale
        self.name = name
        self.dtype = dtype
        self._rvs = expon(scale=self.scale)
        self.transformer = Identity()
        self.transform_ = "identity"

    def set_transformer(self, transform="identity"):
        """Define rvs and transformer spaces."""
        self.transform_ = "identity"

        if self.transform_ not in ["identity"]:
            raise ValueError(f"transform should be 'normalize' or 'identity', got {self.transform_}")

        self._rvs = expon(scale=self.scale)
        self.transformer = Identity()

    def inverse_transform(self, Xt):
        """Inverse transform samples from the warped space back into the original space."""
        inv_transform = np.clip(self.transformer.inverse_transform(Xt), 1e-6, np.inf).astype(self.dtype)
        return inv_transform

    @property
    def transformed_bounds(self):
           return 1e-6, np.inf

    @property
    def bounds(self):
        return (1e-6, np.inf)

    @property
    def is_constant(self):
        return False

    def __repr__(self):
        return f"Exponential(scale={self.scale}, transform='{self.transform_}')"

    def __contains__(self, point):
        if isinstance(point, list):
            point = np.array(point)
        return 1e-6 <= point

    def distance(self, a, b):
        """Compute distance between point `a` and `b`.

        Parameters
        ----------
        a : float
            First point.

        b : float
            Second point.
        """
        if not (a in self and b in self):
            raise RuntimeError("Can only compute distance for values within "
                               "the space, not %s and %s." % (a, b))
        return abs(a - b)


class Beta(Dimension):
    """Search space dimension that takes on values from a scaled beta distribution."""

    def __init__(self, alpha, beta_param, scale=1.0, name=None, dtype=float):
        self.alpha = alpha
        self.beta_param = beta_param
        self.scale = scale
        self.name = name
        self.dtype = dtype
        self._rvs = beta(self.alpha, self.beta_param)
        self.transformer = Identity()
        self.transform_ = "identity"

    def set_transformer(self, transform="identity"):
        # """Define rvs and transformer spaces."""
        # self.transform_ = "identity"
        #
        # if self.transform_ not in ["identity"]:
        #     raise ValueError(f"transform should be 'normalize' or 'identity', got {self.transform_}")
        #
        # self._rvs = beta(self.alpha, self.beta_param)
        # self.transformer = Identity()

        self.transform_ = transform

        if self.transform_ not in ["normalize", "identity"]:
            raise ValueError("transform should be 'normalize' or 'identity'"
                             " got {}".format(self.transform_))

        # XXX: The _rvs is for sampling in the transformed space.
        # The rvs on Dimension calls inverse_transform on the points sampled
        # using _rvs
        if self.transform_ == "normalize":
            # set upper bound to next float after 1. to make the numbers
            # inclusive of upper edge
            self.transformer = Pipeline([Identity(), Normalize(0, self.scale)])

        else:
            self.transformer = Identity()


    def inverse_transform(self, Xt):
        """Inverse transform samples from the warped space back into the original space."""
        inv_transform = np.clip(self.transformer.inverse_transform(Xt), 1e-6, self.scale).astype(self.dtype)
        return inv_transform

    @property
    def transformed_bounds(self):
        return 1e-6, 1

    @property
    def bounds(self):
        return (1e-6, self.scale)

    @property
    def is_constant(self):
        return False

    def __repr__(self):
        return f"Beta(alpha={self.alpha}, beta={self.beta_param}, scale={self.scale}, transform='{self.transform_}')"

    def __contains__(self, point):
        if isinstance(point, list):
            point = np.array(point)
        return 1e-6 <= point <= self.scale

    def distance(self, a, b):
        """Compute distance between point `a` and `b`.

        Parameters
        ----------
        a : float
            First point.

        b : float
            Second point.
        """
        if not (a in self and b in self):
            raise RuntimeError("Can only compute distance for values within "
                               "the space, not %s and %s." % (a, b))
        return abs(a - b)

from skopt.space.transformers import Transformer

class GeneralizedLogitTransform(Transformer):
    """Transformer that applies the generalized logit (inverse softmax) transform."""

    def fit(self, X):
        return self

    def transform(self, X):
        # Clip X to avoid log of zero
        X = np.clip(X, 1e-15, 1 - 1e-15)
        # Number of components
        K = X.shape[-1]
        # Compute log ratios for the first K-1 components
        Y_partial = np.log(X[:, :-1] / X[:, -1:])
        # Append an additional variable (e.g., log(X_last / (1 - X_last)))
        # However, this creates redundancy. Alternatively, set it to zero.
        # Here, we'll set it to zero to maintain K variables.
        Y_extra = np.zeros((X.shape[0], 1))
        # Concatenate to have K transformed variables
        Y = np.hstack([Y_partial, Y_extra])
        return Y

    def inverse_transform(self, Y):
        # Separate the partial transformed variables and the extra variable
        Y_partial = Y[:, :-1]
        # Reconstruct the probabilities using the inverse transformation
        exp_Y = np.exp(Y_partial)
        sum_exp_Y = np.sum(exp_Y, axis=-1, keepdims=True)
        X_rest = exp_Y / (1 + sum_exp_Y)
        X_last = 1 / (1 + sum_exp_Y)
        # Concatenate to get the full K components
        X = np.hstack([X_rest, X_last])
        return X


from scipy.stats import dirichlet
class Dirichlet(Dimension):
    """Search space dimension that takes on values from a Dirichlet distribution."""

    def __init__(self, alpha, name=None, dtype=float):
        """
        Parameters
        ----------
        alpha : array-like
            Concentration parameters of the Dirichlet distribution. Must be positive.
        name : str, optional
            Name of the dimension.
        dtype : type, default=float
            Data type of the samples.
        """
        self.alpha = np.asarray(alpha)
        if np.any(self.alpha <= 0):
            raise ValueError("All concentration parameters must be positive.")
        self.name = name
        self.dtype = dtype
        self._rvs = dirichlet(self.alpha)
        self.transformer = Identity()
        self.transform_ = "identity"
        self.n_dims = len(self.alpha)

    @property
    def transformed_size(self):
        return self.n_dims

    def set_transformer(self, transform="identity"):
        """Define transformer and random variable sampling."""
        self.transform_ = transform

        if self.transform_ == "identity":
            self.transformer = Identity()
        elif self.transform_ == "normalize":
            self.transformer = GeneralizedLogitTransform()
        else:
            raise ValueError(f"transform should be 'identity' or 'logit', got '{self.transform_}'")

        self._rvs = dirichlet(self.alpha)


    def rvs(self, n_samples=1, random_state=None):
        """Draw random samples from the Dirichlet distribution."""
        samples = self._rvs.rvs(size=n_samples, random_state=random_state)
        return samples.astype(self.dtype)

    def inverse_transform(self, Xt):
        """Inverse transform samples from the warped space back into the original space."""
        # For identity transform, this is simply ensuring correct dtype
        inv_transform = self.transformer.inverse_transform(Xt).astype(self.dtype)
        return inv_transform

    @property
    def transformed_bounds(self):
        # Bounds for each component are [0, 1]
        return [(0.0, 1.0)] * self.n_dims

    @property
    def bounds(self):
        # Bounds for each component are [0, 1]
        return [(0.0, 1.0)] * self.n_dims

    @property
    def is_constant(self):
        return False

    def __repr__(self):
        return f"Dirichlet(alpha={self.alpha}, transform='{self.transform_}')"

    def __contains__(self, point):
        point = np.asarray(point)
        if point.shape[-1] != self.n_dims:
            return False
        if np.any(point < 0.0) or np.any(point > 1.0):
            return False
        return np.isclose(np.sum(point), 1.0)

    def distance(self, a, b):
        """Compute distance between points `a` and `b`."""
        a = np.asarray(a)
        b = np.asarray(b)
        if not (a in self and b in self):
            raise RuntimeError(f"Can only compute distance for values within the space, not {a} and {b}.")

        # Use Euclidean distance or another appropriate metric
        return np.linalg.norm(a - b)

    def get_dimensions(self):
        """Return a list of dimensions representing the Dirichlet components."""
        return [
            Real(0.0, 1.0, name=f'{self.name}_{i}')
            for i in range(self.n_dims)
        ]