import numpy as np
from skopt.space import Dimension, Identity, Pipeline, Normalize
from scipy.stats import expon, beta
from skopt.space.space import _uniform_inclusive


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
