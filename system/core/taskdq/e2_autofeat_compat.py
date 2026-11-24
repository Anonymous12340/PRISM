"""AutoFeat's native scaler needs the conventional zero-variance behavior."""
import numpy as np


def standard_scale_with_constant_guard(array):
    from autofeat.nb_utils import nb_standard_scale, nb_nanstd
    # Retain upstream arithmetic exactly for every nonzero-variance column.
    with np.errstate(divide='ignore', invalid='ignore'):
        result = nb_standard_scale(array)
    zero = nb_nanstd(array, 0) == 0
    if zero.any():
        result[:, zero] = 0.
    return result


def install_constant_guard():
    import autofeat.featsel as featsel
    featsel.nb_standard_scale = standard_scale_with_constant_guard


def constant_guard_backend():
    # joblib 1.4 passes executor options through configure, not backend.__init__.
    from joblib._parallel_backends import LokyBackend
    class GuardedLokyBackend(LokyBackend):
        def configure(self, *args, **kwargs):
            kwargs['initializer'] = install_constant_guard
            return super().configure(*args, **kwargs)
    return GuardedLokyBackend()
