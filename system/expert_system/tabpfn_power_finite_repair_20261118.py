"""Handle undefined power-transform values using TabPFN's missing-value convention."""

import numpy as np


def install():
    from tabpfn.model.preprocessing import SafePowerTransformer

    original = SafePowerTransformer.transform
    events = []

    def transform(self, values):
        try:
            return original(self, values)
        except ValueError as error:
            if "contains infinity" not in str(error) or self.method != "yeo-johnson":
                raise
            transformed = np.asarray(values, dtype=float).copy()
            with np.errstate(over="ignore", invalid="ignore"):
                for column, power in enumerate(self.lambdas_):
                    transformed[:, column] = self._yeo_johnson_transform(transformed[:, column], power)
            overflow = np.isinf(transformed)
            if not overflow.any():
                raise
            events.append(dict(nonfinite_power_values=int(overflow.sum()),
                               columns=int(overflow.any(axis=0).sum())))
            transformed[overflow] = np.nan
            if self.standardize:
                transformed = self._scaler.transform(transformed)
            assert not np.isinf(transformed).any()
            return self._revert_failed_features(transformed, values)

    SafePowerTransformer.transform = transform
    return events
