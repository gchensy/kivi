"""Expert scorecards: factor weights from pairwise comparisons (the analytic hierarchy process)."""

import math
import typing
from collections import abc

import numpy as np
import numpy.typing as npt
import pandas as pd

from . import _validation

__all__ = ["ahp_consistency", "ahp_weights", "combine_weights"]

#: Saaty's random consistency index for matrices of order 1 to 15: the average consistency index
#: of random reciprocal matrices on Saaty's 1-9 scale. T. L. Saaty, *The Analytic Hierarchy
#: Process*, McGraw-Hill, 1980; the values are those listed by T. Obata and S. Shiraishi, *Size
#: independence of consistency index for pairwise comparison matrices in analytic hierarchy
#: process*, arXiv:2607.07950 (2026), Table 2. Some references print 1.48 for order 12.
_RANDOM_INDEX = (
    0.0,
    0.0,
    0.58,
    0.90,
    1.12,
    1.24,
    1.32,
    1.41,
    1.45,
    1.49,
    1.51,
    1.53,
    1.56,
    1.57,
    1.59,
)

#: Relative tolerance of the checks that a matrix is reciprocal and that weights sum to 1.
_TOLERANCE = 1e-6


def _eigen(matrix: npt.NDArray[np.float64]) -> tuple[float, npt.NDArray[np.float64]]:
    """Return the principal eigenvalue of a positive matrix and its eigenvector summing to 1."""
    values, vectors = np.linalg.eig(matrix)
    # Perron-Frobenius: the root of largest modulus is real and simple, and its eigenvector has
    # entries of one sign, so dividing by the sum fixes both the sign and the scale.
    principal = int(np.argmax(values.real))
    vector = vectors[:, principal].real
    return float(values[principal].real), vector / vector.sum()


def _comparison_matrix(
    matrix: pd.DataFrame | npt.ArrayLike,
) -> tuple[npt.NDArray[np.float64], pd.Index]:
    """Validate a pairwise comparison matrix; return it as floats with its factor labels."""
    if isinstance(matrix, pd.DataFrame):
        if not matrix.index.equals(matrix.columns):
            raise ValueError(
                "matrix must have the same labels on its index and columns, got "
                f"{_validation.shown_argument(matrix.index)} and "
                f"{_validation.shown_argument(matrix.columns)}"
            )
        if matrix.index.has_duplicates:
            repeated = matrix.index[matrix.index.duplicated()][0]
            raise ValueError(
                f"matrix must name each factor once, got {_validation.shown(repeated)} twice"
            )
    elif np.ndim(matrix) != 2:
        raise ValueError(f"matrix must be two-dimensional, got {np.ndim(matrix)} dimensions")
    values = _validation.as_real(matrix, "matrix")
    rows, columns = values.shape
    if rows != columns:
        raise ValueError(f"matrix must be square, got {rows} rows and {columns} columns")
    if not rows:
        raise ValueError("matrix must compare at least one factor")
    if isinstance(matrix, pd.DataFrame):
        index = matrix.index.rename("factor")
    else:
        index = pd.RangeIndex(rows, name="factor")
    labels = index.tolist()

    def where(row: int, column: int) -> str:
        return f"[{_validation.shown(labels[row])}, {_validation.shown(labels[column])}]"

    invalid = ~(np.isfinite(values) & (values > 0))
    if invalid.any():
        row, column = (int(position) for position in np.argwhere(invalid)[0])
        raise ValueError(
            "matrix must hold finite, positive numbers, got "
            f"{_validation.shown(values[row, column])} at {where(row, column)}"
        )
    off_one = ~np.isclose(np.diag(values), 1.0, rtol=_TOLERANCE, atol=0.0)
    if off_one.any():
        position = int(np.argmax(off_one))
        raise ValueError(
            "matrix must have 1 on its diagonal, got "
            f"{_validation.shown(values[position, position])} at {where(position, position)}"
        )
    off_reciprocal = ~np.isclose(values * values.T, 1.0, rtol=_TOLERANCE, atol=0.0)
    if off_reciprocal.any():
        row, column = (int(position) for position in np.argwhere(off_reciprocal)[0])
        raise ValueError(
            f"matrix must be reciprocal: the entries at {where(row, column)} and "
            f"{where(column, row)} must multiply to 1, got "
            f"{_validation.shown(values[row, column])} and {_validation.shown(values[column, row])}"
        )
    return values, index


def ahp_weights(
    matrix: pd.DataFrame | npt.ArrayLike,
    *,
    method: typing.Literal["eigenvector", "geometric"] = "eigenvector",
) -> pd.Series:
    r"""Weigh factors from a pairwise comparison matrix, as the analytic hierarchy process does.

    Entry :math:`a_{ij}` of the matrix says how much more important factor
    :math:`i` is than factor :math:`j`, on Saaty's scale from 1 (equally
    important) to 9 (extremely more important), and its reciprocal
    :math:`a_{ji} = 1 / a_{ij}` the other way round. The weights are
    proportional to the principal eigenvector of the matrix, or to the
    geometric means of its rows, and sum to 1.

    Parameters
    ----------
    matrix : DataFrame or array_like
        A square matrix of pairwise comparisons with a diagonal of 1 and
        reciprocal entries, :math:`a_{ij} a_{ji} = 1` to a relative tolerance
        of 1e-6. A DataFrame names the factors by its index, which must equal
        its columns; an array numbers them from 0.
    method : {"eigenvector", "geometric"}, default "eigenvector"
        ``"eigenvector"`` is Saaty's principal eigenvector. ``"geometric"``
        takes the geometric mean of each row, which gives the same weights
        for a consistent matrix and needs no eigenvalue computation.

    Returns
    -------
    pandas.Series
        The weights, named ``weight`` and indexed by the factors in an index
        named ``factor``. They are positive and sum to 1.

    Raises
    ------
    ValueError
        If `matrix` is not a square matrix of finite, positive numbers with a
        diagonal of 1 and reciprocal entries, a DataFrame's index and columns
        differ or repeat a label, or `method` is not one of the two names.

    See Also
    --------
    ahp_consistency : Check how consistent the judgments in the matrix are.
    combine_weights : Weigh the sub-factors of a hierarchy.

    Notes
    -----
    For a consistent matrix, :math:`a_{ij} = w_i / w_j`, both methods give
    :math:`w`. Otherwise the eigenvector method solves :math:`A w = \lambda_{\max} w`
    and the geometric method takes

    .. math::

       w_i = \frac{\left(\prod_j a_{ij}\right)^{1/n}}
                  {\sum_k \left(\prod_j a_{kj}\right)^{1/n}}.

    For two or three factors the two methods give the same weights for every
    reciprocal matrix; they differ slightly from four factors on if the
    judgments are inconsistent. The legacy 0.2.x ``AHP`` class used the
    geometric method.

    Examples
    --------
    >>> import pandas as pd
    >>> from kivi.expert import ahp_weights
    >>> factors = ["sales", "margin", "leverage"]
    >>> matrix = pd.DataFrame(
    ...     [[1, 2, 4], [1 / 2, 1, 2], [1 / 4, 1 / 2, 1]], index=factors, columns=factors
    ... )
    >>> ahp_weights(matrix).round(4)
    factor
    sales       0.5714
    margin      0.2857
    leverage    0.1429
    Name: weight, dtype: float64
    """
    if method not in ("eigenvector", "geometric"):
        raise ValueError(
            f"method must be 'eigenvector' or 'geometric', got {_validation.shown_argument(method)}"
        )
    values, index = _comparison_matrix(matrix)
    if method == "eigenvector":
        weights = _eigen(values)[1]
    else:
        # Logarithms keep the product of many large entries from overflowing.
        geometric = np.exp(np.log(values).mean(axis=1))
        weights = geometric / geometric.sum()
    return pd.Series(weights, index=index, name="weight")


def ahp_consistency(matrix: pd.DataFrame | npt.ArrayLike) -> pd.Series:
    r"""Measure how consistent the judgments of a pairwise comparison matrix are.

    Judgments are consistent when they agree with each other: if factor A is
    twice as important as B and B twice as important as C, then A should be
    four times as important as C. The consistency ratio compares the
    inconsistency of the matrix with that of random judgments.

    Parameters
    ----------
    matrix : DataFrame or array_like
        A square matrix that compares at most 15 factors, as for
        `ahp_weights`.

    Returns
    -------
    pandas.Series
        The statistics, named ``value`` and indexed by ``lambda_max``, ``ci``, ``ri`` and ``cr``
        in an index named ``statistic``:

        * ``lambda_max``, the principal eigenvalue of the matrix, at least
          the number of factors :math:`n`;
        * ``ci``, the consistency index, zero for consistent judgments;
        * ``ri``, Saaty's random index for :math:`n`;
        * ``cr``, the consistency ratio ``ci / ri``.

    Raises
    ------
    ValueError
        If `matrix` is not a valid comparison matrix, as for `ahp_weights`, or
        compares more than 15 factors, the largest order with a tabulated random
        index.

    See Also
    --------
    ahp_weights : The weights that the matrix gives.

    Notes
    -----
    With :math:`n` factors,

    .. math::

       \mathrm{CI} = \frac{\lambda_{\max} - n}{n - 1}, \qquad
       \mathrm{CR} = \frac{\mathrm{CI}}{\mathrm{RI}}.

    Saaty suggested that judgments with a CR above 0.1 should be reviewed.
    That is a rule of thumb, not a test, and the function does not raise for a
    larger CR. Every matrix of one or two factors is consistent, so its
    ``ci``, ``ri`` and ``cr`` are 0. For a reciprocal matrix
    :math:`\lambda_{\max} \ge n`; a computed value below :math:`n`, from
    rounding, is set to :math:`n`.

    The random index is the average CI of random reciprocal matrices on
    Saaty's scale, tabulated by T. L. Saaty, *The Analytic Hierarchy
    Process*, McGraw-Hill, 1980, for orders 1 to 15 (the values are those
    listed by T. Obata and S. Shiraishi, arXiv:2607.07950, Table 2). The
    legacy 0.2.x class estimated :math:`\lambda_{\max}` as the mean of
    :math:`(A w)_i / w_i` for the geometric-mean weights, which equals the
    exact value for up to three factors, and used almost the same random
    indices.

    Examples
    --------
    >>> import numpy as np
    >>> from kivi.expert import ahp_consistency
    >>> matrix = np.array([[1, 3, 5], [1 / 3, 1, 3], [1 / 5, 1 / 3, 1]])
    >>> ahp_consistency(matrix).round(4)
    statistic
    lambda_max    3.0385
    ci            0.0193
    ri            0.5800
    cr            0.0332
    Name: value, dtype: float64
    """
    values, _ = _comparison_matrix(matrix)
    order = len(values)
    if order > len(_RANDOM_INDEX):
        raise ValueError(
            f"matrix must compare at most {len(_RANDOM_INDEX)} factors, the largest order with a "
            f"tabulated random index, got {order}"
        )
    if order <= 2:
        lambda_max = float(order)
        consistency = 0.0
        ratio = 0.0
    else:
        lambda_max = max(_eigen(values)[0], float(order))
        consistency = (lambda_max - order) / (order - 1)
        ratio = consistency / _RANDOM_INDEX[order - 1]
    return pd.Series(
        [lambda_max, consistency, _RANDOM_INDEX[order - 1], ratio],
        index=pd.Index(["lambda_max", "ci", "ri", "cr"], name="statistic"),
        name="value",
    )


def combine_weights(parent: pd.Series, children: abc.Mapping[abc.Hashable, pd.Series]) -> pd.Series:
    """Combine the weights of factors and of their sub-factors into global weights.

    In a hierarchy, each factor is split into sub-factors that are weighed
    against each other. The global weight of a sub-factor is the weight of
    its factor times its weight within the factor, so the global weights of
    all sub-factors still sum to 1.

    Parameters
    ----------
    parent : Series
        The weights of the factors, such as the result of `ahp_weights`. They
        must be finite and non-negative and sum to 1 to a relative tolerance
        of 1e-6.
    children : mapping
        For each factor of `parent`, a Series of the weights of its
        sub-factors, indexed by their labels, such as the result of
        `ahp_weights`. Each must be finite and non-negative, sum to 1 and
        name each sub-factor once. A factor without sub-factors takes a
        Series of one label with the weight 1.

    Returns
    -------
    pandas.Series
        The global weights, named ``weight`` and indexed by ``factor`` and
        ``subfactor``, in the order of `parent` and then of each factor's
        sub-factors.

    Raises
    ------
    ValueError
        If `parent` is not a Series, `children` is not a mapping of Series,
        its keys are not the factors of `parent`, or a set of weights is not
        finite, non-negative and summing to 1.

    See Also
    --------
    ahp_weights : Weigh the factors and the sub-factors of each factor.

    Examples
    --------
    >>> import pandas as pd
    >>> from kivi.expert import combine_weights
    >>> parent = pd.Series({"finance": 0.75, "market": 0.25})
    >>> children = {
    ...     "finance": pd.Series({"leverage": 0.6, "margin": 0.4}),
    ...     "market": pd.Series({"position": 1.0}),
    ... }
    >>> combine_weights(parent, children)
    factor   subfactor
    finance  leverage     0.45
             margin       0.30
    market   position     0.25
    Name: weight, dtype: float64
    """
    _check_weights(parent, "parent")
    if not isinstance(children, abc.Mapping):
        raise ValueError(
            "children must be a mapping of the factors of parent to Series of weights, got "
            f"{type(children).__name__}"
        )
    absent = [label for label in parent.index if label not in children]
    if absent:
        raise ValueError(
            "children must have an entry for every factor of parent, missing "
            f"{_validation.shown_list(absent)}"
        )
    unknown = [label for label in children if label not in parent.index]
    if unknown:
        raise ValueError(
            "children must only have entries for the factors of parent, got "
            f"{_validation.shown_list(unknown)}"
        )
    labels: list[tuple[object, object]] = []
    weights: list[float] = []
    for factor, local in children.items():
        _check_weights(local, f"children[{_validation.shown(factor)}]")
    for factor, share in parent.items():
        local = children[factor]
        labels.extend((factor, subfactor) for subfactor in local.index)
        weights.extend(float(share) * local.to_numpy(dtype=float))
    index = pd.MultiIndex.from_tuples(labels, names=["factor", "subfactor"])
    return pd.Series(weights, index=index, name="weight")


def _check_weights(weights: object, name: str) -> None:
    """Require a Series of finite, non-negative weights with distinct labels that sum to 1."""
    if not isinstance(weights, pd.Series):
        raise ValueError(f"{name} must be a Series of weights, got {type(weights).__name__}")
    if not len(weights):
        raise ValueError(f"{name} must have at least one weight")
    if weights.index.has_duplicates:
        repeated = weights.index[weights.index.duplicated()][0]
        raise ValueError(
            f"{name} must name each label once, got {_validation.shown(repeated)} twice"
        )
    values = _validation.as_real(weights, name)
    invalid = ~(np.isfinite(values) & (values >= 0))
    if invalid.any():
        raise ValueError(
            f"{name} must be finite and non-negative, got {_validation.shown(values[invalid][0])}"
        )
    total = float(values.sum())
    if not math.isclose(total, 1.0, rel_tol=_TOLERANCE):
        raise ValueError(f"{name} must sum to 1, got {total!r}")
