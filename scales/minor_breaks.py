"""
Minor-break generators for linear (non-log) scales.

Python port of ``R/minor_breaks.R`` from the R *scales* package
(https://github.com/r-lib/scales).

All public generators are closure factories: they return a callable with
signature ``(major_breaks, limits, n) -> numpy.ndarray``.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "minor_breaks_n",
    "minor_breaks_width",
    "regular_minor_breaks",
]


def loop_breaks(
    range_: ArrayLike,
    breaks: ArrayLike,
    f: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """Apply *f* to each inter-break segment and dedupe.

    Faithful port of R scales ``loop_breaks`` (R/minor_breaks.R:36-47).  *f*
    is applied to every segment ``[range_start, break_1]``,
    ``[break_{i-1}, break_i]`` for ``i = 2..len(breaks)``, and
    ``[break_last, range_end]``; the concatenated results are returned with
    duplicates removed in first-occurrence order (R ``unique()``).

    Parameters
    ----------
    range_ : array-like of length 2
        The ``[start, end]`` range, in data space.
    breaks : array-like
        The major break positions, in data space.
    f : callable
        ``(segment) -> numpy.ndarray`` applied to each 2-element segment.

    Returns
    -------
    numpy.ndarray
    """
    range_ = np.asarray(range_, dtype=float)
    breaks = np.asarray(breaks, dtype=float)
    n = breaks.size
    if n == 0:
        return np.array([], dtype=float)

    out = [np.asarray(f(np.array([range_[0], breaks[0]])), dtype=float)]
    for i in range(1, n):
        out.append(np.asarray(f(np.array([breaks[i - 1], breaks[i]])), dtype=float))
    out.append(np.asarray(f(np.array([breaks[n - 1], range_[1]])), dtype=float))

    allv = np.concatenate([o.ravel() for o in out])
    # R unique(): keep first occurrence, preserve order (shared segment
    # endpoints are exactly equal, so they dedupe cleanly).
    seen: set = set()
    keep: list[float] = []
    for v in allv:
        fv = float(v)
        if fv not in seen:
            seen.add(fv)
            keep.append(fv)
    return np.array(keep, dtype=float)


def minor_breaks_n(
    n: int = 2,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Minor breaks with a fixed number of points per segment.

    Faithful port of R scales ``minor_breaks_n`` (R/minor_breaks.R:26-34):
    returns a ``(range, breaks)`` function that places ``seq(length = n)``
    points across each inter-break segment (via :func:`loop_breaks`).  *n* is
    the number of points spanning each segment (endpoints included), matching
    R's ``seq(rng[1], rng[2], length = n)`` — NOT a count of strictly-interior
    breaks.

    Parameters
    ----------
    n : int, optional
        Points per segment (default 2).

    Returns
    -------
    callable
        A function ``(range, breaks) -> numpy.ndarray`` of minor-break
        positions (the R "new" 2-argument interface).

    Examples
    --------
    >>> fn = minor_breaks_n(4)
    >>> fn(np.array([0, 20]), np.array([0, 10, 20]))
    array([ 0.        ,  3.33333333,  6.66666667, 10.        , 13.33333333,
           16.66666667, 20.        ])
    """

    def f(seg: np.ndarray) -> np.ndarray:
        # R: seq(rng[1], rng[2], length = n)
        return np.linspace(seg[0], seg[1], int(n))

    def _minor_breaks(range_: ArrayLike, breaks: ArrayLike) -> np.ndarray:
        return loop_breaks(range_, breaks, f)

    return _minor_breaks


def minor_breaks_width(
    width: float,
    offset: float = 0,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Minor breaks at a fixed width per segment.

    Faithful port of R scales ``minor_breaks_width`` (R/minor_breaks.R:14-21):
    returns a ``(range, breaks)`` function that applies
    :func:`~scales.breaks.breaks_width` to each inter-break segment (via
    :func:`loop_breaks`).

    Parameters
    ----------
    width : float
        Spacing between consecutive minor breaks.
    offset : float, optional
        Offset for the break grid (default 0).  NOTE: R's
        ``minor_breaks_width`` declares *offset* without a default (an upstream
        inconsistency with the ``breaks_width(offset = 0)`` it wraps and its
        own ``@inheritParams`` docs); we default it to 0 — a documented,
        intentional deviation.

    Returns
    -------
    callable
        A function ``(range, breaks) -> numpy.ndarray`` of minor-break
        positions (the R "new" 2-argument interface).

    Examples
    --------
    >>> fn = minor_breaks_width(2.5)
    >>> fn(np.array([0, 20]), np.array([0, 10, 20]))
    array([-1.25,  1.25,  0.  ,  2.5 ,  5.  ,  7.5 , 10.  , 12.5 , 15.  ,
           17.5 , 20.  , 18.75, 21.25])
    """
    from .breaks import breaks_width

    f = breaks_width(width, offset)

    def _minor_breaks(range_: ArrayLike, breaks: ArrayLike) -> np.ndarray:
        return loop_breaks(range_, breaks, f)

    return _minor_breaks


def regular_minor_breaks(
    reverse: bool = False,
) -> Callable[[np.ndarray, np.ndarray, int], np.ndarray]:
    """Default minor-break placement: ``n - 1`` evenly spaced between majors.

    This is the standard minor-break strategy used by ggplot2.  Between
    each pair of consecutive major breaks, ``n - 1`` minor breaks are
    inserted at equal spacing.

    Parameters
    ----------
    reverse : bool, optional
        If ``True``, the limits are internally reversed before
        computing breaks and the result is reversed back.  Useful for
        reversed continuous scales (default ``False``).

    Returns
    -------
    callable
        A function ``(major_breaks, limits, n) -> numpy.ndarray``
        of minor-break positions.

    Examples
    --------
    >>> fn = regular_minor_breaks()
    >>> fn(np.array([0, 5, 10]), np.array([0, 10]), 2)
    array([ 0. ,  2.5,  5. ,  7.5, 10. ])
    """

    def _minor_breaks(
        major: np.ndarray,
        limits: np.ndarray,
        n: int = 2,
    ) -> np.ndarray:
        # Faithful port of R scales::regular_minor_breaks
        # (R/minor_breaks.R:65-93).  R drops NAs but does NOT sort ``b``.
        # The toward-limits extension is *order-sensitive*: the reversed-scale
        # pipeline passes ``b`` in DESCENDING order, and the SIGNED first
        # difference (``bd``) together with the ``reverse`` branch steer the
        # extension to the correct end.  Sorting ``b`` ascending (as an earlier
        # port did) flips the sign of ``bd`` and sends the extension to the
        # wrong side, which — after the downstream ``discard(breaks, limits)``
        # in ggplot2's get_breaks_minor — silently drops the top-most minor
        # break on every reversed continuous scale whose data limit overruns
        # the highest major break (the common case).  Mirror R exactly: the
        # extension CONDITIONS use min/max(b) (order-independent), while the
        # prepended/appended VALUES use the positional first/last element.
        b = np.asarray(major, dtype=float)
        b = b[~np.isnan(b)]

        if b.size < 2:
            return np.array([], dtype=float)

        lim = np.asarray(limits, dtype=float)
        lo_lim, hi_lim = float(np.min(lim)), float(np.max(lim))

        bd = float(b[1] - b[0])  # R: diff(b)[1] — signed first difference

        if not reverse:
            if lo_lim < b.min():
                b = np.concatenate(([b[0] - bd], b))
            if hi_lim > b.max():
                b = np.concatenate((b, [b[-1] + bd]))
        else:
            if hi_lim > b.max():
                b = np.concatenate(([b[0] - bd], b))
            if lo_lim < b.min():
                b = np.concatenate((b, [b[-1] + bd]))

        # R seq_between: (n+1)-point seq between each consecutive pair,
        # dropping the last point (it's the next major); then re-append the
        # final major.  No special-casing of n: for n=0 every interval
        # contributes nothing and the result is just the final major, exactly
        # as R's `seq(a, b, length.out = 1)[-1]` does.
        pieces: list[np.ndarray] = []
        for i in range(b.size - 1):
            seq = np.linspace(b[i], b[i + 1], int(n) + 1)[:-1]
            pieces.append(seq)
        pieces.append(np.array([b[-1]], dtype=float))
        return np.concatenate(pieces)

    return _minor_breaks
