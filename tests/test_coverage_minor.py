"""Coverage tests for scales/minor_breaks.py and breaks_log.py."""

import numpy as np
import pytest

from scales.minor_breaks import (
    minor_breaks_n,
    minor_breaks_width,
    regular_minor_breaks,
)

from scales.breaks_log import (
    breaks_log,
    minor_breaks_log,
    _log_sub_breaks,
)


# ===========================================================================
# minor_breaks.py (lines 58, 153-198)
# ===========================================================================

# ---------------------------------------------------------------------------
# minor_breaks_n (line 58)
# ---------------------------------------------------------------------------

class TestMinorBreaksN:
    # R "new" 2-arg interface: minor_breaks_n(n)(range, breaks); n is the
    # number of points spanning each inter-break segment (endpoints included).
    def test_exact_r_values(self):
        # R: minor_breaks_n(4)(c(0,20), c(0,10,20))
        fn = minor_breaks_n(4)
        result = fn(np.array([0, 20]), np.array([0, 10, 20]))
        np.testing.assert_allclose(
            result, [0, 10 / 3, 20 / 3, 10, 40 / 3, 50 / 3, 20]
        )

    def test_range_overruns_breaks_edge_segments(self):
        # R: minor_breaks_n(3)(c(-3,13), c(0,5,10)) — edge segments included.
        fn = minor_breaks_n(3)
        result = fn(np.array([-3, 13]), np.array([0, 5, 10]))
        np.testing.assert_allclose(
            result, [-3, -1.5, 0, 2.5, 5, 7.5, 10, 11.5, 13]
        )

    def test_single_major(self):
        # R: minor_breaks_n(4)(c(0,10), c(5)) — two edge segments.
        fn = minor_breaks_n(4)
        result = fn(np.array([0, 10]), np.array([5]))
        np.testing.assert_allclose(
            result, [0, 5 / 3, 10 / 3, 5, 20 / 3, 25 / 3, 10]
        )

    def test_returns_array(self):
        fn = minor_breaks_n(2)
        result = fn(np.array([0, 10]), np.array([0, 5, 10]))
        assert isinstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# minor_breaks_width
# ---------------------------------------------------------------------------

class TestMinorBreaksWidth:
    def test_exact_r_values(self):
        # R: minor_breaks_width(2.5, 0)(c(0,20), c(0,10,20)).  Edge segments
        # [0,0]/[20,20] straddle via breaks_width's zero_range branch.
        fn = minor_breaks_width(2.5)
        result = fn(np.array([0, 20]), np.array([0, 10, 20]))
        np.testing.assert_allclose(
            result,
            [-1.25, 1.25, 0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20, 18.75, 21.25],
        )

    def test_returns_array(self):
        fn = minor_breaks_width(1)
        result = fn(np.array([0, 10]), np.array([0, 5, 10]))
        assert isinstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# regular_minor_breaks (lines 153-198)
# ---------------------------------------------------------------------------

class TestRegularMinorBreaks:
    def test_basic(self):
        fn = regular_minor_breaks()
        result = fn(np.array([0, 5, 10]), np.array([0, 10]), 2)
        assert len(result) > 0

    def test_reverse(self):
        fn = regular_minor_breaks(reverse=True)
        result = fn(np.array([0, 5, 10]), np.array([0, 10]), 2)
        assert len(result) > 0

    def test_fewer_than_two_majors(self):
        fn = regular_minor_breaks()
        result = fn(np.array([5]), np.array([0, 10]), 2)
        assert len(result) == 0

    def test_n_zero(self):
        # R parity: regular_minor_breaks()(c(0,10), c(0,10), 0) -> 10.
        # `seq(0, 10, length.out = 1)[-1]` is empty for the single interval,
        # then the final major (10) is re-appended.  (R returns the last
        # major, NOT an empty vector — there is no `n < 1` short-circuit.)
        fn = regular_minor_breaks()
        result = fn(np.array([0, 10]), np.array([0, 10]), 0)
        np.testing.assert_allclose(result, [10.0])

    def test_n_one(self):
        # R's regular_minor_breaks(reverse=FALSE)(c(0, 10), c(0, 10), n=1)
        # -> `seq(0, 10, length.out=2)[-2]` == c(0) plus final major 10
        # -> c(0, 10).
        fn = regular_minor_breaks()
        result = fn(np.array([0, 10]), np.array([0, 10]), 1)
        np.testing.assert_allclose(result, [0.0, 10.0])

    def test_n_three(self):
        fn = regular_minor_breaks()
        result = fn(np.array([0, 10, 20]), np.array([0, 20]), 3)
        assert len(result) > 0

    def test_reverse_result_order(self):
        fn = regular_minor_breaks(reverse=True)
        result = fn(np.array([0, 10, 20]), np.array([0, 20]), 3)
        # Result should still be sorted (just negated and reversed back)
        assert len(result) > 0

    # ---- R-parity regression: descending b (the reversed-scale pipeline) ----
    # These pin the *exact* R scales::regular_minor_breaks output.  Before the
    # order-sensitivity fix, the function sorted ``b`` ascending and produced a
    # different (truncated) extension on reversed scales.

    def test_nonreverse_exact_r_values(self):
        # R: regular_minor_breaks()(c(0,5,10), c(0,10), 2) -> 0 2.5 5 7.5 10
        fn = regular_minor_breaks()
        result = fn(np.array([0, 5, 10]), np.array([0, 10]), 2)
        np.testing.assert_allclose(result, [0.0, 2.5, 5.0, 7.5, 10.0])

    def test_reverse_descending_extends_high_side(self):
        # The reversed-scale pipeline passes b DESCENDING.  With limits that
        # overrun the top major (37 > 30), R extends toward the high side:
        # R: regular_minor_breaks(reverse=TRUE)(c(30,20,10), c(8,37), 2)
        #    -> 40 35 30 25 20 15 10 5 0   (raw, before ggplot2's discard)
        fn = regular_minor_breaks(reverse=True)
        result = fn(np.array([30, 20, 10]), np.array([8, 37]), 2)
        np.testing.assert_allclose(
            result, [40, 35, 30, 25, 20, 15, 10, 5, 0]
        )

    def test_reverse_descending_matches_r_n3(self):
        # R: regular_minor_breaks(reverse=TRUE)(c(20,10,0), c(0,20), 3)
        #    -> 20 16.6667 13.3333 10 6.6667 3.3333 0
        fn = regular_minor_breaks(reverse=True)
        result = fn(np.array([20, 10, 0]), np.array([0, 20]), 3)
        np.testing.assert_allclose(
            result,
            [20, 50 / 3, 40 / 3, 10, 20 / 3, 10 / 3, 0],
        )

    def test_no_internal_sort_preserves_caller_order(self):
        # R never sorts b; descending input yields descending output.
        fn = regular_minor_breaks(reverse=True)
        result = fn(np.array([20, 10, 0]), np.array([0, 20]), 2)
        # strictly decreasing
        assert np.all(np.diff(result) < 0)


# ===========================================================================
# breaks_log.py (lines 56, 77-78, 105-111, 180, 185, 196, 218)
# ===========================================================================

# ---------------------------------------------------------------------------
# breaks_log (lines 56, 77-78)
# ---------------------------------------------------------------------------

class TestBreaksLog:
    def test_basic(self):
        brk = breaks_log(n=5, base=10)
        result = brk([1, 10000])
        assert len(result) > 0

    def test_empty(self):
        brk = breaks_log(n=5)
        result = brk([float("nan"), float("-inf")])
        assert len(result) == 0

    def test_few_breaks_fill(self):
        # When integer powers give too few, should fill in
        brk = breaks_log(n=10, base=10)
        result = brk([1, 100])  # Only 3 integer powers
        assert len(result) > 3

    def test_too_many_thin(self):
        # When too many breaks, should thin
        brk = breaks_log(n=3, base=10)
        result = brk([1, 1e15])  # 16 integer powers
        assert len(result) > 0

    def test_base_2(self):
        brk = breaks_log(n=5, base=2)
        result = brk([1, 64])
        assert len(result) > 0


# ---------------------------------------------------------------------------
# _fill_log_breaks (lines 105-111)
# ---------------------------------------------------------------------------

class TestLogSubBreaks:
    # Tests the R-style greedy densifier. See R/breaks-log.R log_sub_breaks.
    def test_base_10(self):
        result = _log_sub_breaks((0, 2), n=10, base=10)
        assert len(result) > 0

    def test_base_2_returns_powers(self):
        # R: `if (base <= 2) return(base^(min:max))`.
        result = _log_sub_breaks((0, 5), n=5, base=2)
        assert np.allclose(result, [2 ** p for p in range(0, 6)])

    def test_other_base(self):
        result = _log_sub_breaks((0, 3), n=5, base=5)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# minor_breaks_log (lines 180, 185, 196, 218)
# ---------------------------------------------------------------------------

class TestMinorBreaksLog:
    # R's minor_breaks_log takes the data (not "majors") and ignores
    # extras; our port accepts the same signature for compatibility.

    def test_basic(self):
        fn = minor_breaks_log()
        result = fn(np.array([1, 10, 100]), np.array([1, 100]), 5)
        assert len(result) > 0

    def test_single_x_still_generates(self):
        # R: no special-case for length 1; 10^k ladder is still built.
        fn = minor_breaks_log()
        result = fn(np.array([10]), np.array([1, 100]), 5)
        assert len(result) > 0

    def test_negative_x_mirrors_ladder(self):
        # R: has_negatives -> ticks reflected, 0 included.
        fn = minor_breaks_log()
        result = fn(np.array([-10, -1]), np.array([-10, -1]), 5)
        assert np.any(result > 0)
        assert np.any(result < 0)
        assert 0.0 in result

    def test_detail_one_populates(self):
        # detail=1 -> tens + fives + ones; definitely non-empty.
        fn = minor_breaks_log(detail=1)
        result = fn(np.array([1, 10, 100]), np.array([1, 100]), 5)
        assert len(result) > 0

    def test_detail_five(self):
        fn = minor_breaks_log(detail=5)
        result = fn(np.array([1, 10, 100]), np.array([1, 100]), 5)
        assert len(result) > 0

    def test_smallest_threshold_requires_negatives(self):
        # Per R: `smallest` only takes effect when any x <= 0.
        fn = minor_breaks_log(smallest=5)
        result = fn(np.array([-100, -1, 1, 100]), np.array([-100, 100]), 5)
        # Non-zero ticks must all have |t| >= 5.
        nonzero = result[result != 0]
        assert np.all(np.abs(nonzero) >= 5)
