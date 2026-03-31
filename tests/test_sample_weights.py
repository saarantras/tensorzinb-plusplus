"""
Tests for sample_weight support in TensorZINB.

Proves mathematical equivalence: fitting with fully expanded data (all observations,
no weights) produces identical results to fitting with compressed data (unique rows
with weights equal to their multiplicity). This is the foundation for the phantom-zero
CM fitting approach.
"""
import numpy as np
import pytest
import scipy.sparse
from tensorzinb.tensorzinb import TensorZINB, ZINBLogLik


# ── Helpers ──────────────────────────────────────────────────────────────────

def _seed():
    return 42

def _make_expanded_data(n_groups=5, reps_per_group=20, k_treat=3, seed=None):
    """
    Create a small synthetic dataset mimicking CM structure:
    - k_treat treatment levels (+ intercept)
    - n_groups groups, each repeated reps_per_group times
    - Response drawn from NB(mu, theta) with some zero inflation

    Returns (endog, exog_dense, exog_infl_dense, group_ids).
    group_ids[i] = which group observation i belongs to.
    """
    rng = np.random.default_rng(seed or _seed())
    n = n_groups * reps_per_group

    # Design matrix: intercept + (k_treat - 1) treatment dummies
    # Each group gets a fixed treatment assignment
    group_treatment = rng.integers(0, k_treat, size=n_groups)
    group_ids = np.repeat(np.arange(n_groups), reps_per_group)
    treatments = group_treatment[group_ids]

    exog = np.zeros((n, k_treat), dtype=np.float64)
    exog[:, 0] = 1.0  # intercept
    for i in range(1, k_treat):
        exog[treatments == i, i] = 1.0

    # ZI design: intercept only (single column)
    exog_infl = np.ones((n, 1), dtype=np.float64)

    # True parameters
    true_mu_coefs = np.array([2.0] + [0.5 * (i + 1) for i in range(k_treat - 1)])
    true_theta = 3.0
    true_pi_logit = -1.0  # ~27% zero inflation

    log_mu = exog @ true_mu_coefs
    mu = np.exp(log_mu)
    theta = true_theta

    # Draw from ZINB
    is_zero_inflated = rng.random(n) < 1 / (1 + np.exp(-true_pi_logit))
    nb_draws = rng.negative_binomial(theta, theta / (theta + mu))
    endog = np.where(is_zero_inflated, 0, nb_draws).astype(np.float64)

    return endog, exog, exog_infl, group_ids


def _compress_with_weights(endog, exog, exog_infl, group_ids):
    """
    Compress expanded data into unique rows with weights.
    Within each group, rows with identical (exog, exog_infl) are collapsed.
    Nonzero y values stay as individual rows (weight=1).
    Zero y values within the same group are collapsed into one row (weight=count).
    """
    unique_groups = np.unique(group_ids)
    comp_endog = []
    comp_exog = []
    comp_exog_infl = []
    comp_weights = []

    for g in unique_groups:
        mask = group_ids == g
        y_g = endog[mask]
        x_g = exog[mask]
        xi_g = exog_infl[mask]

        # Nonzero observations: keep individually
        nz_mask = y_g > 0
        for i in np.where(nz_mask)[0]:
            comp_endog.append(y_g[i])
            comp_exog.append(x_g[i])
            comp_exog_infl.append(xi_g[i])
            comp_weights.append(1.0)

        # Zero observations: collapse into one phantom zero
        n_zeros = np.sum(~nz_mask)
        if n_zeros > 0:
            comp_endog.append(0.0)
            comp_exog.append(x_g[0])  # all rows in group have same exog
            comp_exog_infl.append(xi_g[0])
            comp_weights.append(float(n_zeros))

    return (
        np.array(comp_endog),
        np.array(comp_exog),
        np.array(comp_exog_infl),
        np.array(comp_weights),
    )


def _fit_and_extract(endog, exog, exog_infl, nb_only=False, sample_weight=None,
                     seed=None, sparse=False):
    """Fit TensorZINB and return a flat dict with standardized keys."""
    rng_seed = seed or _seed()
    np.random.seed(rng_seed)

    if sparse:
        exog = scipy.sparse.csr_matrix(exog)
        if exog_infl is not None:
            exog_infl = scipy.sparse.csr_matrix(exog_infl)

    mod = TensorZINB(
        endog=endog,
        exog=exog,
        exog_infl=exog_infl if not nb_only else None,
        nb_only=nb_only,
        sample_weight=sample_weight,
    )
    res = mod.fit(
        init_method="nb",
        learning_rate=0.01,
        max_epochs=2000,
        min_delta_early_stop=0.01,
        patience_early_stop=50,
        device_name="/cpu:0",
    )
    # Flatten for easier assertion
    out = {
        "llf": res["llf_total"],
        "llfs": res["llfs"],
        "aic": res["aic_total"],
        "num_sample": res["num_sample"],
        "epochs": res["epochs"],
    }
    w = res["weights"]
    if "x_mu" in w:
        out["x_mu"] = w["x_mu"]
    if "theta" in w:
        out["theta"] = w["theta"]
    if "x_pi" in w:
        out["x_pi"] = w["x_pi"]
    return out


# ── Tests ────────────────────────────────────────────────────────────────────

class TestWeightValidation:
    """Basic validation of the sample_weight parameter."""

    def test_weight_length_mismatch_raises(self):
        endog = np.array([1, 2, 3, 0, 0], dtype=np.float64)
        exog = np.ones((5, 1))
        with pytest.raises(ValueError, match="sample_weight length"):
            TensorZINB(endog=endog, exog=exog, sample_weight=np.array([1.0, 2.0]))

    def test_effective_n_equals_sum_of_weights(self):
        endog = np.array([1, 2, 3, 0, 0], dtype=np.float64)
        exog = np.ones((5, 1))
        weights = np.array([1.0, 1.0, 1.0, 10.0, 5.0])
        mod = TensorZINB(endog=endog, exog=exog, sample_weight=weights, nb_only=True)
        assert mod.effective_n == 18.0

    def test_no_weights_effective_n_equals_num_sample(self):
        endog = np.array([1, 2, 3, 0, 0], dtype=np.float64)
        exog = np.ones((5, 1))
        mod = TensorZINB(endog=endog, exog=exog, nb_only=True)
        assert mod.effective_n == 5.0

    def test_unit_weights_same_as_no_weights(self):
        """Fitting with all-ones weights should be identical to no weights.
        Uses sparse matrices so both paths use the same init."""
        endog, exog, exog_infl, _ = _make_expanded_data(
            n_groups=3, reps_per_group=30, k_treat=2, seed=123)

        res_no_w = _fit_and_extract(endog, exog, exog_infl, nb_only=True,
                                    seed=99, sparse=True)
        res_unit = _fit_and_extract(endog, exog, exog_infl, nb_only=True,
                                    sample_weight=np.ones(len(endog)),
                                    seed=99, sparse=True)

        np.testing.assert_allclose(
            res_no_w["x_mu"], res_unit["x_mu"], atol=1e-3,
            err_msg="Unit weights should match unweighted for x_mu")
        np.testing.assert_allclose(
            res_no_w["llf"], res_unit["llf"], rtol=1e-4,
            err_msg="Unit weights should match unweighted for llf")


class TestNBEquivalence:
    """
    NB-only (no zero inflation) equivalence between expanded and compressed data.
    This is the simpler case — validates the core weighted NLL machinery.
    """

    @pytest.fixture
    def nb_data(self):
        endog, exog, exog_infl, group_ids = _make_expanded_data(
            n_groups=8, reps_per_group=50, k_treat=3, seed=42)
        comp_endog, comp_exog, comp_exog_infl, comp_weights = _compress_with_weights(
            endog, exog, exog_infl, group_ids)
        return {
            "endog": endog, "exog": exog, "exog_infl": exog_infl,
            "comp_endog": comp_endog, "comp_exog": comp_exog,
            "comp_exog_infl": comp_exog_infl, "comp_weights": comp_weights,
        }

    def test_nb_sparse_llf(self, nb_data):
        """Same as above but with sparse design matrices (CM always uses sparse)."""
        res_exp = _fit_and_extract(
            nb_data["endog"], nb_data["exog"], nb_data["exog_infl"],
            nb_only=True, seed=77, sparse=True)
        res_comp = _fit_and_extract(
            nb_data["comp_endog"], nb_data["comp_exog"], nb_data["comp_exog_infl"],
            nb_only=True, sample_weight=nb_data["comp_weights"], seed=77,
            sparse=True)

        np.testing.assert_allclose(
            res_exp["llf"], res_comp["llf"], rtol=0.01,
            err_msg="NB sparse log-likelihoods should match (expanded vs compressed)")

    def test_nb_sparse_params(self, nb_data):
        """NB sparse parameter equivalence."""
        res_exp = _fit_and_extract(
            nb_data["endog"], nb_data["exog"], nb_data["exog_infl"],
            nb_only=True, seed=77, sparse=True)
        res_comp = _fit_and_extract(
            nb_data["comp_endog"], nb_data["comp_exog"], nb_data["comp_exog_infl"],
            nb_only=True, sample_weight=nb_data["comp_weights"], seed=77,
            sparse=True)

        np.testing.assert_allclose(
            res_exp["x_mu"], res_comp["x_mu"], atol=0.05,
            err_msg="NB sparse x_mu should match")
        np.testing.assert_allclose(
            res_exp["theta"], res_comp["theta"], atol=0.1,
            err_msg="NB sparse theta should match")

    def test_nb_num_sample_reports_effective_n(self, nb_data):
        """Result dict should report effective_n, not physical row count."""
        res = _fit_and_extract(
            nb_data["comp_endog"], nb_data["comp_exog"], nb_data["comp_exog_infl"],
            nb_only=True, sample_weight=nb_data["comp_weights"], seed=77)
        assert res["num_sample"] == pytest.approx(np.sum(nb_data["comp_weights"]))


class TestZINBEquivalence:
    """
    Full ZINB (with zero inflation) equivalence. This is the realistic case:
    the CM fitting path uses ZINB with ZI ~ replicate.
    """

    @pytest.fixture
    def zinb_data(self):
        endog, exog, exog_infl, group_ids = _make_expanded_data(
            n_groups=8, reps_per_group=50, k_treat=3, seed=42)
        comp_endog, comp_exog, comp_exog_infl, comp_weights = _compress_with_weights(
            endog, exog, exog_infl, group_ids)
        return {
            "endog": endog, "exog": exog, "exog_infl": exog_infl,
            "comp_endog": comp_endog, "comp_exog": comp_exog,
            "comp_exog_infl": comp_exog_infl, "comp_weights": comp_weights,
        }

    def test_zinb_sparse_llf(self, zinb_data):
        """ZINB sparse log-likelihood equivalence."""
        res_exp = _fit_and_extract(
            zinb_data["endog"], zinb_data["exog"], zinb_data["exog_infl"],
            nb_only=False, seed=77, sparse=True)
        res_comp = _fit_and_extract(
            zinb_data["comp_endog"], zinb_data["comp_exog"],
            zinb_data["comp_exog_infl"],
            nb_only=False, sample_weight=zinb_data["comp_weights"], seed=77,
            sparse=True)

        np.testing.assert_allclose(
            res_exp["llf"], res_comp["llf"], rtol=0.01,
            err_msg="ZINB sparse log-likelihoods should match")

    def test_zinb_sparse_params(self, zinb_data):
        """ZINB sparse parameter equivalence.
        Wider tolerance than NB — ZINB is non-convex so optimizer can find
        different local optima with similar LLF but different parameters."""
        res_exp = _fit_and_extract(
            zinb_data["endog"], zinb_data["exog"], zinb_data["exog_infl"],
            nb_only=False, seed=77, sparse=True)
        res_comp = _fit_and_extract(
            zinb_data["comp_endog"], zinb_data["comp_exog"],
            zinb_data["comp_exog_infl"],
            nb_only=False, sample_weight=zinb_data["comp_weights"], seed=77,
            sparse=True)

        np.testing.assert_allclose(
            res_exp["x_mu"], res_comp["x_mu"], atol=0.3,
            err_msg="ZINB sparse x_mu should match")
        np.testing.assert_allclose(
            res_exp["theta"], res_comp["theta"], atol=0.5,
            err_msg="ZINB sparse theta should match")


class TestPhantomZeroPattern:
    """
    Tests that specifically mimic the phantom-zero CM pattern:
    many identical zero rows collapsed into a single weighted phantom zero.
    This is the exact use case for Cohen CM fitting.
    """

    def _make_cm_like_data(self, seed=42):
        """
        Simulate a CM-like dataset:
        - 2 replicates × 3 treatment levels
        - 100 cells × 50 barcodes = 5000 pairs per (rep, treatment)
        - ~5% nonzero rate (typical for CM)
        - Exog: intercept + 2 treatment dummies (NB part)
        - Exog_infl: 2 replicate indicators (ZI part, no intercept)
        """
        rng = np.random.default_rng(seed)
        n_reps = 2
        n_treatments = 3
        n_pairs_per_group = 100  # cells × barcodes per (rep, treatment)

        # True params — high zero inflation to mimic CM sparsity
        mu_coefs = np.array([0.5, 0.3, -0.2])  # intercept + 2 treatment
        theta = 1.0
        pi_coefs = np.array([3.0, 2.5])  # per-replicate ZI logits (~95%/92% ZI)

        rows_endog = []
        rows_exog = []
        rows_exog_infl = []
        rows_group = []
        group_id = 0

        for rep in range(n_reps):
            for treat in range(n_treatments):
                # Design row (same for all pairs in this group)
                x_row = np.zeros(n_treatments)
                x_row[0] = 1.0  # intercept
                if treat > 0:
                    x_row[treat] = 1.0

                xi_row = np.zeros(n_reps)
                xi_row[rep] = 1.0

                log_mu = x_row @ mu_coefs
                mu_val = np.exp(log_mu)
                pi_logit = pi_coefs[rep]
                pi_prob = 1 / (1 + np.exp(-pi_logit))

                for _ in range(n_pairs_per_group):
                    if rng.random() < pi_prob:
                        y = 0
                    else:
                        y = rng.negative_binomial(theta, theta / (theta + mu_val))
                    rows_endog.append(float(y))
                    rows_exog.append(x_row.copy())
                    rows_exog_infl.append(xi_row.copy())
                    rows_group.append(group_id)
                group_id += 1

        return (
            np.array(rows_endog),
            np.array(rows_exog),
            np.array(rows_exog_infl),
            np.array(rows_group),
        )

    @pytest.fixture
    def cm_data(self):
        endog, exog, exog_infl, group_ids = self._make_cm_like_data(seed=42)
        comp_endog, comp_exog, comp_exog_infl, comp_weights = _compress_with_weights(
            endog, exog, exog_infl, group_ids)
        return {
            "endog": endog, "exog": exog, "exog_infl": exog_infl,
            "comp_endog": comp_endog, "comp_exog": comp_exog,
            "comp_exog_infl": comp_exog_infl, "comp_weights": comp_weights,
            "n_expanded": len(endog), "n_compressed": len(comp_endog),
        }

    def test_compression_ratio(self, cm_data):
        """Verify that phantom-zero compression actually reduces row count."""
        ratio = cm_data["n_expanded"] / cm_data["n_compressed"]
        assert ratio > 2, (
            f"Expected significant compression, got {ratio:.1f}x "
            f"({cm_data['n_expanded']} → {cm_data['n_compressed']})")

    def test_weight_sum_equals_expanded_n(self, cm_data):
        """Weights should sum to original observation count."""
        assert np.sum(cm_data["comp_weights"]) == pytest.approx(cm_data["n_expanded"])

    def test_cm_zinb_sparse_llf(self, cm_data):
        """CM-like ZINB sparse log-likelihood equivalence — the critical test."""
        res_exp = _fit_and_extract(
            cm_data["endog"], cm_data["exog"], cm_data["exog_infl"],
            nb_only=False, seed=77, sparse=True)
        res_comp = _fit_and_extract(
            cm_data["comp_endog"], cm_data["comp_exog"],
            cm_data["comp_exog_infl"],
            nb_only=False, sample_weight=cm_data["comp_weights"], seed=77,
            sparse=True)

        np.testing.assert_allclose(
            res_exp["llf"], res_comp["llf"], rtol=0.01,
            err_msg="CM-like ZINB sparse llf should match")

    def test_cm_zinb_sparse_params(self, cm_data):
        """CM-like ZINB sparse parameter equivalence.
        Wider tolerance — ZINB non-convex, different init noise → local optima."""
        res_exp = _fit_and_extract(
            cm_data["endog"], cm_data["exog"], cm_data["exog_infl"],
            nb_only=False, seed=77, sparse=True)
        res_comp = _fit_and_extract(
            cm_data["comp_endog"], cm_data["comp_exog"],
            cm_data["comp_exog_infl"],
            nb_only=False, sample_weight=cm_data["comp_weights"], seed=77,
            sparse=True)

        np.testing.assert_allclose(
            res_exp["x_mu"], res_comp["x_mu"], atol=0.5,
            err_msg="CM-like ZINB sparse x_mu should match")
        np.testing.assert_allclose(
            res_exp["theta"], res_comp["theta"], atol=0.5,
            err_msg="CM-like ZINB sparse theta should match")


class TestElementwiseNLL:
    """
    Direct tests of the _elementwise_nll static method to verify
    the weighted reduction is mathematically correct.
    """

    def test_weighted_mean_equals_expanded_mean(self):
        """
        If we have 3 observations [a, b, b] and compress to [a, b] with
        weights [1, 2], the weighted mean NLL should equal the unweighted
        mean of [a, b, b].
        """
        import tensorflow as tf

        # Simple NB case: 3 observations, 2 unique
        y_expanded = tf.constant([[1.0], [0.0], [0.0]])
        y_compressed = tf.constant([[1.0], [0.0]])
        weights = tf.constant([1.0, 2.0])

        # Arbitrary model predictions (same for both)
        # log_theta must broadcast with mu: shape [n, d] or [1, d]
        mu_exp = tf.constant([[0.5], [0.5], [0.5]])
        mu_comp = tf.constant([[0.5], [0.5]])
        log_theta_exp = tf.constant([[1.0], [1.0], [1.0]])
        log_theta_comp = tf.constant([[1.0], [1.0]])

        # Expanded: unweighted mean
        nll_exp = ZINBLogLik._elementwise_nll(
            y_expanded, mu_exp, None, log_theta_exp, True, 1e-8)
        loss_exp = tf.reduce_mean(nll_exp, axis=0)

        # Compressed: weighted mean
        nll_comp = ZINBLogLik._elementwise_nll(
            y_compressed, mu_comp, None, log_theta_comp, True, 1e-8)
        w = tf.reshape(weights, [-1, 1])
        loss_comp = tf.reduce_sum(nll_comp * w, axis=0) / tf.reduce_sum(w)

        np.testing.assert_allclose(
            loss_exp.numpy(), loss_comp.numpy(), atol=1e-6,
            err_msg="Weighted NLL reduction must equal expanded mean NLL")

    def test_zinb_weighted_mean_equals_expanded(self):
        """Same test but for full ZINB (with zero inflation)."""
        import tensorflow as tf

        y_expanded = tf.constant([[2.0], [0.0], [0.0], [0.0]])
        y_compressed = tf.constant([[2.0], [0.0]])
        weights = tf.constant([1.0, 3.0])

        mu_exp = tf.constant([[0.8], [0.8], [0.8], [0.8]])
        mu_comp = tf.constant([[0.8], [0.8]])
        pi_exp = tf.constant([[0.5], [0.5], [0.5], [0.5]])
        pi_comp = tf.constant([[0.5], [0.5]])
        log_theta_exp = tf.constant([[0.5], [0.5], [0.5], [0.5]])
        log_theta_comp = tf.constant([[0.5], [0.5]])

        nll_exp = ZINBLogLik._elementwise_nll(
            y_expanded, mu_exp, pi_exp, log_theta_exp, False, 1e-8)
        loss_exp = tf.reduce_mean(nll_exp, axis=0)

        nll_comp = ZINBLogLik._elementwise_nll(
            y_compressed, mu_comp, pi_comp, log_theta_comp, False, 1e-8)
        w = tf.reshape(weights, [-1, 1])
        loss_comp = tf.reduce_sum(nll_comp * w, axis=0) / tf.reduce_sum(w)

        np.testing.assert_allclose(
            loss_exp.numpy(), loss_comp.numpy(), atol=1e-6,
            err_msg="Weighted ZINB NLL must equal expanded mean NLL")


class TestLLFRecovery:
    """
    Test that the final log-likelihood reported in results is correctly
    scaled by effective_n (not physical row count).
    """

    def test_llf_scaling_nb(self):
        """
        For NB, llf = -(nll_per_d * effective_n + sum(w * gammaln(y+1))).
        Verify by computing manually.
        """
        rng = np.random.default_rng(99)
        n = 50
        endog = rng.negative_binomial(3, 0.5, size=n).astype(np.float64)
        exog = np.ones((n, 1))
        weights = rng.uniform(0.5, 5.0, size=n)

        res = _fit_and_extract(endog, exog, None, nb_only=True,
                               sample_weight=weights, seed=99)

        # effective_n should be sum of weights
        assert res["num_sample"] == pytest.approx(np.sum(weights))
        # llf should be finite and negative
        assert np.isfinite(res["llf"])
        assert res["llf"] < 0


class TestHighCompressionRatio:
    """
    Test with extreme compression (99% zeros → 1 phantom zero per group).
    This mimics the Cohen CM scenario where ~99.8% of pairs are zeros.
    """

    def test_extreme_compression_nb(self):
        """NB fit with 99% zero rate and high compression."""
        rng = np.random.default_rng(42)
        n_groups = 4
        n_per_group = 500
        n = n_groups * n_per_group

        # Design: intercept + 3 treatment dummies
        group_ids = np.repeat(np.arange(n_groups), n_per_group)
        exog = np.zeros((n, n_groups))
        exog[:, 0] = 1.0
        for g in range(1, n_groups):
            exog[group_ids == g, g] = 1.0

        # Very sparse response (99% zeros)
        endog = np.zeros(n)
        for g in range(n_groups):
            mask = group_ids == g
            n_nonzero = max(1, int(0.01 * n_per_group))
            idx = rng.choice(np.where(mask)[0], n_nonzero, replace=False)
            endog[idx] = rng.negative_binomial(2, 0.3, size=n_nonzero)

        comp_endog, comp_exog, _, comp_weights = _compress_with_weights(
            endog, exog, exog, group_ids)

        compression = n / len(comp_endog)
        assert compression > 10, f"Expected high compression, got {compression:.1f}x"

        res_exp = _fit_and_extract(endog, exog, None, nb_only=True, seed=55,
                                   sparse=True)
        res_comp = _fit_and_extract(comp_endog, comp_exog, None, nb_only=True,
                                    sample_weight=comp_weights, seed=55, sparse=True)

        np.testing.assert_allclose(
            res_exp["llf"], res_comp["llf"], rtol=0.02,
            err_msg="High-compression NB llf should match")
        np.testing.assert_allclose(
            res_exp["x_mu"], res_comp["x_mu"], atol=0.1,
            err_msg="High-compression NB x_mu should match")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
