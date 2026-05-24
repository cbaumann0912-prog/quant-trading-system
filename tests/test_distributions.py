import pytest
import numpy as np
import scipy
from src.stats.distributions import (normal_pdf,normal_cdf,lognormal_pdf, simulate_price_path, student_t_pdf, tail_mass_comparison)

def test_normal_pdf_standard():
    assert abs(normal_pdf(0, 0, 1) - 0.3989422) < 1e-6


def test_normal_cdf_symmetry():
    assert abs(normal_cdf(0, 0, 1) - 0.5) < 1e-4


def test_normal_cdf_known_value():
    assert abs(normal_cdf(1.96, 0, 1) - 0.975) < 1e-3


def test_lognormal_pdf_positive_only():
    with pytest.raises(ValueError):
        lognormal_pdf(-1, 0, 1)


def test_simulate_price_path_shape():
    path = simulate_price_path(S0=100, mu=0.0001, sigma=0.01, n=253)
    assert path.shape == (253,)


def test_simulate_price_path_starts_at_S0():
    path = simulate_price_path(S0=100, mu=0.0001, sigma=0.01, n=253)
    assert path[0] == 100

def test_tail_mass_returns_dict_with_correct_keys():
    result = tail_mass_comparison(scipy.stats.norm.pdf, scipy.stats.expon.pdf, 8)
    assert set(result.keys()) == {"dist1_tail_mass", "dist2_tail_mass", "difference"}

def test_t_pdf_heavier_tails_than_normal():
    normal_pdf = lambda x: scipy.stats.norm.pdf(x,0,1)
    t_pdf = lambda x: student_t_pdf(x,28,0,1)
    result = tail_mass_comparison(normal_pdf, t_pdf, 8)
    assert result["difference"] > 0