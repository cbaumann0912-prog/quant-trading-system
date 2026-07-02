import numpy as np

from src.evaluation.significance import bonferroni_correction, benjamini_hochberg_correction

sample = [0.00448,0.39341,0.53882,
          0.00671,0.01220,0.98617,
          0.58125,0.00017,0.00907,
          0.33626]

tiny_sample = [0.00001,0.00007,0.00004,0.00009,0.00002]

def test_bonferroni_more_conservative_than_bh():
    result_bf = bonferroni_correction(sample,0.05)
    result_bh = benjamini_hochberg_correction(sample,0.05)

    assert sum(result_bf) < sum(result_bh)

def test_bh_known_example():
    result = benjamini_hochberg_correction(sample,0.05)
    assert result == [True,False,False,True,True,False,False,True,True,False]

def test_all_rejected_when_all_tiny():
    result_bf = bonferroni_correction(tiny_sample,0.05)
    result_bh = benjamini_hochberg_correction(tiny_sample,0.05)

    assert sum(result_bf) == len(result_bf)
    assert sum(result_bh) == len(result_bh)