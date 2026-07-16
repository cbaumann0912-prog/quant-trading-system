## Note — Block length selection (deferred)

`block_bootstrap` (src/evaluation/bootstrap.py) does not select block
length automatically — block_size is a required argument the caller must
choose.

Literature reference for a data-driven approach: Politis & White (2004),
"Automatic Block-Length Selection for the Dependent Bootstrap," with
correction in Patton, Politis & White (2009). Estimates optimal block
length from the series' own autocorrelation/spectral structure rather than
a fixed constant; scales asymptotically like O(n^(1/3)) for block
bootstrap of the mean, with the leading constant estimated from the data.

Caveat: this method assumes stationarity. Given Day 5's finding of
regime-dependent correlation structure in these three forex pairs, an
automatically-selected block length would still be a stationarity-blind
default, not a guarantee of correctness across regime breaks.

Status: not implemented. Candidate for a buffer block, put off for a later
date, or as a limitations footnote in the paper, also put off for a later
date, rather than in-scope for Day 37.