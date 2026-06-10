# Day 18 Research Audit — Eigendecomposition of Forex Covariance Matrix

## 1. Question Investigated
What does the eigenstructure of the EUR/USD, GBP/USD, USD/JPY return covariance matrix reveal about the dominant sources of variance in the forex data?

## 2. Why It Matters
Three pairs, but  there are not three independent risk sources. The eigenstructure tells you how many truly independent dimensions the data occupies. Understanding the eigenstructure and what it is conveying can allow for economic understanding of what specific large contributing factors are. Signals are currently built on raw EUR/USD, GBP/USD, and USD/JPY returns, but these series are not independent. Then by projecting returns into principal component space you can obtaian orthogonal PC return series, allowing signals built on a singular factor to operate without unintentionally reacting to another.

## 3. Methodology
- 15 years of 1-minute OHLCV resampled to daily close
- Log returns computed as ln(Pₜ / Pₜ₋₁)
- Series aligned on common trading dates via pd.concat().dropna()
- Covariance matrix constructed from aligned return DataFrame
- Eigendecomposition via hand-written eigendecomposition() in src/features/pca.py, which wraps np.linalg.eigh with descending sort by eigenvalue magnitude
- Variance explained computed as λᵢ / Σλ

## 4. Assumptions
- Log returns are stationary over the full 15-year sample
- Covariance matrix is stable across the sample period (stationarity of second moments — not yet tested)
- Daily close-to-close returns are representative of the return generating process

## 5. Hypothesis
PC1 is a USD strength factor. Because EUR/USD and GBP/USD both have USD on the right side of the quote, USD appreciation causes both pairs to fall simultaneously. USD/JPY has USD on the left side, so USD appreciation causes it to rise. Therefore:
- PC1 loadings: EURUSD and GBPUSD carry the same sign; USDJPY carries the opposite sign
- PC1 variance explained: > 50%, as USD is the dominant driver of co-movement across all three pairs
- PC2 is expected to capture EUR vs GBP divergence — a European currency factor orthogonal to USD

## 6. Findings

### 6.1 Eigenvalues and Variance Explained
| Component | Eigenvalue | Variance Explained | Cumulative |
|-----------|------------|--------------------|------------|
| PC1       |4.44795359e-05|0.59575426|0.59575426|
| PC2       |2.15918410e-05|0.28919886|0.88495312|
| PC3       |8.58950113e-06|0.11504688|1|

### 6.2 Eigenvector Loadings
| Pair   | PC1 | PC2 | PC3 |
|--------|-----|-----|-----|
| EURUSD |-0.57931132|0.22606286|-0.78313088|
| GBPUSD |-0.61705282|0.50613031|0.60255949|
| USDJPY |0.53258259|0.83230265|-0.15371429|

### 6.3 Hypothesis Evaluation
Hypothesis is patially confirmed. PC1 loadings show EURUSD and GBPUSD with the same sign, and USDJPY with the opposite sign. This is consistent with a USD strength factor. USD appreciation pushes EUR/USD and GBP/USD down while pushing USD/JPY up. The initial hypothesis that PC2 captures EUR vs GBP divergence was not supported — GBP loads positively on PC2 but so does EUR/USD. This makes a clean EUR/GBP divergence reading inconsistent with the data. PC2 loadings show all three pairs positive with USDJPY dominant. After extracting the USD strength component, this residual factor is consistent with a global risk-off / JPY safe-haven dynamic. This pattern is consistent with carry trade unwinding causing simultaneous JPY appreciation and broad currency weakness against USD. Empirical confirmation requires verifying that PC2 scores spike during documented carry crash events. The 0.596 explained variance means that approx 59.6% of the variability in the log returns of the 3 forex pairs is due to the USD factor.

## 7. Alternative Explanations
The observed eigenstructure may reflect broader risk-on/risk-off dynamics rather than a pure USD factor. Additionally, major market stress periods within the 15-year sample, such as the Global Financial Crisis and COVID-19, could disproportionately influence the covariance structure and amplify the importance of the first principal component. Rolling-window analysis is required to determine whether these factors are stable across regimes or sample-specific.

## 8. Open Questions
- Does the eigenstructure remain stable across rolling sub-windows, or do the dominant factors rotate across regimes? 
- Is the PC1 loading pattern consistent with Lustig/Roussanov/Verdelhan dollar factor construction?
- Does mean-centering the returns before covariance construction materially change the eigenvalues? (Returns are near-zero mean — likely negligible but unverified)

## 9. Connection to Strategy Development
Because PC1 explains approximately 59.6% of total variance, signals constructed independently on EUR/USD, GBP/USD, and USD/JPY are likely to contain substantial overlap through their shared exposure to the dominant USD factor. Future strategy development should therefore evaluate signals in principal component space, where orthogonal factors allow for cleaner attribution of predictive power and reduce the risk of unintentionally concentrating exposure to the same underlying driver. 

PC2's characteristic of all three pairs being positive and USDJPY is dominant is consistent with the carry crash factor documented in Brunnermeier, Nagel, and Pedersen (2008). Any carry-adjacent strategy candidate must account for this  risk.