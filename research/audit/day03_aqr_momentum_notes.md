# Fact, Fiction, and Momentum Investing Research

## About

Momentum is a factor which quantitative traders have disputed over for many years. This paper is here to clear up major misconceptions fabricated by those who do not believe in momentum as a valid and profitable factor.

## How the authors validate their findings

They use historical data dating back to the 1930s which helps them refute myths such as momentum's validity as a factor having decreased in recent years. Another way they refute these "myths" is by running backtests on exact claims such as "momentum cannot be used as a stand-alone factor and rather should be used as a screener." They then compare the results of the momentum-only strategy to value-based strategies, which are common conventions of those making these claims.

## Techniques I do not yet know how to implement

- **Mean-Variance Optimization**: Maximizing the Sharpe ratio of a combined portfolio of strategies and factors rather than picking the single best solo-performing factor. In the paper they find the optimal weighting of UMD as a function of its expected return. A negative correlation between two strategies means combining them will reduce the portfolio's volatility, even if a stand-alone strategy has an expected return of zero.
- **Factor Composite Construction**: Instead of picking a single method for determining momentum, it is better to average the value of several valid methods to reduce noise and guard against data mining. I do not yet know how to weight and combine signals systematically.
- **Factor Regression**: Running a regression of returns on market beta to compute market-adjusted alphas. This is how the authors determine whether the premium is compensation for risk or another phenomenon such as human behavioral tendencies.

## How this connects back to my current forex strategy

My current strategy uses a single method of representing a factor that at its core is momentum. The definition given by the paper of momentum is "the phenomenon that securities that have performed well relative to peers on average continue to outperform, and securities which have performed relatively poorly tend to continue to underperform" (Asness et al., 2014). This is the underlying logic of the break-of-structure component that serves as the last signal before entry into a trade. My current methodology for representing momentum goes against the methods discussed in the paper with regard to factor composite construction. By using a single factor for momentum, I risk introducing noise into my system and accidentally overfitting to the data the backtest is being run on.