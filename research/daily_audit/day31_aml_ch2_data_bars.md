# Day 31 — AML Ch. 2: Financial Data Bars

## Methodology
Reviewed time, tick, volume, and dollar bar sampling schemes from AML Chapter 2 and evaluated their statistical properties and practicality for the current forex research framework using 1-minute OHLCV data.

## Findings
Time bars sample the market at fixed time intervals, while tick, volume, and dollar bars sample based on market activity instead of clock time. Activity-based bars better align observations with information flow, often producing return series with lower autocorrelation and more stable statistical properties. However, constructing true dollar or volume bars requires higher-frequency transaction data than is currently available.

## Interpretation
True dollar bars cannot be constructed from the current 1-minute OHLCV dataset because the underlying tick-by-tick trading activity is unavailable. In addition, the volume provided for spot FX is only a proxy rather than a centralized measure of market volume, making volume- and dollar-based bars less reliable than they are for equities. Given these data limitations, time bars remain the most appropriate choice for the current forex framework, while activity-based bars are an important direction to revisit if higher-frequency tick data becomes available.