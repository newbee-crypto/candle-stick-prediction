# Candlestick ViT Stock Trend Prediction Report

## Labeling Strategy
Images are labeled from the future return in the prepared dataset. The project labels are mapped as `0=Down`, `1=Neutral`, and `2=Up`. If using `D:\full-stack2\data\labels\labeled_dataset.csv`, the label file includes a volatility-adjusted `trend_score`; the earlier config uses thresholds `Up >= 0.5`, `Down <= -0.5`, and values between them as `Neutral`.

## Model
The trained model in `best_model.pth` uses `vit_small_patch16_224` as the image encoder. Its ViT embedding is concatenated with three technical features: RSI, MACD, and trend_score, then classified into Down, Neutral, or Up.

## Test Evaluation
- Accuracy: 0.7363
- Macro F1: 0.7285
- Confusion matrix order: Down, Neutral, Up
- Confusion matrix: `[[135, 98, 4], [11, 319, 138], [0, 57, 406]]`

## XAI
Grad-CAM overlays were generated from the last transformer block. Bright regions show chart patches that most influenced the selected class score. Use the saved images in `outputs/xai` to discuss whether the model focuses on recent candle bodies, upper wicks, lower wicks, or volume/lower chart regions.

## Comparative Study
Classical rule checks were applied to the final candle of each chart window: Doji, Hammer, Bullish Engulfing, and Bearish Engulfing.

- Rows with any classical pattern: 258
- Classical signal agreement with ground truth: 0.3527
- Classical signal agreement with ViT prediction: 0.3682
- Pattern counts: {'Hammer': 58, 'None': 910, 'Doji': 129, 'Bullish Engulfing': 34, 'Bearish Engulfing': 45}

## Discussion
Deep learning can rediscover some classical candlestick ideas because candle bodies, wicks, gaps, and local reversals are visible image structures. However, ViT is not limited to named patterns: it can combine weak signals across many candles, technical indicators, and relative position in the chart. Classical rules are interpretable but rigid; the ViT may find softer or multi-candle structures that do not have a textbook name. The XAI overlays should be treated as evidence of focus, not as full proof of causality.
