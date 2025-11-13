## Project summary

- This is a student's learning project exploring simple ML models for price movement prediction on crypto assets.
- The dataset is real and was collected via web scraping.
- Data structure: a collection of watch sessions on different coins, split into ten-minute chunks of tick data.
- Each ten-minute chunk is recorded with metadata and grouped under a unique `watch_id`.

## File structure

- `main.py`: Orchestrates running all models and prints a final summary.
- `model_cnn.py`: CNN model that predicts if price will rise by TARGET_GAIN and hold for MIN_HOLD_TICKS after an initial window. Returns metrics in a dict and prints results.
- `model_mlp.py`: MLP model with same goal as CNN. Returns metrics in a dict and prints results.
- `model_random_forest.py`: Random Forest model with same goal at CNN using windowed price changes and metadata. Returns metrics in a dict and prints results and feature importance.
- `model_rnn_simple.py`: Runs precompute first, then trains an RNN to predict per-tick probability (>2% over entry). Returns metrics in a dict and prints results.
- `data/watch_tracking.csv`: Session-level metadata per `watch_id` (10-minute chunks).
- `data/price_history.csv`: Tick-level prices aligned to `watch_id`.
- `precomputed_sessions.npz`: Saved session tensors used by the RNN pipeline (auto-generated).
- `README.md`: Project overview, data description, and file map.

## Observations and Results

### RNN (rnn_simple.py)
Goal: Sequence model predicting, for each tick, the probability that price exceeds entry by >2% at that tick.

Output:

```
Label distribution: {0.0: 361183, 1.0: 205725}
Pos weight: 1.755659253858306
Epoch 1: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 800/800 [03:37<00:00,  3.68it/s]
Epoch 1, Total Loss: 717.8496, Train Accuracy: 59.81%
Epoch 2: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 800/800 [03:21<00:00,  3.98it/s]
Epoch 2, Total Loss: 736.2807, Train Accuracy: 62.99%
Epoch 3: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 800/800 [03:18<00:00,  4.02it/s]
Epoch 3, Total Loss: 743.1751, Train Accuracy: 63.02%
Epoch 4: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 800/800 [03:10<00:00,  4.19it/s]
Epoch 4, Total Loss: 736.1082, Train Accuracy: 60.38%
Epoch 5: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 800/800 [03:20<00:00,  4.00it/s] 
Epoch 5, Total Loss: 739.3183, Train Accuracy: 62.03%
Epoch 6: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 800/800 [03:26<00:00,  3.87it/s] 
Epoch 6, Total Loss: 736.0037, Train Accuracy: 62.07%
Epoch 7: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 800/800 [03:10<00:00,  4.19it/s] 
Epoch 7, Total Loss: 738.8810, Train Accuracy: 60.42%
Epoch 8: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 800/800 [03:19<00:00,  4.00it/s] 
Epoch 8, Total Loss: 738.1798, Train Accuracy: 61.76%
Epoch 9: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 800/800 [03:13<00:00,  4.14it/s] 
Epoch 9, Total Loss: 743.4821, Train Accuracy: 62.12%
Epoch 10: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 800/800 [03:13<00:00,  4.13it/s] 
Epoch 10, Total Loss: 738.2664, Train Accuracy: 61.74%
Epoch 11: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 800/800 [03:14<00:00,  4.12it/s] 
Epoch 11, Total Loss: 736.3553, Train Accuracy: 60.67%
Epoch 12: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 800/800 [03:23<00:00,  3.93it/s] 
Epoch 12, Total Loss: 738.5010, Train Accuracy: 62.96%
Epoch 13: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 800/800 [03:00<00:00,  4.43it/s] 
Epoch 13, Total Loss: 730.1842, Train Accuracy: 61.59%
Epoch 14: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 800/800 [03:01<00:00,  4.42it/s] 
Epoch 14, Total Loss: 736.6676, Train Accuracy: 61.57%
Epoch 15: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 800/800 [02:54<00:00,  4.58it/s] 
Epoch 15, Total Loss: 745.3458, Train Accuracy: 62.14%
Test Accuracy: 61.65%
Average predicted confidence (sigmoid output): 0.4017
```

Observations:
- Heavy class imbalance; train accuracy hovers near 60–63%.
- Training is slow due to sequence length and per-timestep outputs.
- Test accuracy around ~61%; average confidence relatively low.


### CNN (cnn.py)
Goal: Given an initial window and metadata, predict whether price rises by TARGET_GAIN and holds for MIN_HOLD_TICKS after the window.

Output:

```
Epoch 53 - Validation loss: 0.1556
Epoch 54 - Validation loss: 0.1555
Early stopping triggered
Accuracy: 0.5450
              precision    recall  f1-score   support

         0.0     0.5260    0.8182    0.6403        99
         1.0     0.6087    0.2772    0.3810       101

    accuracy                         0.5450       200
   macro avg     0.5673    0.5477    0.5106       200
weighted avg     0.5677    0.5450    0.5093       200

Probability stats - mean: 0.5031, min: 0.0238, max: 0.6580
Average Precision Score: 0.6115
ROC AUC Score: 0.6772
PR AUC Score: 0.6036

Correct Predictions: 109 / 200 (54.5%)
```

Observations:
- Overall accuracy ~54.5%; better recall for class 0 than class 1.
- AUCs in the mid‑0.6 range indicate some separation.
- Score range is narrow (max ~0.66), suggesting cautious predictions.


### MLP (mlp.py)
Goal: Same as CNN- predict sustained TARGET_GAIN after the observation window using only the initial window and metadata.

Output:

```
Epoch 10 - Validation loss: 0.7261
Epoch 11 - Validation loss: 0.7239
Early stopping triggered
Accuracy: 0.6400
              precision    recall  f1-score   support

         0.0     0.6875    0.4583    0.5500        72
         1.0     0.6176    0.8077    0.7000        78

    accuracy                         0.6400       150
   macro avg     0.6526    0.6330    0.6250       150
weighted avg     0.6512    0.6400    0.6280       150

Probability stats - mean: 0.5511, min: 0.4424, max: 0.9071
Average Precision Score: 0.6700
ROC AUC Score: 0.6677
PR AUC Score: 0.6657

Correct Predictions: 96 / 150 (64.0%)
```

Observations:
- Accuracy ~64% with higher recall for class 1.
- Probability mean above 0.5; scores extend up to ~0.90.
- Early stopping at epoch 11; faster training than sequence models.


### Random Forest (random_forest.py)
Goal: Using LOOKBACK_TICKS, momentum, and metadata, classify if perc_change_from_entry will increase over the next FORECAST_HORIZON ticks.

Output:

```
Random Forest Classification Model
Lookback ticks: 30
Forecast horizon: 10
Meta features: ['liq', 'mcap', 'vol', 'm5', 'h1', 'h6', 'h24']
Splitting data...
Training set: 421,196 samples
Test set: 105,300 samples

Training Random Forest classifier...
Model training complete.
Evaluating model...
Model performance
Accuracy: 0.6271
Classification report:
              precision    recall  f1-score   support

           0     0.7801    0.5224    0.6258     62846
           1     0.5252    0.7820    0.6284     42454

    accuracy                         0.6271    105300
   macro avg     0.6527    0.6522    0.6271    105300
weighted avg     0.6773    0.6271    0.6268    105300

Probability stats - mean: 0.4903, min: 0.0000, max: 0.8621
Average Precision Score: 0.6084
ROC AUC Score: 0.7136
PR AUC Score: 0.6084
Feature importance
tick_1         : 0.0159
tick_2         : 0.0133
tick_3         : 0.0129
tick_4         : 0.0110
tick_5         : 0.0104
tick_6         : 0.0106
tick_7         : 0.0103
tick_8         : 0.0105
tick_9         : 0.0097
tick_10        : 0.0100
tick_11        : 0.0092
tick_12        : 0.0133
tick_13        : 0.0120
tick_14        : 0.0173
tick_15        : 0.0133
tick_16        : 0.0140
tick_17        : 0.0153
tick_18        : 0.0105
tick_19        : 0.0106
tick_20        : 0.0134
tick_21        : 0.0141
tick_22        : 0.0157
tick_23        : 0.0144
tick_24        : 0.0132
tick_25        : 0.0102
tick_26        : 0.0187
tick_27        : 0.0189
tick_28        : 0.0207
tick_29        : 0.0265
tick_30        : 0.0349
momentum       : 0.0968
liq            : 0.0415
mcap           : 0.0360
vol            : 0.0446
m5             : 0.0601
h1             : 0.1725
h6             : 0.0390
h24            : 0.0496
day_of_week    : 0.0114
hour_of_day    : 0.0181
Model training completed successfully.
```

Observations:
- Accuracy ~62.7% on a large test set.
- Class 0 shows higher precision; class 1 shows higher recall.
- Momentum and short-term ticks contribute strongly per importance.



## Overall conclusions
- Results are modest across all models and unreliable in a live setting.
- The problem (predicting crypto price movement) is complex, and these simple models caught only small patterns.
- Treat these runs as the learning exercises of a student, not something to use live.
- Another note: due to github file size limitations, the scope of the data set has been dramatically reduced. Poor results reflect this.
