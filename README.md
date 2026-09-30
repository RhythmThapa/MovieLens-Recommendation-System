# MovieLens 1M Recommendation System

Rating prediction and top-N movie recommendations on the MovieLens 1M dataset (1,000,209 ratings, 6,040 users, 3,883 movies).

## Models compared
Evaluated on the same 10k test sample (80/20 split, `random_state=42`):

| Model | MAE | RMSE |
|---|---|---|
| User-Item Bias | 0.7218 | **0.9108** |
| Item-Based CF (cosine, k=20) | 0.7212 | 0.9320 |
| Popularity-Based (shrunk movie mean) | 0.7887 | 0.9851 |
| Global Mean Baseline | 0.9358 | 1.1225 |

**Final model:** User-Item Bias, full test set (200,042 ratings): **MAE 0.7223, RMSE 0.9120**.

## Structure
```
notebooks/   MovieLens_Recommender.ipynb
models/      recommendation_model.joblib (saved by the notebook)
results/     comparison table, final metrics, EDA plots
requirements.txt
```

## Run it
1. Download MovieLens 1M (`ml-1m.zip`) from https://grouplens.org/datasets/movielens/1m/ and put it in `notebooks/`.
2. `pip install -r requirements.txt`
3. Run the notebook top to bottom. It saves the model to `models/`.

## Use the saved model
```python
import joblib
a = joblib.load("models/recommendation_model.joblib")
pred = a["global_mean"] + a["user_bias"].get(1, 0) + a["item_bias"].get(1193, 0)
print(min(max(pred, 1), 5))
```

## Notes
- Random split, so a user's other ratings appear in training. A time-based split would be stricter.
- New users fall back to popular movies (cold start).
