# Marketing Campaign for a Restaurant Chain — EDA on Zomato India Data

Exploratory data analysis (EDA) of the [Zomato Restaurants in India](https://www.kaggle.com/datasets/abhijitdahatonde/zomato-restaurants-in-india) dataset, aimed at understanding customer preferences, dining trends, and the competitive landscape across Indian cities to design an effective marketing campaign for a restaurant chain.

## Objective

To utilize EDA skills to understand customer preferences, dining trends, and the competitive landscape in various regions of India, and to design an effective marketing campaign for a restaurant chain.

## Dataset

- **File:** `zomato_restaurants_in_India.csv`
- **Rows / Columns:** ~211,944 restaurants × 26 columns
- **Key fields:** `name`, `establishment`, `city`, `locality`, `latitude`, `longitude`, `cuisines`, `timings`, `average_cost_for_two`, `price_range`, `aggregate_rating`, `rating_text`, `votes`, `photo_count`, `opentable_support`, `delivery`, `takeaway`

> The raw CSV is not included in this repo — download it from Kaggle and place it in the project root before running the script.

## What the analysis covers

1. **Data cleaning & preparation**
   - Missing addresses filled via reverse geocoding (lat/long) with `geopy`
   - Missing `cuisines` filled with `"Other"`; missing `timings` filled with the most common value
   - Dropped the mostly-empty `zipcode` column
   - Missing `opentable_support` filled with `0`
   - Fixed mis-encoded en-dash characters in `timings`

2. **Descriptive statistics** — summary stats and skewness/kurtosis for cost, rating, votes, and photo count

3. **Distribution analysis** — aggregate rating, price range, and top 10 cuisines

4. **Correlation analysis** — heatmap of numeric features (cost, price range, rating, votes, photo count, delivery)

5. **Regional analysis** — cuisine distribution across the top 10 cities by restaurant count

6. **Customer preference analysis** — cuisine popularity across cities, rating vs. popularity (votes), and rating vs. price range

7. **Competitive analysis** — top-rated competitors per city, their cuisines, and pricing ranges

8. **Market gap analysis** — cost distribution across price ranges to spot underserved price tiers

## Key findings

- Ratings are left-skewed with a large spike at 0 (unrated restaurants); genuinely rated restaurants cluster around 3.5–4.2.
- `average_cost_for_two`, `votes`, and `photo_count` are all heavily right-skewed with high kurtosis — a small number of restaurants drive most cost, popularity, and engagement.
- **North Indian, Chinese, and Fast Food** are the most common cuisines nationwide.
- `average_cost_for_two` and `price_range` are strongly correlated (0.79), and higher price ranges correlate with higher ratings.
- Chennai, Mumbai, and New Delhi have the largest share of restaurants among the top 10 cities analyzed.
- Higher-rated restaurants tend to receive disproportionately more votes, suggesting a strong popularity/rating feedback loop worth targeting in campaign design.

## Repository contents

| File | Description |
|---|---|
| `Marketing_Campaign_for_a_Restaurant_Chain.ipynb` | Original Jupyter notebook (source of this analysis) |
| `marketing_campaign_analysis.py` | Standalone Python script version of the analysis |
| `README.md` | This file |

## Requirements

```
pandas
numpy
matplotlib
seaborn
geopy
```

Install with:

```bash
pip install pandas numpy matplotlib seaborn geopy
```

## Usage

1. Download `zomato_restaurants_in_India.csv` and place it in the project root.
2. Run the script:

```bash
python marketing_campaign_analysis.py
```

Charts will be displayed via matplotlib as the script runs.

## License

This project is provided for educational and portfolio purposes. The underlying dataset is subject to its own license on Kaggle.
