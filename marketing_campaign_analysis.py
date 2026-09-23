"""
Marketing Campaign for a Restaurant Chain
==========================================
EDA on the Zomato Restaurants in India dataset to understand customer
preferences, dining trends, and the competitive landscape across regions
of India, in order to design an effective marketing campaign for a
restaurant chain.

Usage:
    python marketing_campaign_analysis.py

Requires `zomato_restaurants_in_India.csv` to be in the same directory
(or update CSV_PATH below).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

CSV_PATH = "zomato_restaurants_in_India.csv"


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
def load_data(path=CSV_PATH):
    df = pd.read_csv(path)
    print(df.head(5))
    return df


# ---------------------------------------------------------------------------
# 2. Data cleaning and preparation
# ---------------------------------------------------------------------------
def clean_data(df):
    print("\nMissing values before cleaning:")
    print(df.isna().sum())

    # Fill missing addresses using reverse geocoding on lat/long
    try:
        from geopy.geocoders import Nominatim

        geolocator = Nominatim(user_agent="my_geocoder")

        def get_missing_addresses(row):
            if pd.isnull(row["address"]):
                location = geolocator.reverse(
                    (row["latitude"], row["longitude"]), exactly_one=True
                )
                return location.address if location else None
            return row["address"]

        df["address"] = df.apply(get_missing_addresses, axis=1)
    except Exception as e:
        print(f"Skipping reverse geocoding (geopy unavailable or failed): {e}")

    # Missing cuisines filled with "Other"
    df["cuisines"].fillna("Other", inplace=True)

    # Missing timings filled with the most common timing
    most_common_timing = df["timings"].mode()[0]
    df["timings"].fillna(most_common_timing, inplace=True)

    # Drop zipcode (mostly missing, not needed for this analysis)
    if "zipcode" in df.columns:
        df.drop(columns=["zipcode"], inplace=True)

    # Missing opentable_support filled with 0
    df["opentable_support"].fillna(0, inplace=True)

    # Fix mis-encoded en-dash in timings
    df["timings"] = df["timings"].str.replace("â€“", "to")

    print("\nMissing values after cleaning:")
    print(df.isnull().sum())
    return df


# ---------------------------------------------------------------------------
# 3. Descriptive statistics
# ---------------------------------------------------------------------------
def descriptive_statistics(df):
    numeric_columns = ["average_cost_for_two", "aggregate_rating", "votes", "photo_count"]
    print("\nDescriptive statistics:")
    print(df[numeric_columns].describe())

    fig, axes = plt.subplots(nrows=2, ncols=len(numeric_columns), figsize=(15, 8))
    for i, col in enumerate(numeric_columns):
        sns.histplot(df[col], ax=axes[0, i], kde=True)
        axes[0, i].set_title(f"{col} (Skewness: {df[col].skew():.2f})")

    for i, col in enumerate(numeric_columns):
        sns.histplot(y=df[col], ax=axes[1, i])
        axes[1, i].set_title(f"{col} (Kurtosis: {df[col].kurtosis():.2f})")

    plt.tight_layout()
    plt.show()

    categorical_columns = ["establishment", "city", "locality"]
    print(df[categorical_columns].apply(lambda x: x.value_counts()))

    boolean_columns = ["opentable_support", "delivery", "takeaway"]
    print(df[boolean_columns].apply(pd.Series.value_counts))


# ---------------------------------------------------------------------------
# 4. Distribution analysis
# ---------------------------------------------------------------------------
def distribution_analysis(df):
    # Ratings distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df["aggregate_rating"], kde=True, color="green")
    plt.title("Distribution of Aggregate Ratings")
    plt.xlabel("Aggregate Rating")
    plt.ylabel("Frequency")
    plt.show()

    # Price range distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(x="price_range", data=df, palette="viridis")
    plt.title("Distribution of Price Range")
    plt.xlabel("Price Range")
    plt.ylabel("Frequency")
    plt.show()

    # Top 10 cuisines distribution
    top_cuisines = (
        df["cuisines"].str.split(", ", expand=True).stack().value_counts().head(10)
    )
    plt.figure(figsize=(10, 6))
    top_cuisines.plot(kind="bar", color="salmon")
    plt.title("Top 10 Cuisines")
    plt.xlabel("Cuisine")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45, ha="right")
    plt.show()


# ---------------------------------------------------------------------------
# 5. Correlation analysis
# ---------------------------------------------------------------------------
def correlation_analysis(df):
    correlation_matrix = df.corr(numeric_only=True)
    columns_to_hide = ["country_id", "opentable_support", "takeaway"]
    columns_to_hide = [c for c in columns_to_hide if c in correlation_matrix.columns]
    correlation_matrix_subset = correlation_matrix.drop(
        columns=columns_to_hide, index=columns_to_hide
    )

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix_subset, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix")
    plt.show()


# ---------------------------------------------------------------------------
# 6. Regional analysis
# ---------------------------------------------------------------------------
def regional_analysis(df):
    region_groups = df.groupby("city")
    city_counts = region_groups.size().sort_values(ascending=False)
    top_10_cities = city_counts.head(10).index

    df_top_10 = df[df["city"].isin(top_10_cities)]
    region_groups_top_10 = df_top_10.groupby("city")
    cuisine_counts_top_10 = (
        region_groups_top_10["cuisines"].value_counts().unstack().fillna(0)
    )

    # Pie chart: distribution across top 10 cities
    plt.figure(figsize=(12, 8))
    colors = plt.cm.tab20.colors
    cuisine_counts_top_10.sum(axis=1).plot(
        kind="pie", autopct="%1.1f%%", colors=colors
    )
    plt.title("Distribution of Cuisine Types Across Top 10 Cities")
    plt.ylabel("")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 7. Customer preference analysis
# ---------------------------------------------------------------------------
def customer_preference_analysis(df):
    region_groups = df.groupby("city")
    cuisine_counts = region_groups["cuisines"].value_counts().unstack().fillna(0)
    cuisine_counts_sum = cuisine_counts.sum(axis=0)
    top_10_cities = cuisine_counts_sum.nlargest(10).index
    cuisine_counts_top_10 = cuisine_counts[top_10_cities]
    cuisine_counts_sum_top_10 = cuisine_counts_top_10.sum(axis=1)

    plt.figure(figsize=(12, 8))
    sns.barplot(
        x=cuisine_counts_sum_top_10.index,
        y=cuisine_counts_sum_top_10.values,
        palette="viridis",
    )
    plt.title("Distribution of Cuisine Types Across Top 10 Cities")
    plt.xlabel("Cuisine Type")
    plt.ylabel("Total Count Across Top 10 Cities")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    # Ratings vs popularity (votes)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x="aggregate_rating", y="votes", data=df)
    plt.title("Relationship between Restaurant Ratings and Popularity")
    plt.xlabel("Aggregate Rating")
    plt.ylabel("Number of Votes")
    plt.tight_layout()
    plt.show()

    # Ratings across price ranges
    plt.figure(figsize=(8, 6))
    sns.barplot(x="price_range", y="aggregate_rating", data=df)
    plt.title("Restaurant Ratings Across Price Ranges")
    plt.xlabel("Price Range")
    plt.ylabel("Aggregate Rating")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 8. Competitive analysis
# ---------------------------------------------------------------------------
def competitive_analysis(df):
    region_groups = df.groupby("city")
    competitors = region_groups.apply(lambda x: x.nlargest(5, "aggregate_rating"))

    top_10_cities = competitors["city"].value_counts().head(10).index
    competitors_top_10 = competitors[competitors["city"].isin(top_10_cities)]

    plt.figure(figsize=(10, 6))
    sns.barplot(x="city", y="aggregate_rating", data=competitors_top_10)
    plt.title("Strengths and Weaknesses of Competitors based on Ratings (Top 10 Cities)")
    plt.xlabel("City")
    plt.ylabel("Aggregate Rating")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    # Cuisine distribution among top competitors
    cuisine_counts = competitors["cuisines"].value_counts()
    top_10_cuisines = cuisine_counts.head(10).index
    competitors_top_10_cuisines = competitors[
        competitors["cuisines"].isin(top_10_cuisines)
    ]

    plt.figure(figsize=(10, 6))
    sns.countplot(x="cuisines", data=competitors_top_10_cuisines, order=top_10_cuisines)
    plt.title("Distribution of Top 10 Cuisines among Competitors")
    plt.xlabel("Cuisine Type")
    plt.ylabel("Number of Restaurants")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    # Pricing range of competitors
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="average_cost_for_two", y="city", data=competitors)
    plt.title("Pricing Range of Competitors")
    plt.xlabel("Average Cost for Two")
    plt.ylabel("City")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 9. Market gap analysis
# ---------------------------------------------------------------------------
def market_gap_analysis(df):
    plt.figure(figsize=(8, 6))
    sns.boxplot(x="price_range", y="average_cost_for_two", data=df)
    plt.title("Average Cost for Two Across Price Ranges")
    plt.xlabel("Price Range")
    plt.ylabel("Average Cost for Two")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    df = load_data()
    df = clean_data(df)
    descriptive_statistics(df)
    distribution_analysis(df)
    correlation_analysis(df)
    regional_analysis(df)
    customer_preference_analysis(df)
    competitive_analysis(df)
    market_gap_analysis(df)


if __name__ == "__main__":
    main()
