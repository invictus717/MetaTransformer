from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path("/Users/javier/Projects/tabulardl-benchmark/")
RAW_DATA_DIR = ROOT_DIR / "raw_data/"

colnames = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]
colnames = [c.replace("-", "_") for c in colnames]
adult_train = pd.read_csv(RAW_DATA_DIR / "adult/adult.data", names=colnames)
adult_test = pd.read_csv(RAW_DATA_DIR / "adult/adult.test", skiprows=1, names=colnames)
adult = pd.concat([adult_train, adult_test])
adult = adult.replace(to_replace=" ?", value=np.nan).dropna()
adult_rows, adult_cols = adult.shape
adult["target"] = (adult["income"].apply(lambda x: ">50" in x)).astype(int)
adult.drop("income", axis=1, inplace=True)
pos, neg = adult.target.value_counts().values
adult_neg_pos_ratio = round(neg / pos, 4)


bank_marketing = pd.read_csv(
    RAW_DATA_DIR / "bank_marketing/bank-additional-full.csv", sep=";"
)
bank_marketing.drop("duration", axis=1, inplace=True)
bank_marketing_rows, bank_marketing_cols = bank_marketing.shape
bank_marketing["target"] = (bank_marketing["y"].apply(lambda x: x == "yes")).astype(int)
bank_marketing.drop("y", axis=1, inplace=True)
pos, neg = bank_marketing.target.value_counts().values
bank_marketing_rows_neg_pos_ratio = round(neg / pos, 4)

# airbnb_raw = pd.read_csv(RAW_DATA_DIR / "airbnb/listings.csv.gz")
# # this is just subjective. One can choose some other columns
# keep_cols = [
#     "id",
#     "host_id",
#     "host_since",
#     "description",
#     "host_name",
#     "host_neighbourhood",
#     "host_listings_count",
#     "host_verifications",
#     "host_has_profile_pic",
#     "host_identity_verified",
#     "neighbourhood_cleansed",
#     "latitude",
#     "longitude",
#     "property_type",
#     "room_type",
#     "accommodates",
#     "bathrooms_text",
#     "bedrooms",
#     "beds",
#     "amenities",
#     "price",
#     "minimum_nights",
#     "instant_bookable",
#     "reviews_per_month",
# ]
# airbnb = airbnb_raw[keep_cols]
# airbnb = airbnb[~airbnb.reviews_per_month.isna()]
# airbnb = airbnb[~airbnb.description.isna()]
# airbnb = airbnb[~airbnb.host_listings_count.isna()]
# airbnb = airbnb[airbnb.host_has_profile_pic == "t"].reset_index(drop=True)
# airbnb.drop("host_has_profile_pic", axis=1, inplace=True)
# airbnb_rows, airbnb_cols = airbnb.shape

nyc_taxi = pd.read_csv(RAW_DATA_DIR / "nyc_taxi/train_extended.csv")
nyc_taxi_rows, nyc_taxi_cols = nyc_taxi.shape

fb_comments = pd.read_csv(RAW_DATA_DIR / "fb_comments/Features_Variant_5.csv")
fb_comments_rows, fb_comments_cols = fb_comments.shape

datasets = ["adult", "bank_marketing", "nyc_taxi", "facebook_comments_vol"]
n_rows = [adult_rows, bank_marketing_rows, nyc_taxi_rows, fb_comments_rows]
n_cols = [adult_cols, bank_marketing_cols, nyc_taxi_cols, fb_comments_cols]

basic_stats_df = pd.DataFrame({"Dataset": datasets, "n_rows": n_rows, "n_cols": n_cols})
basic_stats_df["objective"] = ["binary_classification"] * 2 + ["regression"] * 2
basic_stats_df["neg_pos_ratio"] = [
    adult_neg_pos_ratio,
    bank_marketing_rows_neg_pos_ratio,
    "NA",
    "NA",
]

basic_stats_df.to_csv(RAW_DATA_DIR / "basic_stats_df.csv", index=False)
