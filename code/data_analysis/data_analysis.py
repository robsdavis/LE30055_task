# %% IMPORTS
import os
import math
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from bokeh import models as bm
from bokeh.io import show, save
from bokeh.plotting import figure
from bokeh.resources import CDN
from sklearn.feature_selection import mutual_info_regression
from sklearn.neighbors import LocalOutlierFactor
from sklearn.manifold import TSNE

# %% GLOBAL VARIABLES
CWD = os.path.abspath(os.path.join(os.path.abspath(""), ".."))
ROOT_DIR = os.path.join(CWD, "../")
DATA_DIR = os.path.join(ROOT_DIR, "data")

# %% LOADS
data_file = "tcga.csv"
df = pd.read_csv(os.path.join(DATA_DIR, data_file))
feature_columns = [c for c in df.columns if "gene_" in c]
feature_df = df[feature_columns]
# %% TASK 1.1
print(f"The data sheet contains data for {len(df.index)} patients.")
print(f"The data sheet contains {len(feature_columns)} features for each patient.")

# %% TASK 1.2
def get_max_for_feature(feature, df):
    return df[feature].max()  # The max functions skips NaN values by default


feature = "gene_0"
print(
    f"""
        The maximum value for feature '{feature}' is:
        {get_max_for_feature(feature, feature_df)}
    """
)

# %% TASK 1.3
def get_max_val_and_patient_for_feature(feature, df):
    max_val = df[feature].max()  # The max functions skips NaN values by default
    max_indices = df.index[df[feature] == max_val]
    return {"max_val": max_val, "patient_index": max_indices.astype(str).to_list()}


feature = "gene_0"
max_for_feature = get_max_val_and_patient_for_feature(feature, feature_df)
print(
    f"""
        The maximum value for feature '{feature}' is:
            {max_for_feature['max_val']}
        The patient(s) with this max value have id(s):
            "{'; '.join(max_for_feature['patient_index'])}"
    """
)
# %% TASK 1.4
percentage_null = (100 * feature_df.isnull().sum().sum()) / (
    len(feature_df.index) * len(feature_columns)
)
print(
    f"""
        The percentage of values in the covariates dataset is:
            {percentage_null:.7f}%
    """
)
# %% TASK 1.5
def get_mutual_info(feature, df):
    treatment_to_mutual_info = {}
    groups = df.groupby(by="treatment")
    for t, t_df in groups:
        df_no_nans_for_feature = t_df.dropna(subset=[feature])
        feature_vector = df_no_nans_for_feature[feature].to_numpy().reshape(-1, 1)
        mutual_info = mutual_info_regression(
            feature_vector, df_no_nans_for_feature["outcome"].to_numpy()
        )
        treatment_to_mutual_info[t] = mutual_info[0]
    return treatment_to_mutual_info


feature = "gene_1"
treatment_to_mutual_info = get_mutual_info(feature, df)
print(treatment_to_mutual_info)

# %% TASK 1.6
def make_outlier_plot(inliers, outliers):
    tooltips = [
        (f"{feature} value", "@x"),
        ("Outlier", "@outlier"),
    ]
    hover = bm.HoverTool(tooltips=tooltips)
    wheel_zoom = bm.WheelZoomTool()
    reset = bm.ResetTool()
    plot = bm.Plot(
        title="Outlier Plot",
        width=1200,
        height=200,
        min_border=0,
        tools=[hover, wheel_zoom, reset],
    )
    plot.title.text_font_size = "16pt"
    outlier_source = bm.ColumnDataSource(
        {
            "x": outliers[:, 0],
            "y": [1 for x in outliers[:, 0]],
            "feature": [feature for x in outliers[:, 0]],
            "outlier": [True for x in outliers[:, 0]],
        }
    )
    glyph = bm.Scatter(x="x", y="y", size=6, fill_color="#f54e42")
    plot.add_glyph(outlier_source, glyph)
    inlier_source = bm.ColumnDataSource(
        {
            "x": inliers[:, 0],
            "y": [1 for x in inliers[:, 0]],
            "feature": [feature for x in inliers[:, 0]],
            "outlier": [False for x in inliers[:, 0]],
        }
    )
    glyph = bm.Scatter(x="x", y="y", size=5, fill_color="#74add1")
    plot.add_glyph(inlier_source, glyph)
    xaxis = bm.LinearAxis()
    plot.add_layout(xaxis, "below")
    plot.xaxis.axis_label = f"{feature} value"
    plot_file_html = os.path.join(
        ROOT_DIR, f"output/data_analysis_plots/outliers_plot.html"
    )
    save(plot, title="Outlier Plot", filename=plot_file_html, resources=CDN)
    show(plot)


def detect_outliers(feature, feature_df):
    df_no_nans_for_feature = feature_df.dropna(subset=[feature])
    feature_vector = df_no_nans_for_feature[feature].to_numpy().reshape(-1, 1)
    outlier_clf = LocalOutlierFactor()
    y_pred = outlier_clf.fit_predict(feature_vector)
    inliers = feature_vector[np.where(y_pred != -1)]
    outliers = feature_vector[np.where(y_pred == -1)]
    make_outlier_plot(inliers, outliers)
    return [
        {
            "outlier_patient_index": feature_df.index[
                feature_df[feature] == o
            ].to_list(),
            "outlier_value": o,
        }
        for o in outliers[:, 0]
    ]


feature = "gene_1"
outliers = detect_outliers(feature, feature_df)
print(outliers)

# %% TASK 1.7
def make_TSNE_plot(embedded_feature_vector, treatments):
    tooltips = [
        ("patient index", "@patient_index"),
        ("treatment", "@treatment"),
        ("x", "@x"),
        ("y", "@y"),
    ]
    hover = bm.HoverTool(tooltips=tooltips)
    wheel_zoom = bm.WheelZoomTool()
    reset = bm.ResetTool()
    plot = bm.Plot(
        title="TSNE Plot",
        width=600,
        height=600,
        min_border=0,
        tools=[hover, wheel_zoom, reset],
    )
    plot.title.text_font_size = "16pt"
    source_df = pd.DataFrame(
        data={
            "patient_index": [i for i in range(len(embedded_feature_vector[:, 0]))],
            "treatment": treatments,
            "x": embedded_feature_vector[:, 0],
            "y": embedded_feature_vector[:, 1],
        }
    )
    colours = ["#6cc9d4", "#30d16b", "#d9415f"]
    for idx, df in source_df.groupby(by=["treatment"]):
        source = bm.ColumnDataSource(df.to_dict(orient="list"))
        glyph = bm.Scatter(x="x", y="y", size=6, fill_color=colours[idx])
        plot.add_glyph(source, glyph)
    xaxis = bm.LinearAxis()
    plot.add_layout(xaxis, "below")
    yaxis = bm.LinearAxis()
    plot.add_layout(yaxis, "left")
    plot_file_html = os.path.join(
        ROOT_DIR, f"output/data_analysis_plots/TSNE_plot.html"
    )
    save(plot, title="TSNE Plot", filename=plot_file_html, resources=CDN)
    show(plot)


def dimentionality_reduction(feature_df, treatments):
    # impute missing data by filling with median value
    feature_df = feature_df.replace(np.NaN, feature_df.median())
    feature_vector = feature_df.to_numpy()
    embedded_feature_vector = TSNE(
        n_components=2, learning_rate="auto", init="pca"
    ).fit_transform(feature_vector)
    make_TSNE_plot(embedded_feature_vector, treatments)
    return embedded_feature_vector


feature_columns_with_treatment = [
    c for c in df.columns if "gene_" in c or c == "treatment"
]
feature_df_with_treatments = df[feature_columns_with_treatment]
treatments = df["treatment"]
embedded_feature_vector = dimentionality_reduction(
    feature_df_with_treatments, treatments
)
# print(embedded_feature_vector)
# make_TSNE_plot(embedded_feature_vector, treatments)

# %% TASK 1.8 Data Anonymization
def get_k_anon_and_l_dev(s_cols, df):
    """
    supress performance warning, because although the dataframe
    is highly fragmented, the processing of fragments is minimal and
    values are returned near instantly.
    """
    warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

    df = df.replace(np.NaN, df.median())
    quasi_identifiers = ["treatment"]
    eqiv_classes = df.groupby(by=quasi_identifiers)

    distinct_min_l = 10e1000000000000
    entropy_min_l = 10e1000000000000
    for idx_, eqiv_class in eqiv_classes:
        # distinct
        distinct_l = len(eqiv_class.groupby(by=s_cols).size().reset_index(name="count"))
        distinct_min_l = min(distinct_min_l, distinct_l)
        # entropy
        probabilities = eqiv_class[s_cols].value_counts(normalize=True)
        prob_log_prob = probabilities * np.log(probabilities)
        entropy_l = math.exp(-prob_log_prob.sum())
        entropy_min_l = min(entropy_min_l, entropy_l)

    counts = eqiv_classes.size().reset_index(name="count")["count"]
    counts = counts[counts != 0]
    k = counts.min()

    return {
        "k-anonymity metric, k": k,
        "Distinct l-diversity metric, l": distinct_min_l,
        "Entropy l-diversity metric, l": int(entropy_min_l),
    }


sensitive_columns = [f"gene_{i}" for i in range(100)]
anonymity_df = df.iloc[0:1000].copy()
print(get_k_anon_and_l_dev(sensitive_columns, anonymity_df))
