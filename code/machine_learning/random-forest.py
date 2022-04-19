# %% IMPORTS
import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
import forestci as fci

# %% GLOBAL VARIABLES
CWD = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT_DIR = os.path.join(CWD, "../")
DATA_DIR = os.path.join(ROOT_DIR, "data")

# %% LOADS
data_file = "tcga.csv"
df = pd.read_csv(os.path.join(DATA_DIR, data_file))

# %% PREPROCESS
"""
NB: There are no duplicate rows.
NB: Are columns are float64 except treatment which is int64.
NB: There are no columns that contain a single value
    (no zero-variance features).
NB: There are columns that have near-zero variance. These
    will not be removed in the first instance, but it is
    worth baring them in mind. If we set our threshold for
    "near-zero" variance to containing less than 1% unique values,
    they are: gene_332, gene_573, gene_836, gene_837, gene_838, gene_839,
    gene_841, gene_871, gene_891, gene_1530, gene_1534, gene_1540,
    gene_1976, gene_1977, gene_1978, gene_2106 and gene_3063.
NB: Columns do contain missing values and will be imputed with the median
    value. This does not factor the covariance between features, but there
    are too many missing values to impute with linear regression.

"""
df.replace(np.NaN, df.median(), inplace=True)

preprocessed_outfile = os.path.join(DATA_DIR, "tcga_preprocessed.csv")
df.to_csv(preprocessed_outfile, index=False)

# %% LOADS - Preprocessed
preprocessed_outfile = os.path.join(DATA_DIR, "tcga_preprocessed.csv")
df = pd.read_csv(os.path.join(DATA_DIR, preprocessed_outfile))
feature_columns = [c for c in df.columns if "gene_" in c or c == "treatment"]
feature_df = df[feature_columns]
outcome = df["outcome"]

# %% SET UP VECTORS
near_zero_v_columns = [
    "gene_332",
    "gene_573",
    "gene_836",
    "gene_837",
    "gene_838",
    "gene_839",
    "gene_841",
    "gene_871",
    "gene_891",
    "gene_1530",
    "gene_1534",
    "gene_1540",
    "gene_1976",
    "gene_1977",
    "gene_1978",
    "gene_2106",
    "gene_3063",
]
v_columns = [col for col in feature_columns if col not in near_zero_v_columns]
feature_df_no_near_zero_variance = feature_df[v_columns]

X_train, X_test, y_train, y_test = train_test_split(
    feature_df, outcome, test_size=0.25, random_state=5
)

# %% Create and fit regressor
regressor = RandomForestRegressor(
    n_estimators=200,
    min_samples_leaf=1,
    max_samples=0.7,
    max_features="auto",
    max_depth=4,
    random_state=5,
)
regressor.fit(X_train, y_train)

# %% PREDICT
y_pred = regressor.predict(X_test)
y_pred_train = regressor.predict(X_train)

# %% PRINT PREDICTIONS
pred_df = pd.DataFrame(
    data={
        "outcome": y_test,
        "RF_Prediction": y_pred,
        "error": abs(y_test - y_pred),
        "squared_error": (y_test - y_pred) ** 2,
    }
)

pred_df_train = pd.DataFrame(
    data={
        "outcome": y_train,
        "RF_Prediction": y_pred_train,
        "error": y_train - y_pred_train,
    }
)
display(pred_df)

# %% ACCURACY
print(
    "Training RMSE: ",
    metrics.mean_squared_error(y_train, y_pred_train, squared=False),
)
print(
    "Test RMSE: ",
    metrics.mean_squared_error(y_test, y_pred, squared=False),
)

print("Training R^2: ", metrics.r2_score(y_train, y_pred_train))
print("Test R^2: ", metrics.r2_score(y_test, y_pred))
# %% Uncertainty
# # Calculate the variance
# # Plot predicted outcome without error bars
# plt.scatter(pred_df["outcome"], pred_df["RF_Prediction"])
# plt.xlabel("Reported outcome")
# plt.ylabel("Predicted outcome")
# plt.show()

# # Calculate the variance - More work required to get fci.random_forest_error working
# outcome_forrest_error = fci.random_forest_error(regressor, X_train, X_test, calibrate=True)

# # Plot error bars for predicted outcome using unbiased variance
# plt.errorbar(pred_df["outcome"], pred_df["RF_Prediction"], yerr=np.sqrt(outcome_forrest_error), fmt="o")
# plt.xlabel("Reported outcome")
# plt.ylabel("Predicted outcome")
# plt.show()

# %% Max error
print("Here are the rows in the top 1% for squared error:")
display(pred_df.nlargest(len(pred_df) // 100, columns="error", keep="all"))

# %% FEATURE IMPORTANCES
drop_tratment = False
feature_importance_df = pd.DataFrame(
    data={
        "Feature": [f"gene_{i}" for i in range(4000)] + ["treatment"],
        "importance": regressor.feature_importances_,
    }
)
if drop_tratment:
    feature_importance_df = feature_importance_df.drop(
        feature_importance_df.tail(1).index, inplace=True
    )
    n = 20
else:
    n = 21
feature_importance_df.plot()
display(feature_importance_df.nlargest(n, columns="importance", keep="all"))


# %% Hyperparameter tuning with grid search
regressor = RandomForestRegressor(random_state=5)
params = {
    "n_estimators": [100, 200],
    "max_features": ["auto", "sqrt", "log2"],
    "max_depth": [4, 6, 8],
    "criterion": ["squared_error", "absolute_error"],
}
CV_regressor = GridSearchCV(
    estimator=regressor, param_grid=params, n_jobs=-1, cv=3, verbose=2
)
CV_regressor.fit(X_train, y_train)

# %% Output Gridsearch results
cv_results = pd.DataFrame(data=CV_regressor.cv_results_)
cv_outfile = os.path.join(ROOT_DIR, "output/random-forest_gridsearch_results.csv")
cv_results.to_csv(cv_outfile, index=False)
display(pd.DataFrame(data=CV_regressor.cv_results_))

# %% Analyse max_features
for idx, group in cv_results.groupby(
    by=["param_max_depth", "param_criterion", "param_n_estimators"]
):
    fig, ax = plt.subplots()
    max_features = ["auto", "sqrt", "log2"]
    x_pos = range(len(max_features))
    auto_mean = group.loc[group["param_max_features"] == "auto"][
        "mean_test_score"
    ].values[0]
    sqrt_mean = group.loc[group["param_max_features"] == "sqrt"][
        "mean_test_score"
    ].values[0]
    log2_mean = group.loc[group["param_max_features"] == "log2"][
        "mean_test_score"
    ].values[0]
    auto_std = group.loc[group["param_max_features"] == "auto"][
        "std_test_score"
    ].values[0]
    sqrt_std = group.loc[group["param_max_features"] == "sqrt"][
        "std_test_score"
    ].values[0]
    log2_std = group.loc[group["param_max_features"] == "log2"][
        "std_test_score"
    ].values[0]

    test_scores = [auto_mean, sqrt_mean, log2_mean]
    error = [auto_std, sqrt_std, log2_std]
    ax.bar(
        x_pos,
        test_scores,
        yerr=error,
        align="center",
        alpha=0.5,
        ecolor="black",
        capsize=10,
    )
    ax.set_ylabel("Test score")
    ax.set_xlabel("max_features")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(max_features)
    ax.set_title(
        f"max_features comparison - max_depth={idx[0]}, criterion={idx[1]}, n_estimators={idx[2]}"
    )
    ax.yaxis.grid(True)

    # Save the figure and show
    plt.tight_layout()
    max_features_comparison_plot_file = os.path.join(
        ROOT_DIR,
        f"output/machine_learning_plots/criterion_comparison_max_depth={idx[0]}_criterion={idx[1]}_n_estimators={idx[2]}.png",
    )
    plt.savefig(max_features_comparison_plot_file, bbox_inches="tight")
    plt.show()
# %% Analyse max_depth
for idx, group in cv_results.groupby(
    by=["param_max_features", "param_criterion", "param_n_estimators"]
):
    fig, ax = plt.subplots()
    max_depth = ["4", "6", "8"]
    x_pos = range(len(max_depth))
    mean_4 = group.loc[group["param_max_depth"] == 4]["mean_test_score"].values[0]
    mean_6 = group.loc[group["param_max_depth"] == 6]["mean_test_score"].values[0]
    mean_8 = group.loc[group["param_max_depth"] == 8]["mean_test_score"].values[0]
    std_4 = group.loc[group["param_max_depth"] == 4]["std_test_score"].values[0]
    std_6 = group.loc[group["param_max_depth"] == 6]["std_test_score"].values[0]
    std_8 = group.loc[group["param_max_depth"] == 8]["std_test_score"].values[0]

    test_scores = [mean_4, mean_6, mean_8]
    error = [std_4, std_6, std_8]
    ax.bar(
        x_pos,
        test_scores,
        yerr=error,
        align="center",
        alpha=0.5,
        ecolor="black",
        capsize=10,
    )
    ax.set_ylabel("Test score")
    ax.set_xlabel("max_depth")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(max_depth)
    ax.set_title(
        f"max_depth comparison - param_max_features={idx[0]}, criterion={idx[1]}, n_estimators={idx[2]}"
    )
    ax.yaxis.grid(True)
    if idx[0] == "auto":
        ax.set_ylim(ymin=0.7)

    # Save the figure and show
    plt.tight_layout()
    max_depth_comparison_plot_file = os.path.join(
        ROOT_DIR,
        f"output/machine_learning_plots/criterion_comparison_param_max_features={idx[0]}_criterion={idx[1]}_n_estimators={idx[2]}.png",
    )
    plt.savefig(max_depth_comparison_plot_file, bbox_inches="tight")
    plt.show()
# %% Gridsearch for customizing
regressor2 = RandomForestRegressor(random_state=5)
params = {
    "n_estimators": [100],
    "max_features": ["auto"],
    "max_depth": [4, 6, 8, 10, 12, 14, None],
    "criterion": ["squared_error"],
}
CV_regressor2 = GridSearchCV(estimator=regressor2, param_grid=params, cv=3, verbose=2)
CV_regressor2.fit(X_train, y_train)

# %% Output Gridsearch results
cv_results2 = pd.DataFrame(data=CV_regressor2.cv_results_)
cv_outfile2 = os.path.join(ROOT_DIR, "output/random-forest_search for max depth.csv")
cv_results2.to_csv(cv_outfile2, index=False)
display(pd.DataFrame(data=CV_regressor2.cv_results_))

# %% Analyse max_depth
fig, ax = plt.subplots()
max_depths = [4, 6, 8, 10, 12, 14, "None"]
x_pos = range(len(max_depths))
test_scores = cv_results2["mean_test_score"]
error = cv_results2["std_test_score"]
ax.bar(
    x_pos,
    test_scores,
    yerr=error,
    align="center",
    alpha=0.5,
    ecolor="black",
    capsize=10,
)
ax.set_ylabel("Test score")
ax.set_xlabel("max_depth")
ax.set_xticks(x_pos)
ax.set_xticklabels(max_depths)
ax.set_title(f"max_depth vs test_score")
ax.yaxis.grid(True)
ax.set_ylim(ymin=0.7)

# Save the figure and show
plt.tight_layout()
max_depth_comparison_plot_file = os.path.join(
    ROOT_DIR, f"output/machine_learning_plots/max_depth_line_plot.png"
)
plt.savefig(max_depth_comparison_plot_file, bbox_inches="tight")
plt.show()
