# %% IMPORTS
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from scikeras.wrappers import KerasRegressor
import matplotlib.pyplot as plt

# %% GLOBAL VARIABLES
CWD = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT_DIR = os.path.join(CWD, "../")
DATA_DIR = os.path.join(ROOT_DIR, "data")

# %% MODEL DEFINITION
def baseline_model(input_dim, hidden_dim1, hidden_dim2):
    # create model
    model = Sequential()
    model.add(
        Dense(
            hidden_dim1,
            input_dim=input_dim,
            kernel_initializer="normal",
            activation="relu",
        )
    )
    # model.add(
    #     Dense(
    #         hidden_dim2,
    #         kernel_initializer="normal",
    #         activation="relu",
    #     )
    # )
    model.add(Dense(1, kernel_initializer="normal"))
    # Compile model
    model.compile(loss="mean_squared_error", optimizer="adam")
    return model


#  %% fix random seed for reproducibility
seed = 7
np.random.seed(seed)
# %% LOADS - Preprocessed
preprocessed_outfile = os.path.join(DATA_DIR, "tcga_preprocessed.csv")
df = pd.read_csv(os.path.join(DATA_DIR, preprocessed_outfile))
feature_columns = [c for c in df.columns if "gene_" in c or c == "treatment"]
feature_df = df[feature_columns]
outcome = df["outcome"]
# %% CREATE MODEL
input_dim = 4001
hidden_dim1 = input_dim
hidden_dim2 = hidden_dim1 // 2

model = KerasRegressor(model=baseline_model(input_dim, hidden_dim1, hidden_dim2))
# %% GRID SEARCH
# Use scikit-learn to grid search the batch size and epochs
# define the grid search parameters

params = {
    "batch_size": [60],
    "epochs": [
        60,
        70,
        80,
        90,
        100,
    ],
}
CV_regressor = GridSearchCV(
    estimator=model,
    param_grid=params,
    # n_jobs=-1,
    cv=3,
    verbose=2,
)

# %% TEST/TRAIN SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    feature_df, outcome, test_size=0.25, random_state=5
)
# %% FIT
CV_regressor.fit(X_train, y_train)
# %% PREDICT
y_pred = CV_regressor.predict(X_test)
y_pred_train = CV_regressor.predict(X_train)
# %%
cv_results_file = "output/neural_net_gridsearch_results5.csv"
cv_results = pd.DataFrame(data=CV_regressor.cv_results_)
cv_outfile = os.path.join(ROOT_DIR, cv_results_file)
cv_results.to_csv(cv_outfile, index=False)
display(pd.DataFrame(data=CV_regressor.cv_results_))
# %% Analyse epochs
cv_results_file = os.path.join(ROOT_DIR, cv_results_file)
cv_results = pd.read_csv(cv_results_file)
for idx, group in cv_results.groupby(by=["param_batch_size"]):
    fig, ax = plt.subplots()
    epochs = ["10", "50", "100"]
    x_pos = range(len(epochs))
    mean_10 = group.loc[group["param_epochs"] == 10]["mean_test_score"].values[0]
    mean_50 = group.loc[group["param_epochs"] == 50]["mean_test_score"].values[0]
    mean_100 = group.loc[group["param_epochs"] == 100]["mean_test_score"].values[0]
    std_10 = group.loc[group["param_epochs"] == 10]["std_test_score"].values[0]
    std_50 = group.loc[group["param_epochs"] == 50]["std_test_score"].values[0]
    std_100 = group.loc[group["param_epochs"] == 100]["std_test_score"].values[0]

    test_scores = [mean_10, mean_50, mean_100]
    error = [std_10, std_50, std_100]
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
    ax.set_xlabel("epochs")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(epochs)
    ax.set_title(f"epochs comparison - param_batch_size={idx}")
    ax.yaxis.grid(True)

    # Save the figure and show
    plt.tight_layout()
    epochs_comparison_plot_file = os.path.join(
        ROOT_DIR,
        f"output/machine_learning_plots/epochs_comparison_param_batch_size={idx}.png",
    )
    plt.savefig(epochs_comparison_plot_file, bbox_inches="tight")
    plt.show()

# %% Analyse batch_size
cv_results_file = "output/neural_net_gridsearch_results2.csv"
cv_results_file = os.path.join(ROOT_DIR, cv_results_file)
cv_results = pd.read_csv(cv_results_file)
for idx, group in cv_results.groupby(by=["param_epochs"]):
    fig, ax = plt.subplots()
    batch_sizes = ["10", "20", "40", "60", "80", "100"]
    x_pos = range(len(batch_sizes))
    mean_10 = group.loc[group["param_batch_size"] == 10]["mean_test_score"].values[0]
    mean_20 = group.loc[group["param_batch_size"] == 20]["mean_test_score"].values[0]
    mean_40 = group.loc[group["param_batch_size"] == 40]["mean_test_score"].values[0]
    mean_60 = group.loc[group["param_batch_size"] == 60]["mean_test_score"].values[0]
    mean_80 = group.loc[group["param_batch_size"] == 80]["mean_test_score"].values[0]
    mean_100 = group.loc[group["param_batch_size"] == 100]["mean_test_score"].values[0]
    std_10 = group.loc[group["param_batch_size"] == 10]["std_test_score"].values[0]
    std_20 = group.loc[group["param_batch_size"] == 20]["std_test_score"].values[0]
    std_40 = group.loc[group["param_batch_size"] == 40]["std_test_score"].values[0]
    std_60 = group.loc[group["param_batch_size"] == 60]["std_test_score"].values[0]
    std_80 = group.loc[group["param_batch_size"] == 80]["std_test_score"].values[0]
    std_100 = group.loc[group["param_batch_size"] == 100]["std_test_score"].values[0]

    test_scores = [mean_10, mean_20, mean_40, mean_60, mean_80, mean_100]
    error = [std_10, std_20, std_40, std_60, std_80, std_100]
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
    ax.set_xlabel("batch_size")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(batch_sizes)
    ax.set_title(f"epochs comparison - param_epoch={idx}")
    ax.yaxis.grid(True)

    # Save the figure and show
    plt.tight_layout()
    batch_size_comparison_plot_file = os.path.join(
        ROOT_DIR,
        f"output/machine_learning_plots/batch_size_comparison_param_epoch={idx}.png",
    )
    plt.savefig(batch_size_comparison_plot_file, bbox_inches="tight")
    plt.show()
