import streamlit as st
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
import plotly.express as px
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(layout="wide")

df_merged = pd.read_csv("df_merged_2.csv")
df_merged["ID"] = df_merged["ID"].astype(str)
scaler = StandardScaler()

st.title("PTSD Analysis ")
st.write(
    "This app performs a PCA analysis and identifies important features in the BIOPACK data."
)
with st.sidebar:
    Normalize = st.checkbox("Normalize", value=True)
    if Normalize:
        df_numeric = df_merged.select_dtypes(include=[np.number])
        df_numeric_without_target = df_numeric.drop(columns=["CAPS", "PTSD"])
        np_without_target = scaler.fit_transform(df_numeric_without_target)
        df_numeric_without_target = pd.DataFrame(
            np_without_target, columns=df_numeric_without_target.columns
        )
        df = pd.merge(
            df_merged[["ID", "CAPS", "PTSD"]],
            df_numeric_without_target,
            left_index=True,
            right_index=True,
        )
    else:
        df = df_merged
st.header("Dataset Description")
st.write("The columns are in the following format:")
st.code(
    """
        TOS session: TOS_<Task>_<Event>_<Var>
        FIX session: FIX_<Task>_<Var>
        """
)
st.write("For both sessions the task number defined the same way:")
st.markdown(
    """
    * Negative task number means that this interval is between tasks (resting phase). -1 is the first resting phase (before any task) and -4 is the last resting phase after all tasks
    * Positive task number means that this interval is during a task. 1 is the first task and 3 is the last task
    """
)
st.write("For FIX sessions the task numbers are defined the following way: ")
event_dict_fix = {  # 1 = resting state, 2 = nback, 3 = faces
    1: "Rest",
    2: "Nback",
    3: "Faces",
}
st.dataframe(event_dict_fix)
st.write(
    "For TOS sessions, the task numbers define the number of the TOS sessions (TOS_1,TOS_2,TOS_3) and the event numbers are defined within each TOS session the following way:"
)
event_dict = {5: "No threat", 6: "Unpredictable threat", 7: "Predictable threat"}
st.dataframe(event_dict)

st.info(
    "Example 1: FIX_-1_EDA_Autocorrelation is the autocorrelation of EDA signal during the first resting phase"
)
st.info(
    "Example 2: TOS_2_7_SCR_Peaks_N is the number of peaks in SCR signal during the 2nd TOS session with predictable threat"
)
st.write("The first few rows of the dataset are:")
st.write(df.head())

# st.dataframe(event_dict)
st.header("1. PCA Analysis")
st.write("PCA is used to plot the data in 2D and 3D.")
link = "https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA"
st.info(
    "[PCA](%s) is a dimensionality reduction technique that projects the data into a lower-dimensional space."
    % link
)
# # Assuming df_pca and df_merged are already defined
pca = PCA(n_components=3)

df_numeric = df.select_dtypes(include=[np.number])
df_numeric = df_numeric.drop(columns=["CAPS", "PTSD"])

pca_result = pca.fit_transform(df_numeric)

df_pca = pd.DataFrame(pca_result, columns=["pca1", "pca2", "pca3"])
df_pca_with_labels = pd.concat([df_pca, df_merged[["CAPS", "PTSD", "ID"]]], axis=1)

# df_pca['CAPS'] = df_merged['CAPS']
# df_pca['PTSD'] = df_merged['PTSD']
# df_pca['ID'] = df_merged['ID']
title = f"Variance Explained: PC1={pca.explained_variance_ratio_[0]:.2f}, PC2={pca.explained_variance_ratio_[1]:.2f}, PC3={pca.explained_variance_ratio_[2]:.2f}"
# # Create the scatter plot
fig_2d = px.scatter(
    df_pca_with_labels,
    x="pca1",
    y="pca2",
    color="CAPS",
    opacity=0.7,
    color_continuous_scale="Bluered",
    hover_data=["ID"],
    symbol="PTSD",
    title=title,
)

# # Update the layout to adjust the legend position
fig_2d.update_layout(
    coloraxis_colorbar=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.5,
        xanchor="left",
        x=-0.05,
        title="CAPS",
        len=1,
    ),
)
fig_2d.update_traces(marker=dict(size=12))

# df_pca_with_labels = pd.concat([df_pca, df_merged[['CAPS', 'PTSD']]], axis=1)

# # Train SVM classifier
svm = SVC(kernel="linear")
svm.fit(df_pca, df_merged["PTSD"])

# # Get the coefficients of the decision plane
coef = svm.coef_[0]
intercept = svm.intercept_[0]

# # Create a mesh grid
x_min, x_max = df_pca["pca1"].min() - 1, df_pca["pca1"].max() + 1
y_min, y_max = df_pca["pca2"].min() - 1, df_pca["pca2"].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50), np.linspace(y_min, y_max, 50))

# Calculate the corresponding z values for the decision plane
zz = (-coef[0] * xx - coef[1] * yy - intercept) / coef[2]

color1 = "grey"
fig_3d = px.scatter_3d(
    df_pca_with_labels,
    x="pca1",
    y="pca2",
    z="pca3",
    color="CAPS",
    symbol="PTSD",
    color_continuous_scale="Bluered",
    opacity=0.7,
)
fig_3d.update_traces(marker=dict(size=6))

fig_3d.add_trace(
    go.Surface(
        x=xx,
        y=yy,
        z=zz,
        colorscale=[[0, color1], [1, color1]],
        opacity=0.5,
        showscale=False,
    )
)


fig_3d.update_layout(
    title=title,
    scene=dict(xaxis_title="PCA1", yaxis_title="PCA2", zaxis_title="PCA3"),
    coloraxis_colorbar=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.2,
        xanchor="left",
        x=0.1,
        title="CAPS",
    ),
)
#


tab_2d, tab_3d = st.tabs(("2D", "3D"))


with tab_2d:
    st.plotly_chart(fig_2d)
with tab_3d:
    st.plotly_chart(fig_3d)


pca = PCA(n_components=10)
pca_result = pca.fit_transform(df_numeric)
# plot explained variance ratio
fig = go.Figure()
fig.add_trace(go.Scatter(x=np.arange(1, 11), y=pca.explained_variance_ratio_))
fig.update_layout(
    title="Explained Variance Ratio",
    xaxis_title="Principal Component",
    yaxis_title="Explained Variance Ratio",
)
st.plotly_chart(fig)

# with st.tabs:]
# st.plotly_chart(fig)
st.header("Feature Importance")
st.markdown(
    """
    Feature importance is a measure of the contribution of each feature to the model's prediction.
    In this section, we will use:
    * Lasso regression
    * SelectKBest with f_classif
    * Random Forest
        """)
X = df.drop(columns=["CAPS", "PTSD"])
y = df["CAPS"]

# Normalize the data
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# Train Lasso regression
link = "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html#sklearn.linear_model.Lasso"
st.subheader("Lasso Regression")
st.info(
    "[Lasso](%s) is a linear model that estimates sparse coefficients.\
        alpha is a constant that multiplies the L1 term, controlling regularization strength "
    % link
)
alpha = st.slider("Alpha", min_value=0.1, max_value=1.0, value=0.2, step=0.1)
lasso = Lasso(alpha=alpha)  # You can adjust the alpha parameter
# lasso.fit(X_scaled, y)
lasso.fit(X, y)

# Get the coefficients
coefficients = lasso.coef_

# Create a DataFrame to display the feature importance
feature_importance = pd.DataFrame({"Feature": X.columns, "Coefficient": coefficients})

# Sort by absolute value of the coefficient
feature_importance["Abs_Coefficient"] = feature_importance["Coefficient"].abs()
feature_importance = feature_importance.sort_values(
    by="Abs_Coefficient", ascending=False
)

# Display the important features

st.write("Important features identified by Lasso regression:")
st.write(feature_importance)

from sklearn.feature_selection import SelectKBest, f_classif

# Assuming df_merged_2 is already defined and contains the features
# X = df.drop(columns=['CAPS', 'PTSD', 'ID'])
# y = df['CAPS']

# Initialize SelectKBest with f_classif and the number of features to select

# Initialize SelectKBest with f_classif and the number of features to select
st.subheader("SelectKBest with f_classif")

link = "https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html"
st.info(
    "[SelectKBest](%s) is a univariate feature selection method that selects the best features based on univariate statistical tests.(ANOVA F-value between label/feature for classification tasks)"
    % link
)
# url = "https://www.streamlit.io"
# st.write("check out this [link](%s)" % url)
# st.markdown("check out this [link](%s)" % url)

k = 10
selector = SelectKBest(score_func=f_classif, k=k)

# Fit the selector to the data
X_new = selector.fit_transform(X, y)

# Get the selected feature names
selected_features = X.columns[selector.get_support()]
selected_scores = selector.scores_[selector.get_support()]
selcted_pvalue = selector.pvalues_[selector.get_support()]
feature_scores = pd.DataFrame(
    {"Feature": selected_features, "Score": selected_scores, "PValue": selcted_pvalue}
)
st.write("Important features identified by SelectKBest with f_classif:")
st.dataframe(
    feature_scores.sort_values(by="Score", ascending=False).reset_index(drop=True),
    column_config={"PValue": st.column_config.NumberColumn(format="%.2g")},
)
# st.write("Using random forest to identify important features")

# Initialize the random forest classifier
rf = RandomForestRegressor()

# Fit the classifier to the data
rf.fit(X, y)

# Get the feature importances
importances = rf.feature_importances_

# Create a DataFrame to display the feature importance
feature_importance_rf = pd.DataFrame({"Feature": X.columns, "Importance": importances})

# Sort by importance
feature_importance_rf = feature_importance_rf.sort_values(
    by="Importance", ascending=False
)

# Display the important features
st.subheader("Random Forest")
link = "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html"
st.info(
    "[Random Forest](%s) is an ensemble learning method that fits a number of decision tree regressors on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting."
    % link
)
st.write("Important features identified by Random Forest:")
st.write(feature_importance_rf)
# st.subheader("Gradient Boosting Machines")
# link = "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html"
# st.info(
#     "[Gradient Boosting Machines](%s) is a machine learning technique for regression and classification problems, which produces a prediction model in the form of an ensemble of weak prediction models, typically decision trees."
#     % link
# )
# # st.write("Using Gradient Boosting Machines (GBMs) to identify important features")
# from sklearn.ensemble import GradientBoostingRegressor

# # Initialize the gradient boosting classifier
# gb = GradientBoostingRegressor()

# # Fit the classifier to the data
# gb.fit(X, y)

# # Get the feature importances
# importances = gb.feature_importances_

# # Create a DataFrame to display the feature importance
# feature_importance_gb = pd.DataFrame({"Feature": X.columns, "Importance": importances})

# # Sort by importance
# feature_importance_gb = feature_importance_gb.sort_values(
#     by="Importance", ascending=False
# )

# # Display the important features
# # st.write("Important features identified by Gradient Boosting Machines:")
# st.write(feature_importance_gb)


# st.table(feature_scores,)
# st.write(selected_features.tolist())
# print(selected_features.tolist())

# feature_importance[['Feature', 'Coefficient']]
