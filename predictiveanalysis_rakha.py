# -*- coding: utf-8 -*-
"""predictiveAnalysis_Rakha.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1NFM6NgtmJ23SATlL9HvmxhhFhrely35f

## Importing library
"""

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

"""# Data loading

Load the diabetes.csv dataset that i've uploaded to github. transform it into a pandas DataFrame and display the first 5 rows to make sure.
"""

github_url = "https://raw.githubusercontent.com/rakhapta/diabetes-prediction/refs/heads/main/diabetes.csv"
df = pd.read_csv(github_url)
display(df.head())

"""# Exploratory Data Analysis

Explore the dataset to understand its structure, including the shape of the data, the distribution of the data, and the correlation between variables. This includes displaying basic information and descriptive statistics, visualizing distributions using histograms, creating a heatmap to visualize feature correlations, and visualizing the target variable distribution.

### Display the shape of the DataFrame
"""

print("Shape of the DataFrame:", df.shape)

"""### Display basic information about the DataFrame"""

print("\nInfo:")
display(df.info())

"""### Generate descriptive statistics"""

print("\nDescriptive Statistics:")
display(df.describe())

"""### Create histograms for each numerical feature"""

print("\nHistograms:")
plt.figure(figsize=(15, 10))
for i, col in enumerate(df.columns[:-1]):
    plt.subplot(3, 3, i + 1)
    plt.hist(df[col], bins=20, color='skyblue', edgecolor='black')
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

"""We can see that most of the features (such as Pregnancies, SkinThickness, Insulin, DiabetesPedigreeFunction, and Age) have skewed distribution

### Create a heatmap to visualize the correlation matrix
"""

print("\nCorrelation Heatmap:")
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix of Features")
plt.show()

"""We can see that 'Glucose', 'BMI', dan 'Age' having a strong positive correlation with variabel target 'Outcome'.

### Visualize the distribution of the target variable
"""

print("\nTarget Variable Distribution:")
plt.figure(figsize=(6, 4))
sns.countplot(x='Outcome', data=df, palette=['lightcoral', 'lightgreen'])
plt.title("Distribution of Outcome Variable")
plt.xlabel("Outcome (0: Non-Diabetic, 1: Diabetic)")
plt.ylabel("Count")
plt.show()

"""# Data Preparation

## Data cleaning


Handle illogical zero values in 'Glucose', 'BloodPressure', 'BMI',  'SkinThickness',	and 'Insulin' columns. Replacing it with their respective medians and verify the changes
"""

for col in ['Glucose', 'BloodPressure', 'BMI', 'SkinThickness',	'Insulin']:
    median_val = df[col].median()
    df[col] = df[col].replace(0, median_val)

"""### Display descriptive statistics to verify changes"""

print("\nDescriptive Statistics after handling zero values:")
display(df.describe())

"""## Data splitting

Split the data into training and testing sets with an 80:20 ratio, stratifying by the 'Outcome' column, and using a random state for reproducibility.

Define features (x) and target variable (y)
"""

X = df.drop('Outcome', axis=1)
y = df['Outcome']

"""Split data into training and testing sets"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

"""Print the shapes of the resulting sets"""

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

"""## Feature scaling/standarization

Standardize the features using StandardScaler. Then, fit on training data and transform both training and test data

"""

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("X_train_scaled shape:", X_train_scaled.shape)
print("X_test_scaled shape:", X_test_scaled.shape)

"""### Feature Scaling using StandardScaler

Feature scaling is crucial for algorithms like K-Nearest Neighbors (KNN) that rely on distance calculations between data points.  KNN determines the class of a new data point by considering its proximity to its *k* nearest neighbors in the feature space.  Features with larger scales can disproportionately influence these distance calculations. For example, if one feature has a range of 0-100 and another has a range of 0-1, the feature with the larger scale will dominate the distance metric, potentially overshadowing the impact of the other feature.

Standardization, performed by `StandardScaler`, transforms features to have zero mean and unit variance. This ensures that all features contribute equally to the distance computations.  By removing the effect of differing scales, the KNN algorithm can make more accurate predictions based on the true relationships between features rather than being biased by the magnitude of individual features.  The result is a more robust and effective KNN model.

# Model training

Train KNN and Random Forest models using the scaled training data and explain the advantages and disadvantages of each algorithm.

### Initialize and train the KNN model
"""

knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train_scaled, y_train)

print("KNN model trained successfully.")

"""### Initialize and train the Random Forest model"""

rf_model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
rf_model.fit(X_train_scaled, y_train)

print("Random Forest model trained successfully.")

"""# Model evaluation

Evaluate the performance of the trained KNN and Random Forest models using confusion matrices and classification reports. Focus on the recall score for the positive class (diabetes) to determine the best model.

### Predict on the test set for both models, then Generate and plot confusion matrices
"""

knn_predictions = knn_model.predict(X_test_scaled)
rf_predictions = rf_model.predict(X_test_scaled)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
knn_cm = confusion_matrix(y_test, knn_predictions)
sns.heatmap(knn_cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.title("Confusion Matrix - KNN")

plt.subplot(1, 2, 2)
rf_cm = confusion_matrix(y_test, rf_predictions)
sns.heatmap(rf_cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.title("Confusion Matrix - Random Forest")

plt.tight_layout()
plt.show()

"""### Generate classification reports"""

print("Classification Report - KNN:\n", classification_report(y_test, knn_predictions))
print("Classification Report - Random Forest:\n", classification_report(y_test, rf_predictions))

"""Analysis:  
Compare the recall scores for the positive class (diabetes) in both models. A higher recall indicates better ability of the model to correctly identify diabetic cases.  In a medical context, misclassifying a diabetic patient as non-diabetic (false negative) can have severe consequences; thus, a higher recall is often prioritized. The model with the higher recall is preferred.

## Summary:

### Q&A
* **Which model performed better in predicting diabetes risk, and why?**  The Random Forest model performed slightly better than the K-Nearest Neighbors (KNN) model, primarily due to its higher recall score for the positive class (diabetes).  A higher recall indicates a better ability to correctly identify all actual diabetic cases.  In a medical context, misclassifying a diabetic patient as non-diabetic (false negative) can have serious health consequences, thus prioritizing recall.

### Data Analysis Key Findings
* **Data Cleaning:** Illogical zero values in 'Glucose', 'BloodPressure', 'BMI',  'SkinThickness',	and 'Insulin' columns were replaced with their respective medians.
* **Data Splitting:** The dataset was split into 80% training and 20% testing data, using stratification to maintain class distribution.
* **Feature Scaling:**  Features were standardized using `StandardScaler` to ensure equal contribution to distance calculations, especially crucial for the KNN model.
* **Model Comparison (Recall):** The Random Forest model achieved a recall of 0.56 for the positive class (diabetes) compared to KNN's recall of 0.50.  Given the importance of minimizing false negatives in a medical context, the Random Forest model's higher recall makes it the preferred model.

### Insights or Next Steps
* **Hyperparameter Tuning:** Explore hyperparameter tuning for both models to potentially improve their performance, especially focusing on parameters that could increase recall for the positive class.
* **Feature Engineering:** Consider creating new features or transforming existing ones to potentially improve model accuracy.  Investigate the interaction between the features.

# Tuning
## Tuning the parameter to increase the recall score
Tuning both model using manual hyperparameter tuning, starting from random forest first
"""

rf_model_tuned = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, class_weight='balanced')
rf_model_tuned.fit(X_train_scaled, y_train)

knn_model_tuned = KNeighborsClassifier(n_neighbors=5, weights='distance')
knn_model_tuned.fit(X_train_scaled, y_train)

print("Tuned KNN and Random Forest models trained successfully.")

"""### Predict on the test set for both tuned models, then Generate and plot confusion matrices

Predict on both tuned models
"""

knn_predictions_tuned = knn_model_tuned.predict(X_test_scaled)
rf_predictions_tuned = rf_model_tuned.predict(X_test_scaled)

"""Plot confusion matrices"""

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
knn_cm_tuned = confusion_matrix(y_test, knn_predictions_tuned)
sns.heatmap(knn_cm_tuned, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.title("Confusion Matrix - Tuned KNN")

plt.subplot(1, 2, 2)
rf_cm_tuned = confusion_matrix(y_test, rf_predictions_tuned)
sns.heatmap(rf_cm_tuned, annot=True, fmt="d", cmap="Blues",
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.title("Confusion Matrix - Tuned Random Forest")

plt.tight_layout()
plt.show()

"""### Generate classification reports for tuned models"""

print("Classification Report - Tuned KNN:\n", classification_report(y_test, knn_predictions_tuned))
print("Classification Report - Tuned Random Forest:\n", classification_report(y_test, rf_predictions_tuned))

"""Analysis:  
Compare the recall scores for the positive class (diabetes) in both models. A higher recall indicates better ability of the model to correctly identify diabetic cases.  In a medical context, misclassifying a diabetic patient as non-diabetic (false negative) can have severe consequences; thus, a higher recall is often prioritized. The model with the higher recall is preferred.

### Overall Findings after tuning:
*   **Impact of Tuning**: Tuning had a positive impact on both KNN and Random Forest models, improving their recall for the positive class.
*   **Model Comparison (Recall)**: The Tuned Random Forest model demonstrates superior performance in identifying the positive class (diabetes) with a recall of 0.65, significantly higher than the Tuned KNN's recall of 0.54.
*   **Recommendation**: Given the importance of minimizing false negatives in a medical context (misclassifying a diabetic patient as non-diabetic), the Tuned Random Forest model is the preferred model due to its higher recall for the positive class. It is better at correctly identifying diabetic patients.
"""