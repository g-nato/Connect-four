import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix

#----------------Data visualization---------------------#

def count_pieces(line):
    return line.count('R'), line.count('Y')

# Representation of the distribution of results.
def plot_outcome_distribution(data, title, colors):
    data['outcome'] = data['game_state'].str[-1]
    outcome_distribution = data['outcome'].value_counts()

    plt.figure(figsize=(5, 5))
    outcome_distribution.plot(kind='bar', title=title, color=colors)
    plt.xlabel('Outcome')
    plt.ylabel('Count')
    plt.show()

train_data = pd.read_csv('c4-train.txt', header=None, names=["game_state"])
print("First 5 rows of the training dataset:")
print(train_data.head())


#Calculate the distribution of red and yellow pieces in each game state for each set
red_counts = []
yellow_counts = []
for line in train_data['game_state']:
    red_count, yellow_count = count_pieces(line)
    red_counts.append(red_count)
    yellow_counts.append(yellow_count)

plot_outcome_distribution(train_data, 'Train Data', ['red', 'yellow', 'gray'])


test_data = pd.read_csv('c4-test.txt', header=None, names=["game_state"])
print("\nFirst 5 rows of the test dataset:")
print(test_data.head())


red_counts_test = []
yellow_counts_test = []
for line in test_data['game_state']:
    red_count, yellow_count = count_pieces(line)
    red_counts_test.append(red_count)
    yellow_counts_test.append(yellow_count)

plot_outcome_distribution(test_data, 'Test Data', ['red', 'yellow', 'gray'])

validation_data = pd.read_csv("c4-validation.txt", header=None, names=["game_state"])
print("\nFirst 5 rows of the validation dataset:")
print(validation_data.head())


red_counts_validation = []
yellow_counts_validation = []
for line in validation_data['game_state']:
    red_count, yellow_count = count_pieces(line)
    red_counts_validation.append(red_count)
    yellow_counts_validation.append(yellow_count)

plot_outcome_distribution(validation_data, 'Validation Data', ['red', 'yellow', 'gray'])

file_path = 'c4-train.txt'
data = pd.read_csv(file_path, header=None, names=['Data'])
data['Board'] = data['Data'].apply(lambda x: x[:-2].split(' '))
data['Result'] = data['Data'].apply(lambda x: x[-1])
data.drop('Data', axis=1, inplace=True)


# Graph for counting pieces per column

counts = np.zeros(7)
for index, row in data.iterrows():
    for i, column in enumerate(row['Board']):
        counts[i] += len([cell for cell in column if cell != '.'])

plt.figure(figsize=(10, 6))
plt.bar(range(1, 8), counts, color='grey')
plt.title('Distribution of Moves per Column')
plt.xlabel('Column')
plt.ylabel('Number of Moves')
plt.xticks(range(1, 8))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Count the pieces for each player in each column

red_counts = np.zeros(7)
yellow_counts = np.zeros(7)
for index, row in data.iterrows():
    for i, column in enumerate(row['Board']):
        red_counts[i] += column.count('R')
        yellow_counts[i] += column.count('Y')

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].bar(range(1, 8), red_counts, color='red')
axes[0].set_title('Distribution of Moves by Red Player per Column')
axes[0].set_xlabel('Column')
axes[0].set_ylabel('Number of Moves')
axes[0].set_xticks(range(1, 8))
axes[0].grid(axis='y', linestyle='--', alpha=0.7)

axes[1].bar(range(1, 8), yellow_counts, color='yellow')
axes[1].set_title('Distribution of Moves by Yellow Player per Column')
axes[1].set_xlabel('Column')
axes[1].set_xticks(range(1, 8))
axes[1].grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# Graph representing the calculation of the first player's advantage.

data['FirstMover'] = data['Board'].apply(lambda x: 'R' if x.count('R') > x.count('Y') else 'Y')
first_mover_wins = data[data['FirstMover'] == data['Result']].shape[0]
total_games = data.shape[0]
first_mover_win_rate = (first_mover_wins / total_games) * 100

categories = ['First Player Wins', 'Total Games']
values = [first_mover_wins, total_games]
plt.figure(figsize=(8, 5))
plt.bar(categories, values, color=['gray', 'gray'])
plt.title('Impact of the First Move on Game Outcome')
plt.ylabel('Number of Games')
plt.grid(axis='y', linestyle='--', alpha=0.7)
for i, value in enumerate(values):
    plt.text(i, value + 1000, f'{value}', ha='center', va='bottom')
plt.tight_layout()
plt.show()


print(f'Percentage of First Player Wins: {first_mover_win_rate:.2f}%')


frequency_matrix = np.zeros((6, 7), dtype=int)

# Function to update cell frequency based on game states in the training dataset
def update_frequency(game_state, freq_matrix):
    for index, col in enumerate(game_state.strip().split(' ')[0:7]):
        for row, cell in enumerate(reversed(col)):
            if cell in ['R', 'Y']:
                freq_matrix[row, index] += 1


train_file_path = 'c4-train.txt'

with open(train_file_path, 'r') as file:
    for line in file:
        update_frequency(line, frequency_matrix)

row_labels = [str(i) for i in range(1, 7)][::-1]


plt.figure(figsize=(10, 6))
ax = sns.heatmap(frequency_matrix, annot=True, fmt="d", cmap="YlGnBu", cbar=True)
plt.title("Heatmap of Cell Usage in Training Set")
ax.set_ylim(len(frequency_matrix), 0)
ax.set_yticklabels(row_labels)
ax.set_xticklabels(range(1, 8))
plt.xlabel("Column")
plt.ylabel("Row")
plt.show()


#-------------------Pre-processing data------------------#

# Checks for missing data in the datasets
missing_train = train_data.isnull().sum()
missing_test = test_data.isnull().sum()
missing_validation = validation_data.isnull().sum()

print("Missing data in training set:\n", missing_train)
print("\nMissing data in test set:\n", missing_test)
print("\nMissing data in validation set:\n", missing_validation)

# Heatmap for missing data in each dataset

plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
sns.heatmap(train_data.isnull(), cbar=False, cmap='grey')
plt.title('Missing Data in Training Set')
plt.xlabel('Columns')
plt.ylabel('Data Points')

plt.subplot(1, 3, 2)
sns.heatmap(test_data.isnull(), cbar=False, cmap='grey')
plt.title('Missing Data in Test Set')
plt.xlabel('Columns')
plt.ylabel('Data Points')

plt.subplot(1, 3, 3)
sns.heatmap(validation_data.isnull(), cbar=False, cmap='grey')
plt.title('Missing Data in Validation Set')
plt.xlabel('Columns')
plt.ylabel('Data Points')

plt.tight_layout()
plt.show()

# Encoding data into numerical values
def preprocess_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            grid = [0 if char == '.' else 1 if char == 'Y' else 2 for char in line.strip() if char != ' '][:-1]
            label = 1 if line.strip()[-1] == 'Y' else 2 if line.strip()[-1] == 'R' else 3
            data.append(grid + [label])
    return pd.DataFrame(data)


train_data = preprocess_data("c4-train.txt")
test_data = preprocess_data("c4-test.txt")
validation_data = preprocess_data("c4-validation.txt")

print(train_data.head())
print(test_data.head())
print(validation_data.head())

#---------------Random Forest model-------------#

# Random Forest with default hyperparameters

X_train = train_data.iloc[:, :-1]
y_train = train_data.iloc[:, -1]
X_validation = validation_data.iloc[:, :-1]
y_validation = validation_data.iloc[:, -1]
X_test = test_data.iloc[:, :-1]
y_test = test_data.iloc[:, -1]


rf_model_default = RandomForestClassifier(random_state=42)
rf_model_default.fit(X_train, y_train)


y_validation_pred_default = rf_model_default.predict(X_validation)
validation_accuracy_default = accuracy_score(y_validation, y_validation_pred_default)
validation_report_default = classification_report(y_validation, y_validation_pred_default)


y_test_pred_default = rf_model_default.predict(X_test)
test_accuracy_default = accuracy_score(y_test, y_test_pred_default)
test_report_default = classification_report(y_test, y_test_pred_default)


print(f"Accuracy on the validation set (default): {validation_accuracy_default}")
print("Classification report on the validation set (default):\n", validation_report_default)
print(f"Accuracy on the test set (default): {test_accuracy_default}")
print("Classification report on the test set (default):\n", test_report_default)


#--------Random Forest - optimized------#

param_grid = {
    'n_estimators': [100, 150, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 4, 6],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False],
    'class_weight': [None, 'balanced']
}

best_model = None
best_accuracy = 0
best_params = {}


for n_estimators in param_grid['n_estimators']:
    for max_depth in param_grid['max_depth']:
        for min_samples_split in param_grid['min_samples_split']:
            for min_samples_leaf in param_grid['min_samples_leaf']:
                for bootstrap in param_grid['bootstrap']:
                    for class_weight in param_grid['class_weight']:
                        rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                                    min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                                    bootstrap=bootstrap, class_weight=class_weight, random_state=42,n_jobs=-1)
                        rf.fit(X_train, y_train)
                        y_validation_pred = rf.predict(X_validation)
                        accuracy = accuracy_score(y_validation, y_validation_pred)
                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_model = rf
                            best_params = {'n_estimators': n_estimators, 'max_depth': max_depth,
                                           'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf,
                                           'bootstrap': bootstrap, 'class_weight': class_weight}



print("Best parameters:", best_params)
print(f"Best validation accuracy: {best_accuracy}")
validation_report_optimized = classification_report(y_validation, y_validation_pred)
print("Classification report on the validation set (optimized):\n", validation_report_optimized)


y_test_pred_optimized = best_model.predict(X_test)
test_accuracy_optimized = accuracy_score(y_test, y_test_pred_optimized)
test_report_optimized = classification_report(y_test, y_test_pred_optimized)

print(f"Accuracy on the test set (optimized): {test_accuracy_optimized}")
print("Classification report on the test set (optimized):\n", test_report_optimized)



#--------------Confusion matrix---------------#

cm = confusion_matrix(y_test, y_test_pred_optimized)


plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()



#-----------------AUC-ROC-------------------#

y_test_binarized = label_binarize(y_test, classes=[1, 2, 3])

y_prob = best_model.predict_proba(X_test)


auc_roc_scores = roc_auc_score(y_test_binarized, y_prob, multi_class='ovr', average=None)

for i, score in enumerate(auc_roc_scores, 1):
    print(f"AUC-ROC Class {i}: {score:.2f}")

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(y_test_binarized.shape[1]):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(8, 8))
for i in range(y_test_binarized.shape[1]):
    plt.plot(fpr[i], tpr[i], label=f'Class {i+1} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Each Class')
plt.legend(loc="lower right")
plt.show()


# ----------Important features------------#

feature_importances_optimized = best_model.feature_importances_


features_df_optimized = pd.DataFrame({
    'Feature': range(X_train.shape[1]),
    'Importance': feature_importances_optimized
}).sort_values(by='Importance', ascending=False)


plt.figure(figsize=(15, 5))
plt.bar(features_df_optimized['Feature'].astype(str), features_df_optimized['Importance'])
plt.title('Feature Importances from the Optimized Model')
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.show()


importance_matrix_optimized = np.zeros(6*7)
for feature, importance in zip(features_df_optimized['Feature'], features_df_optimized['Importance']):
    importance_matrix_optimized[feature] = importance
importance_matrix_optimized = importance_matrix_optimized.reshape(6, 7)


plt.figure(figsize=(8, 6))
sns.heatmap(importance_matrix_optimized, annot=True, cmap='YlGnBu', fmt=".4f", cbar=True)
plt.xticks(np.arange(0.5, 7.5, 1), labels=np.arange(1, 8, 1))
plt.yticks(np.arange(0.5, 6.5, 1), labels=np.arange(1, 7, 1))
plt.title('Heatmap of Feature Importances for the Optimized Model')
plt.xlabel('Column')
plt.ylabel('Row')
plt.gca().invert_yaxis()
plt.show()


