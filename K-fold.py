threshold = np.median(y_train)
y_train_binary = Binarizer(threshold=threshold).fit_transform(y_train.values.reshape(-1, 1)).ravel()

# Number of folds
n_folds = 10


# Perform 10-fold cross-validation
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
rmse_list = []
accuracy_list = []

fold_number = 1
for train_index, test_index in kf.split(X_train):
    X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
    y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]
    y_train_binary_fold, y_test_binary_fold = y_train_binary[train_index], y_train_binary[test_index]

    # Fit the model
    model.fit(X_train_fold, y_train_fold)

    # Predict
    y_pred_fold = model.predict(X_test_fold)

    # Calculate RMSE for the fold
    rmse = np.sqrt(mean_squared_error(y_test_fold, y_pred_fold))
    rmse_list.append(rmse)

    # Calculate Accuracy for the fold (binary classification)
    y_pred_binary_fold = Binarizer(threshold=threshold).fit_transform(y_pred_fold.reshape(-1, 1)).ravel()
    accuracy = accuracy_score(y_test_binary_fold, y_pred_binary_fold)
    accuracy_list.append(accuracy)

    # Calculate fold-wise statistics
    fold_mean_rmse = np.mean(rmse_list)
    fold_std_rmse = np.std(rmse_list)
    fold_mean_accuracy = np.mean(accuracy_list)
    fold_std_accuracy = np.std(accuracy_list)

    # Print results for the current fold
    print(f"Fold {fold_number} - RMSE: {rmse:.4f}, Accuracy: {accuracy:.4f}")
    print(f"Fold {fold_number} - Mean RMSE: {fold_mean_rmse:.4f}, Std RMSE: {fold_std_rmse:.4f}")
    print(f"Fold {fold_number} - Mean Accuracy: {fold_mean_accuracy:.4f}, Std Accuracy: {fold_std_accuracy:.4f}")
    fold_number += 1

# Calculate overall statistics
mean_rmse = np.mean(rmse_list)
std_rmse = np.std(rmse_list)
mean_accuracy = np.mean(accuracy_list)
std_accuracy = np.std(accuracy_list)

print("\nOverall RMSE for each fold:", rmse_list)
print("Overall Mean RMSE:", mean_rmse)
print("Overall Standard Deviation of RMSE:", std_rmse)
print("Overall Accuracy for each fold:", accuracy_list)
print("Overall Mean Accuracy:", mean_accuracy)
print("Overall Standard Deviation of Accuracy:", std_accuracy)
