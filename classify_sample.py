# import cupy as cp
# import numpy as np
# from cuml.svm import SVC
# from cuml.metrics import accuracy_score
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import classification_report
# from sklearn.model_selection import train_test_split

# # Prepare features
# X = np.vstack(small_data["diff_vec"].values).astype(np.float32)

# # Encode string labels -> integers
# le = LabelEncoder()
# y = le.fit_transform(small_data["label"].values).astype(np.int32)

# # Split train/test (stratified so class balance is preserved)
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )

# # Move to GPU
# X_train_gpu = cp.asarray(X_train)
# y_train_gpu = cp.asarray(y_train)
# X_test_gpu = cp.asarray(X_test)
# y_test_gpu = cp.asarray(y_test)

# # Train SVM on GPU
# clf = SVC(kernel="rbf", C=1.0, gamma="scale")
# clf.fit(X_train_gpu, y_train_gpu)

# # Predict on test set
# y_pred = clf.predict(X_test_gpu)

# # Accuracy
# acc = accuracy_score(y_test_gpu, y_pred)
# print("Accuracy (test):", acc)

# # Classification report (convert GPU preds back + inverse transform labels)
# y_pred_cpu = cp.asnumpy(y_pred)
# print("\nClassification report (test):\n")
# print(classification_report(le.inverse_transform(y_test), le.inverse_transform(y_pred_cpu)))
