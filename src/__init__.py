import mlflow

# Option 1: Enable autologging only for PyTorch
mlflow.pytorch.autolog()

# # Option 2: Disable autologging for scikit-learn, but enable it for other libraries
# mlflow.sklearn.autolog(disable=True)
mlflow.autolog()