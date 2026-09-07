import mlflow
from sklearn.datasets import load_breast_cancer


def load_and_predict():
    """Simulates a production scenario by loading a model using an alias

    from the MLflow Model Registry and predicting on two samples (one for each class).
    """
    MODEL_NAME = "cancer-classifier-prod"
    MODEL_ALIAS = "staging"

    print(f"Loading model '{MODEL_NAME}' with alias '@{MODEL_ALIAS}'...")

    # 1. โหลดโมเดลจาก MLflow Model Registry ผ่าน Alias URI
    try:
        model = mlflow.pyfunc.load_model(
            model_uri=f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
        )
    except mlflow.exceptions.MlflowException as e:
        print(f"\nError loading model: {e}")
        print(
            f"Please make sure a model version has the alias '@{MODEL_ALIAS}' in the MLflow UI."
        )
        return

    # 2. โหลดชุดข้อมูล Breast Cancer
    cancer_data = load_breast_cancer(as_frame=True)
    X = cancer_data.data
    y = cancer_data.target
    target_names = cancer_data.target_names  # ['malignant', 'benign']

    # 3. ดึงตัวอย่างรายแรกของแต่ละคลาส (คลาส 0 = malignant, คลาส 1 = benign)
    idx_malignant = y[y == 0].index[0]
    idx_benign = y[y == 1].index[0]

    sample_indices = [idx_malignant, idx_benign]
    sample_X = X.loc[sample_indices]
    actual_y = y.loc[sample_indices]

    # 4. ใช้โมเดลทำนายผล
    predictions = model.predict(sample_X)

    # 5. แสดงผลการทำนายแบบคำ (malignant / benign) พร้อมระบุว่าทำนายถูกหรือไม่
    print("\n" + "=" * 50)
    print("EVALUATION RESULT")
    print("=" * 50)

    for i, idx in enumerate(sample_indices):
        actual_code = actual_y.loc[idx]
        pred_code = int(predictions[i])

        actual_name = target_names[actual_code]
        pred_name = target_names[pred_code]

        is_correct = actual_code == pred_code
        status = (
            "ทำนายถูกต้อง (Correct)"
            if is_correct
            else "ทำนายผิด (Incorrect)"
        )

        print(f"Sample #{i+1} (Row Index {idx}):")
        print(f"  - Actual Class   : {actual_name} ({actual_code})")
        print(f"  - Predicted Class: {pred_name} ({pred_code})")
        print(f"  - Result         : {status}\n")

    print("=" * 50)


if __name__ == "__main__":
    load_and_predict()