from sklearn.datasets import load_breast_cancer


def test_schema():
    data = load_breast_cancer(as_frame=True)
    df = data.frame
    # Breast Cancer มี 30 features + 1 target = 31 คอลัมน์
    assert len(df.columns) == 31


def test_two_classes():
    data = load_breast_cancer(as_frame=True)
    y = data.target
    # ข้อมูลเป็น Binary Classification มี 2 คลาส (0 และ 1)
    assert y.nunique() == 2


def test_feature_range():
    data = load_breast_cancer(as_frame=True)
    X = data.data
    # ค่า mean radius ต้องมากกว่า 0
    assert (X["mean radius"] > 0).all()