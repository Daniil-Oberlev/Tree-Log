import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(filepath):
    """Загрузка данных из CSV-файла."""
    return pd.read_csv(filepath)


def preprocess_data(df):
    """Предобработка данных: масштабирование, кодирование и разбиение."""
    df = df.dropna()

    # Проверяем наличие целевой переменной
    if "loan_status" not in df.columns:
        raise ValueError("В данных отсутствует колонка 'loan_status'.")

    # Разделяем на признаки и целевую переменную
    x = df.drop("loan_status", axis=1)
    y = df["loan_status"]

    # Разделяем на обучающую и тестовую выборки
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    return x_train, x_test, y_train, y_test
