import os
import pytest
import pandas as pd
from src.data_preprocessing import load_data, preprocess_data


@pytest.fixture
def create_sample_csv():
    """Создаёт временный CSV файл для тестов."""
    test_dir = 'tests'
    os.makedirs(test_dir, exist_ok=True)

    test_data = {
        'loan_status': [1, 0, 1],
        'column1': [1, 2, 3],
        'column2': ['A', 'B', 'C']
    }
    df = pd.DataFrame(test_data)
    filepath = os.path.join(test_dir, 'data.csv')

    df.to_csv(filepath, index=False)

    yield filepath

    os.remove(filepath)

    if not os.listdir(test_dir):
        os.rmdir(test_dir)


def test_load_data(create_sample_csv):
    """Тест загрузки данных из CSV файла."""
    filepath = create_sample_csv
    data = load_data(filepath)
    assert not data.empty, "Данные не должны быть пустыми"
    assert set(data.columns) == {'loan_status', 'column1', 'column2'}, "Столбцы данных не соответствуют ожиданиям"


def test_load_data_file_not_found():
    """Тест на несуществующий файл."""
    with pytest.raises(FileNotFoundError):
        load_data('non_existent_file.csv')


def test_preprocess_data(create_sample_csv):
    """Тест предобработки данных."""
    filepath = create_sample_csv
    df = load_data(filepath)
    x_train, x_test, y_train, y_test = preprocess_data(df)

    # Проверяем, что разделение данных прошло корректно
    assert x_train.shape[0] == 2, "Неверное количество строк в обучающей выборке"
    assert x_test.shape[0] == 1, "Неверное количество строк в тестовой выборке"
    assert y_train.shape[0] == 2, "Неверное количество меток в обучающей выборке"
    assert y_test.shape[0] == 1, "Неверное количество меток в тестовой выборке"


def test_preprocess_data_missing_target():
    """Тест для обработки отсутствия колонки 'loan_status'."""
    # Создаем данные без 'loan_status'
    test_data = {
        'column1': [1, 2, 3],
        'column2': ['A', 'B', 'C']
    }
    df = pd.DataFrame(test_data)

    with pytest.raises(ValueError):
        preprocess_data(df)
