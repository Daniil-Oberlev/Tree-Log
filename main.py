from pathlib import Path
from src.data_preprocessing import load_data, preprocess_data
from src.model_training import train_decision_tree, tune_decision_tree
from src.evaluation import evaluate_model
from src.visualization import plot_decision_tree
from config.config import Config


def main():
    config_path = (
        Path(__file__).resolve().parent / "config.yaml"
    )

    # Загрузка конфигурации
    config = Config(config_path=config_path)
    data_filepath = config.get("data.filepath", "data/data.csv")
    output_dir = Path(config.get("outputs.graphs_dir", "outputs/graphs")).resolve()

    print("Запуск программы...")

    try:
        # Шаг 1: Загрузка данных
        print("Загрузка данных из:", data_filepath)
        df = load_data(data_filepath)
    except FileNotFoundError as e:
        print(f"Ошибка при загрузке данных: {e}")
        return

    try:
        # Шаг 2: Предобработка данных
        print("Предобработка данных...")
        x_train, x_test, y_train, y_test = preprocess_data(df)
    except Exception as e:
        print(f"Ошибка при предобработке данных: {e}")
        return

    try:
        # Шаг 3: Обучение базового дерева решений
        print("Обучение базового дерева решений...")
        dt_model = train_decision_tree(x_train, y_train)
    except Exception as e:
        print(f"Ошибка при обучении дерева решений: {e}")
        return

    try:
        # Шаг 4: Оценка базового дерева решений
        print("Оценка базового дерева решений...")
        evaluate_model(dt_model, x_test, y_test)
    except Exception as e:
        print(f"Ошибка при оценке модели: {e}")
        return

    try:
        # Визуализация базового дерева решений
        print("Визуализация базового дерева решений...")
        feature_names = df.drop(
            columns=["loan_status"]
        ).columns.tolist()
        plot_decision_tree(
            dt_model, feature_names, ["0", "1"], output_dir / "decision_tree_base.png"
        )
    except Exception as e:
        print(f"Ошибка при визуализации дерева решений: {e}")
        return

    try:
        # Шаг 5: Настройка гиперпараметров
        print("Настройка гиперпараметров дерева решений...")
        tuned_dt_model = tune_decision_tree(x_train, y_train)
    except Exception as e:
        print(f"Ошибка при настройке гиперпараметров: {e}")
        return

    try:
        # Шаг 6: Оценка настроенного дерева решений
        print("Оценка настроенного дерева решений...")
        evaluate_model(tuned_dt_model, x_test, y_test)
    except Exception as e:
        print(f"Ошибка при оценке настроенной модели: {e}")
        return

    try:
        # Визуализация настроенного дерева решений
        print("Визуализация настроенного дерева решений...")
        plot_decision_tree(
            tuned_dt_model,
            feature_names,
            ["0", "1"],
            output_dir / "decision_tree_tuned.png",
        )
    except Exception as e:
        print(f"Ошибка при визуализации настроенной модели: {e}")
        return

    print("Программа завершена.")


if __name__ == "__main__":
    main()
