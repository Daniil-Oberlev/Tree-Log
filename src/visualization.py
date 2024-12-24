import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from pathlib import Path
from config.config import Config

# Загружаем конфигурацию
config = Config(config_path="../config.yaml")

# Путь к папке для сохранения графиков
graphs_dir = Path(config.get("outputs.graphs_dir")).resolve()


# Функция для визуализации матрицы ошибок
def plot_confusion_matrix(cm, labels):
    """Визуализация матрицы ошибок."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels
    )
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.show()


# Функция для визуализации дерева решений
def plot_decision_tree(model, feature_names, class_names, filepath):
    """Визуализация дерева решений."""
    fig = plt.figure(figsize=(25, 20))
    _ = tree.plot_tree(
        model, feature_names=feature_names, class_names=class_names, filled=True
    )

    # Проверка, существует ли директория для графиков
    os.makedirs(graphs_dir, exist_ok=True)  # Создание папки, если её нет

    # Сохранение графика
    save_path = Path(filepath).resolve()
    print(f"График сохраняется по пути: {save_path}")
    fig.savefig(save_path, format="png")

    plt.show()
