from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
)


def evaluate_model(model, x_test, y_test):
    """Оценка модели и вывод метрик."""
    y_pred = model.predict(x_test)

    # Balanced Accuracy
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    print("Balanced Accuracy:", balanced_acc)

    # Выводим отчёт классификации
    report = classification_report(y_test, y_pred)
    print("Classification Report:\n", report)

    # Матрица ошибок
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)

    return report, cm
