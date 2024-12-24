from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from config.config import Config

# Загружаем конфигурацию
config = Config(config_path="../config.yaml")

# Чтение глобальных параметров
random_state = config.get("model.random_state", 42)
logistic_max_iter = config.get(
    "model.logistic_max_iter", 1000
)  # Лимит итераций для логистической регрессии
grid_params = config.get(
    "model.grid_params",
    {
        "criterion": ["gini", "entropy"],
        "splitter": ["best", "random"],
        "max_depth": [3, 5, 10, None],
        "max_features": [None, 3, 5, 7],
        "min_samples_leaf": [1, 2, 3],
        "min_samples_split": [2, 3, 5],
    },
)
cv_folds = config.get("model.cv_folds", 5)


def train_logistic_model(x_train, y_train):
    """
    Обучение модели логистической регрессии на сбалансированных данных.
    """
    # Балансировка данных
    ros = RandomOverSampler(random_state=random_state)
    x_resampled, y_resampled = ros.fit_resample(x_train, y_train)

    # Создание и обучение модели
    model = LogisticRegression(random_state=random_state, max_iter=logistic_max_iter)
    model.fit(x_resampled, y_resampled)

    return model


def train_decision_tree(x_train, y_train):
    """
    Обучение базовой модели дерева решений.
    """
    model = DecisionTreeClassifier(random_state=random_state)
    model.fit(x_train, y_train)
    return model


def tune_decision_tree(x_train, y_train):
    """
    Настройка гиперпараметров дерева решений с помощью GridSearchCV.
    """
    model = DecisionTreeClassifier(random_state=random_state)

    # Настройка гиперпараметров через GridSearchCV
    grid_search = GridSearchCV(
        estimator=model, param_grid=grid_params, cv=cv_folds, n_jobs=-1
    )
    grid_search.fit(x_train, y_train)

    print("Лучшие параметры:", grid_search.best_params_)
    return grid_search.best_estimator_
