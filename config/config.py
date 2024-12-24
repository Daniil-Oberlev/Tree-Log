import yaml
import os


class Config:
    def __init__(self, config_path="config.yaml"):
        self._config_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), config_path
        )
        self._config = self._load_config()

    def _load_config(self):
        """Загружает конфигурацию из YAML-файла."""
        try:
            with open(self._config_path, "r") as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Конфигурационный файл {self._config_path} не найден."
            )
        except yaml.YAMLError as e:
            raise ValueError(f"Ошибка чтения YAML-файла: {e}")

    def get(self, key, default=None):
        """Получает значение из конфигурации по ключу, поддерживает вложенные ключи."""
        keys = key.split(".")
        value = self._config
        try:
            for k in keys:
                value = value[k]
            return value
        except KeyError:
            if default is not None:
                return default
            raise KeyError(f"Ключ '{key}' не найден в конфигурации.")
