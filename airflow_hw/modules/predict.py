import os
import dill
import pandas as pd
import json
import numpy as np
from sklearn.pipeline import Pipeline
from datetime import datetime
# Укажем путь к файлам проекта
path = os.environ.get('PROJECT_PATH', './airflow_hw')


def load_model(model_path: str) -> Pipeline:
    """Загружает обученную модель из файла."""
    with open(model_path, 'rb') as file:
        model = dill.load(file)
    return model


def predict_data(model: Pipeline, data: pd.DataFrame) -> pd.Series:
    """Делает предсказания для переданных данных."""
    return model.predict(data)


def process_test_data(test_folder: str) -> pd.DataFrame:
    """Читает все JSON-файлы в папке с тестовыми данными и объединяет их в один DataFrame."""
    all_data = []
    for file_name in os.listdir(test_folder):
        if file_name.endswith('.json'):
            file_path = os.path.join(test_folder, file_name)

            # Пробуем загрузить файл как JSON
            with open(file_path, 'r') as f:
                try:
                    data = json.load(f)
                    # Если данные в формате словаря, преобразуем в DataFrame
                    if isinstance(data, dict):
                        df = pd.DataFrame([data])
                    elif isinstance(data, list):
                        df = pd.DataFrame(data)
                    else:
                        raise ValueError(f"Unsupported JSON format in {file_name}")
                    all_data.append(df)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON in file {file_name}")
                    continue
                except ValueError as ve:
                    print(f"Error processing file {file_name}: {ve}")
                    continue
    # Объединяем все DataFrame в один
    return pd.concat(all_data, ignore_index=True)


def save_predictions(predictions: pd.Series, test_data: pd.DataFrame, output_folder: str, output_filename: str) -> None:
    """Сохраняет предсказания и исходные данные в CSV файл."""
    output_path = os.path.join(output_folder, output_filename)

    if isinstance(predictions, np.ndarray):
        predictions = pd.Series(predictions)

    test_data = pd.DataFrame({
        'car_id': test_data['id'],
        'pred': predictions
    })

    test_data.to_csv(output_path, index=False)
    print(f'Predictions saved to {output_path}')


def get_latest_model_path(models_folder: str) -> str:
    """Возвращает путь к последнему файлу модели с расширением .pkl."""
    # Список всех файлов с расширением .pkl в папке
    model_files = [f for f in os.listdir(models_folder) if f.endswith('.pkl')]

    # Если файлы найдены, сортируем по имени, предполагая, что имя содержит временную метку
    if model_files:
        latest_model_file = max(model_files, key=lambda f: f.split('_')[-1])  # Сортировка по последнему элементу в имени
        return os.path.join(models_folder, latest_model_file)
    else:
        raise FileNotFoundError("No model files found in the specified folder.")


def predict():
    """Основная функция для загрузки модели, предсказания и сохранения результатов."""
    # Путь к тестовым данным и папке для сохранения предсказаний
    test_folder = f'{path}/data/test'
    output_folder = f'{path}/data/predictions'

    # Получаем путь к последнему файлу модели
    model_path = get_latest_model_path(f'{path}/data/models')

    # Загружаем модель
    model = load_model(model_path)

    # Читаем тестовые данные
    test_data = process_test_data(test_folder)

    # Применяем модель для предсказания
    predictions = predict_data(model, test_data)

    # Сохраняем предсказания в файл
    output_filename = f'preds_{datetime.now().strftime("%Y%m%d%H%M")}.csv'
    save_predictions(predictions, test_data, output_folder, output_filename)


if __name__ == '__main__':
    predict()
