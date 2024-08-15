import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import os
from tqdm import tqdm
from utils import (
    prepare_ml_data_from_features,
    cyclical_feature_encoding,
    calculate_horizons,
    generate_time_series_features,
)
import warnings
import random

# Фиксация random seed для воспроизводимости
random_seed = 42
np.random.seed(random_seed)
tf.random.set_seed(random_seed)
random.seed(random_seed)

# Игнорировать все предупреждения
warnings.filterwarnings("ignore")

# Загрузка данных и подготовка
file_path = "Meteo1-2023-15min(resampled).csv"
df = pd.read_csv(file_path, delimiter=",")
df["time_YYMMDD_HHMMSS"] = pd.to_datetime(
    df["time_YYMMDD_HHMMSS"], format="%Y-%m-%d %H:%M:%S"
)
df.set_index("time_YYMMDD_HHMMSS", inplace=True)

# Добавление циклических признаков
df["month"] = df.index.month
df["day_of_month"] = df.index.day
df["hour"] = df.index.hour

# Границы перебора параметров
lookback_range = range(3, 37, 3)  # уменьшено количество значений
neuron_options = [50, 100]  # уменьшено количество значений
layer_options = [1, 3, 5]  # уменьшено количество значений

# Параметры варьирования для лагов и окон
lag_steps_options = [1, 2, 4]  # уменьшено количество значений
rolling_window_min_max = [(2, 10), (3, 15)]  # уменьшено количество значений
expanding_window_min_max = [(2, 10), (3, 15)]  # уменьшено количество значений

# DataFrame для хранения результатов
results = []

# Папка для сохранения лучших моделей
os.makedirs("models", exist_ok=True)

best_r2 = -np.inf
best_model_path = None

# Подсчет количества возможных комбинаций для прогресс-бара
total_combinations = (
    len(lookback_range)
    * len(neuron_options)
    * len(layer_options)
    * len(lag_steps_options)
    * len(rolling_window_min_max)
    * len(expanding_window_min_max)
)

# Итеративный перебор всех возможных комбинаций с прогресс-баром
with tqdm(total=total_combinations) as pbar:
    for lookback_hours in lookback_range:
        for neurons in neuron_options:
            for layers in layer_options:
                for lag_step in lag_steps_options:
                    for roll_min, roll_max in rolling_window_min_max:
                        for exp_min, exp_max in expanding_window_min_max:
                            # Обновление данных перед каждой итерацией
                            df_transformed = cyclical_feature_encoding(
                                df.copy(),
                                ["WindDirection", "month", "day_of_month", "hour"],
                            )

                            pbar.set_description(
                                f"Параметры: lookback={lookback_hours}, neurons={neurons}, layers={layers}, lag_step={lag_step}"
                            )
                            # Вычисление горизонта прогноза и лагов
                            forecast_hours = 3
                            target_variable = "WindSpeed"
                            forecast_horizon, lookback_horizon = calculate_horizons(
                                df_transformed, forecast_hours, lookback_hours
                            )

                            lags = list(range(1, lookback_horizon, lag_step))
                            rolling_window_sizes = list(
                                range(
                                    roll_min, min(roll_max, lookback_horizon), lag_step
                                )
                            )
                            expanding_window_sizes = list(
                                range(exp_min, min(exp_max, lookback_horizon), lag_step)
                            )

                            # Генерация признаков
                            df_transformed = generate_time_series_features(
                                df_transformed,
                                columns=[
                                    "WindSpeedMax",
                                    "AirTemperature",
                                    "AirPressure",
                                    "AirHumidity",
                                    "WindSpeed",
                                    "WindDirection_sin",
                                    "WindDirection_cos",
                                ],
                                lags=lags,
                                rolling_window_sizes=rolling_window_sizes,
                                expanding_window_sizes=expanding_window_sizes,
                            )

                            # Разделение на обучающую и тестовую выборки
                            df_train, df_test = train_test_split(
                                df_transformed,
                                test_size=0.33,
                                random_state=random_seed,
                                shuffle=False,
                            )

                            # Подготовка данных
                            X_train, y_train = prepare_ml_data_from_features(
                                df_train,
                                target_variable=target_variable,
                                forecast_horizon=forecast_horizon,
                            )
                            X_test, y_test = prepare_ml_data_from_features(
                                df_test,
                                target_variable=target_variable,
                                forecast_horizon=forecast_horizon,
                            )

                            # Масштабирование данных
                            scaler_X = StandardScaler()
                            X_train_scaled = scaler_X.fit_transform(X_train)
                            X_test_scaled = scaler_X.transform(X_test)

                            scaler_y = StandardScaler()
                            y_train_scaled = scaler_y.fit_transform(y_train)
                            y_test_scaled = scaler_y.transform(y_test)

                            # Создание модели
                            model = Sequential()
                            model.add(
                                Dense(
                                    neurons,
                                    activation="relu",
                                    input_shape=(X_train_scaled.shape[1],),
                                )
                            )
                            for _ in range(layers - 1):
                                model.add(Dense(neurons, activation="relu"))
                            model.add(
                                Dense(forecast_horizon)
                            )  # Прогнозируем N шагов вперед

                            # Компиляция модели
                            model.compile(
                                optimizer=Adam(learning_rate=0.0001), loss="mse"
                            )

                            # Настройка коллбека для сохранения всей модели
                            model_checkpoint_path = f"models/temp_model_{lookback_hours}_{neurons}_{layers}_{lag_step}_{roll_min}_{roll_max}_{exp_min}_{exp_max}.keras"
                            checkpoint = ModelCheckpoint(
                                model_checkpoint_path,
                                monitor="val_loss",
                                save_best_only=True,
                                mode="min",
                                save_weights_only=False,  # Сохраняем всю модель
                            )

                            # Настройка коллбека для динамического изменения learning rate
                            reduce_lr = ReduceLROnPlateau(
                                monitor="val_loss",
                                factor=0.2,
                                patience=5,
                                min_lr=0.00001,
                            )

                            # Обучение модели
                            model.fit(
                                X_train_scaled,
                                y_train_scaled,
                                epochs=50,
                                batch_size=256,
                                validation_split=0.2,
                                callbacks=[checkpoint, reduce_lr],
                                verbose=0,
                            )

                            # Загрузка лучшей сохраненной модели
                            model = tf.keras.models.load_model(model_checkpoint_path)

                            # Прогнозирование
                            y_pred_scaled = model.predict(X_test_scaled)
                            y_pred = scaler_y.inverse_transform(y_pred_scaled)

                            # Оценка точности
                            r2 = r2_score(y_test, y_pred)
                            print(f"R^2 Score: {r2}")

                            # Сохранение результатов
                            results.append(
                                {
                                    "lookback_hours": lookback_hours,
                                    "neurons": neurons,
                                    "layers": layers,
                                    "lag_step": lag_step,
                                    "roll_min": roll_min,
                                    "roll_max": roll_max,
                                    "exp_min": exp_min,
                                    "exp_max": exp_max,
                                    "r2_score": r2,
                                }
                            )

                            # Обновление Excel-файла с результатами
                            results_df = pd.DataFrame(results)
                            results_df.to_excel("models/results.xlsx", index=False)

                            # Проверка на лучшую модель
                            if r2 > best_r2:
                                best_r2 = r2
                                if best_model_path:
                                    os.remove(
                                        best_model_path
                                    )  # Удаление предыдущей лучшей модели
                                best_model_path = model_checkpoint_path

                            # Обновление прогресс-бара
                            pbar.update(1)

print("Все итерации завершены.")
