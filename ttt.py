import os
import pandas as pd
import warnings
import numpy as np
import tensorflow as tf
import random
import itertools
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam
from utils import (
    calculate_metrics,
    cyclical_feature_encoding,
    generate_time_series_features,
    calculate_horizons,
    prepare_ml_data_from_features,
)

# Фиксация random seed для воспроизводимости
random_seed = 42
np.random.seed(random_seed)
tf.random.set_seed(random_seed)
random.seed(random_seed)

# Фиксация сида для Python OS-level random functions
os.environ["PYTHONHASHSEED"] = str(random_seed)

# Ограничение влияния других факторов на выполнение кода
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# Игнорировать все предупреждения
warnings.filterwarnings("ignore")

# Загрузка данных
file_path = "Meteo1-2023-15min(resampled).csv"
df = pd.read_csv(file_path, delimiter=",")

# Преобразование времени в datetime и установка индекса
df["time_YYMMDD_HHMMSS"] = pd.to_datetime(
    df["time_YYMMDD_HHMMSS"], format="%Y-%m-%d %H:%M:%S"
)
df.set_index("time_YYMMDD_HHMMSS", inplace=True)

# Добавление новых признаков
df["month"] = df.index.month
df["day_of_month"] = df.index.day
df["hour"] = df.index.hour

# Циклическое кодирование признаков
df_transformed = cyclical_feature_encoding(
    df, ["WindDirection", "month", "day_of_month", "hour"]
)

# Параметры для поиска
lookback_hours_list = [3, 6, 12]
lag_step_list = [1, 2, 3]
rolling_window_min_list = [1, 5, 10]
rolling_window_max_list = [10, 25, 50]
expanding_window_min_list = [1, 5, 10]
expanding_window_max_list = [10, 25, 50]
hidden_layers_list = [5]
neurons_list = [25, 100]

# Результаты для сохранения
results = []

# Прогресс бар
total_combinations = len(
    list(
        itertools.product(
            lookback_hours_list,
            lag_step_list,
            rolling_window_min_list,
            rolling_window_max_list,
            expanding_window_min_list,
            expanding_window_max_list,
            hidden_layers_list,
            neurons_list,
        )
    )
)
pbar = tqdm(total=total_combinations)

# Поиск по сетке
for (
    lookback_hours,
    lag_step,
    rolling_window_min,
    rolling_window_max,
    expanding_window_min,
    expanding_window_max,
    hidden_layers,
    neurons,
) in itertools.product(
    lookback_hours_list,
    lag_step_list,
    rolling_window_min_list,
    rolling_window_max_list,
    expanding_window_min_list,
    expanding_window_max_list,
    hidden_layers_list,
    neurons_list,
):

    # Вычисление горизонта прогноза и лагов
    forecast_horizon, lookback_horizon = calculate_horizons(
        df_transformed, forecast_hours=3, lookback_hours=lookback_hours
    )

    lags = list(range(1, lookback_horizon, lag_step))
    rolling_window_sizes = list(
        range(rolling_window_min, min(rolling_window_max, lookback_horizon), lag_step)
    )
    expanding_window_sizes = list(
        range(
            expanding_window_min, min(expanding_window_max, lookback_horizon), lag_step
        )
    )

    # Генерация признаков
    df_transformed_features = generate_time_series_features(
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

    # Разделение данных на обучающую и тестовую выборки
    df_train, df_test = train_test_split(
        df_transformed_features, test_size=0.1, random_state=42, shuffle=False
    )

    # Подготовка данных для машинного обучения
    X_train, y_train = prepare_ml_data_from_features(
        df_train, target_variable="WindSpeed", forecast_horizon=forecast_horizon
    )
    X_test, y_test = prepare_ml_data_from_features(
        df_test, target_variable="WindSpeed", forecast_horizon=forecast_horizon
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
    model.add(Dense(neurons, activation="relu", input_shape=(X_train_scaled.shape[1],)))
    for _ in range(hidden_layers - 1):
        model.add(Dense(neurons, activation="relu"))
    model.add(Dense(y_train_scaled.shape[1]))  # Прогнозируем N шагов вперед

    model.compile(optimizer=Adam(learning_rate=0.00005), loss="mse")

    # Настройка коллбеков
    checkpoint = ModelCheckpoint(
        "best_model.weights.h5",
        monitor="val_loss",
        save_best_only=True,
        mode="min",
        save_weights_only=True,
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=5, min_lr=0.00001
    )
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=25, restore_best_weights=True
    )

    # Обучение модели
    model.fit(
        X_train_scaled,
        y_train_scaled,
        epochs=150,
        batch_size=256,
        validation_split=0.2,
        callbacks=[checkpoint, reduce_lr, early_stopping],
        verbose=0,
    )

    # Загрузка лучших сохраненных весов
    model.load_weights("best_model.weights.h5")

    # Прогнозирование
    y_pred_scaled = model.predict(X_test_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)

    # Вычисление метрик
    metrics = calculate_metrics(y_test, y_pred, y_train)

    # Сохранение результатов
    results.append(
        {
            "lookback_hours": lookback_hours,
            "lag_step": lag_step,
            "rolling_window_min": rolling_window_min,
            "rolling_window_max": rolling_window_max,
            "expanding_window_min": expanding_window_min,
            "expanding_window_max": expanding_window_max,
            "hidden_layers": hidden_layers,
            "neurons": neurons,
            **metrics,
        }
    )

    # Обновление прогресс бара
    pbar.update(1)

# Закрытие прогресс бара
pbar.close()

# Сохранение результатов в Excel
results_df = pd.DataFrame(results)
results_df.to_excel("grid_search_results.xlsx", index=False)

print("Grid search completed and results saved to 'grid_search_results.xlsx'.")
