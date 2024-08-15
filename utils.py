import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.graph_objs as go
import plotly.subplots as sp
import os


def cyclical_feature_encoding(df, columns):
    """
    Выполняет циклическое кодирование признаков, преобразуя указанные столбцы в синус и косинус.

    Аргументы:
    df (pd.DataFrame): Входной датафрейм.
    columns (list of str): Список столбцов, для которых нужно выполнить циклическое кодирование.

    Возвращает:
    pd.DataFrame: Копия датафрейма с добавленными синусом и косинусом для указанных столбцов.
    """
    df_copy = df.copy()
    for column in columns:
        df_copy[f"{column}_sin"] = np.sin(
            2 * np.pi * df_copy[column] / df_copy[column].max()
        )
        df_copy[f"{column}_cos"] = np.cos(
            2 * np.pi * df_copy[column] / df_copy[column].max()
        )
        df_copy = df_copy.drop(columns=[column])

    return df_copy


def generate_time_series_features(
    df,
    columns,
    lags=[],
    rolling_window_sizes=[],
    expanding_window_sizes=[],
    include_diffs=False,
):
    df_features = df.copy()

    for column in columns:
        # Создание лагов в порядке от самого старого к новому
        if lags:
            for lag in sorted(lags, reverse=True):
                df_features[f"{column}_lag_{lag}"] = df[column].shift(lag)

        # Создание скользящих окон только для среднего, для указанных размеров окон
        if rolling_window_sizes:
            for window_size in rolling_window_sizes:
                df_features[f"{column}_rolling_mean_{window_size}"] = (
                    df[column].rolling(window=window_size).mean()
                )

        # Создание раскрывающихся окон для указанных размеров
        if expanding_window_sizes:
            for window_size in expanding_window_sizes:
                df_features[f"{column}_cum_mean_{window_size}"] = (
                    df[column].expanding(min_periods=window_size).mean()
                )

        # Создание разностей
        if include_diffs:
            df_features[f"{column}_diff"] = df[column].diff()

    # Удаление строк с NaN значениями
    df_features.dropna(inplace=True)

    return df_features


def check_time_delta(df):
    """
    Функция для вычисления дельты времени между наблюдениями в индексированном по времени DataFrame.
    Выдает ошибку, если дельты различаются.

    :param df: DataFrame с индексом времени
    :return: Дельта времени (в виде pd.Timedelta), если все дельты одинаковые
    """
    # Вычисляем разницу между соседними значениями индекса
    time_deltas = df.index.to_series().diff().dropna().unique()

    if len(time_deltas) == 1:
        return time_deltas[0]
    else:
        raise ValueError(
            "Обнаружены разные значения дельты времени между наблюдениями."
        )


def calculate_horizons(df, forecast_hours, lookback_hours):
    """
    Рассчитывает количество шагов для горизонта прогноза и максимально необходимый лаг на основе временных интервалов.

    :param df: DataFrame с индексом времени
    :param forecast_hours: Горизонт прогноза в часах (на сколько часов вперед прогнозируем)
    :param lookback_hours: Количество часов "до", которые используем для прогнозирования
    :return: Горизонт прогноза и окно "до" в шагах (forecast_horizon, lookback_horizon)
    """
    # Вычисляем дельту времени между наблюдениями
    delta = check_time_delta(df)

    # Вычисляем горизонт прогноза в шагах
    forecast_horizon = int(pd.Timedelta(hours=forecast_hours) / delta)

    # Вычисляем количество шагов для "lookback"
    lookback_horizon = int(pd.Timedelta(hours=lookback_hours) / delta)

    return forecast_horizon, lookback_horizon


def stitch_forecasts(predictions):
    """
    Объединяет прогнозируемые значения из массива, где каждый столбец представляет
    собой прогноз на разный временной шаг, сохраняя наложения.

    Аргументы:
    predictions (np.array): Массив формы (n_samples, n_steps), содержащий прогнозируемые значения.

    Возвращает:
    np.array: Объединенная последовательность в виде одномерного массива.
    """
    n_samples, n_steps = predictions.shape
    final_sequence = []

    # Итерация по каждому прогнозу (строке) в predictions
    for i in range(0, n_samples, n_steps):
        final_sequence.extend(predictions[i])

    return np.array(final_sequence)


def prepare_ml_data_from_features(df_features, target_variable, forecast_horizon=12):
    """
    Подготавливает данные для машинного обучения на основе уже сгенерированных временных признаков.

    :param df_features: DataFrame с уже сгенерированными временными признаками
    :param target_variable: Название целевой переменной
    :param forecast_horizon: Горизонт прогноза в шагах (количество шагов вперед)
    :return: X - признаки для модели, y - целевые значения для модели
    """
    # Создание y, содержащего только будущие значения
    y = np.array(
        [
            df_features[target_variable].shift(-i).values
            for i in range(1, forecast_horizon + 1)
        ]
    ).T

    # Удаление строк, содержащих NaN
    valid_idx = ~np.isnan(y).any(axis=1)
    y = y[valid_idx]

    # Приведение X к соответствующей длине y и удаление NaN
    X = df_features.iloc[valid_idx]

    return X, y


# Объединенная функция для вычисления метрик
def calculate_metrics(y_true, y_pred, y_train):
    metrics = {}

    # MASE
    naive_forecast = np.roll(y_train, 1)[1:]
    mae_naive = np.mean(np.abs(y_train[1:] - naive_forecast))
    metrics["MASE"] = np.mean(np.abs(y_true - y_pred)) / mae_naive

    # sMAPE
    metrics["sMAPE"] = 100 * np.mean(
        2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))
    )

    # RMSE
    metrics["RMSE"] = np.sqrt(mean_squared_error(y_true, y_pred))

    # MAE
    metrics["MAE"] = mean_absolute_error(y_true, y_pred)

    # R²
    metrics["R²"] = r2_score(y_true, y_pred)

    # MSE
    metrics["MSE"] = mean_squared_error(y_true, y_pred)

    # MAPE
    metrics["MAPE"] = 100 * np.mean(np.abs((y_true - y_pred) / y_true))

    return metrics


def plot_sample_with_features(
    X,
    y,
    target_variable,
    columns,
    y_pred_values=None,
    output_path=None,
    sample_index=None,
):
    """
    Функция для визуализации одного случайного примера из данных X и y с использованием Plotly и сохранения графика.

    :param X: Матрица признаков (включая лаги целевой переменной)
    :param y: Целевая переменная (например, ветер на 12 шагов вперед)
    :param target_variable: Название целевой переменной
    :param columns: Список всех колонок, которые будут визуализироваться
    :param y_pred_values: Прогнозируемые значения (если есть)
    :param output_path: Путь для сохранения графика
    :param sample_index: Индекс примера для визуализации (если None, выбирается случайный)
    """
    # Выбор случайного индекса, если sample_index не задан
    if sample_index is None:
        idx = np.random.randint(0, len(X))
    else:
        idx = sample_index

    # Фильтрация лагов для каждой колонки в columns и сортировка по убыванию
    lagged_features = {
        col: [col_name for col_name in X.columns if f"{col}_lag" in col_name]
        for col in columns
    }

    # Создание субплотов
    num_features = len(columns)
    fig = sp.make_subplots(
        rows=num_features, cols=1, shared_xaxes=False, vertical_spacing=0.05
    )

    for i, column in enumerate(columns):
        # Данные по лагам для текущей колонки и текущее значение
        x_lags = X[lagged_features[column] + [column]].iloc[idx].values

        # Добавление данных на график для текущего признака
        fig.add_trace(
            go.Scatter(
                x=list(range(len(x_lags))),
                y=x_lags,
                mode="lines+markers",
                name=f"{column} (lags and current)",
                line=dict(color="green"),
                showlegend=(i == 0),
            ),
            row=i + 1,
            col=1,
        )

        if column == target_variable:
            # Значения y для прогноза, только для целевой переменной
            y_values = y[idx]

            fig.add_trace(
                go.Scatter(
                    x=list(range(len(x_lags), len(x_lags) + len(y_values))),
                    y=y_values,
                    mode="lines+markers",
                    name=f"{column} (Future)",
                    line=dict(color="red"),
                    showlegend=(i == 0),
                ),
                row=i + 1,
                col=1,
            )

            if y_pred_values is not None:
                fig.add_trace(
                    go.Scatter(
                        x=list(
                            range(len(x_lags), len(x_lags) + len(y_pred_values[idx]))
                        ),
                        y=y_pred_values[idx],
                        mode="lines+markers",
                        name=f"{column} (Predicted)",
                        line=dict(color="blue"),
                        showlegend=(i == 0),
                    ),
                    row=i + 1,
                    col=1,
                )

        # Вертикальная линия для разделения лагов и прогнозов
        fig.add_vline(
            x=len(x_lags) - 1, line=dict(color="gray", dash="dash"), row=i + 1, col=1
        )

    # Обновление разметки и отображение графика
    fig.update_layout(
        height=num_features * 400,
        width=1500,
        title_text="Sample Visualization with Features",
    )

    # Сохранение графика
    if output_path is not None:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        output_file = os.path.join(output_path, f"sample_{idx}.png")
        fig.write_image(output_file)
        print(f"График сохранен: {output_file}")
    else:
        fig.show()
