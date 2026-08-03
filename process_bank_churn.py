"""
Модуль попередньої обробки даних для змагання
"Bank Customer Churn Prediction (DLU Course)" на Kaggle.

Цей модуль винесено з дослідницького ноутбука (HW 2.7, логістична
регресія) у окремий `.py` файл за прикладом з лекції "Майстер-клас
з перенесення коду з jupyter notebook у Python модуль", щоб потім
його можна було імпортувати в будь-який інший ноутбук (наприклад,
для дерева прийняття рішень) без копіювання коду.

Основні функції, які варто використовувати ззовні:
    * preprocess_data(raw_df, ...)      - обробка даних для тренування моделі
    * preprocess_new_data(new_df, ...)  - обробка нових даних (наприклад test.csv)
       з уже навченими scaler/encoder.

Решта функцій - допоміжні "цеглинки", кожна відповідає за один крок
обробки (розбиття даних, кодування категорій, масштабування тощо),
щоб код було легко читати, тестувати і повторно використовувати.
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Колонки, які не несуть корисної інформації для моделі:
# - id, CustomerId - технічні ідентифікатори рядка/клієнта;
# - Surname - практично унікальне для кожного клієнта прізвище
#   (сотні унікальних значень), яке не узагальнюється на нових
#   клієнтів і лише додає зайві one-hot колонки.
COLUMNS_TO_DROP = ["id", "CustomerId", "Surname"]
TARGET_COL = "Exited"


def split_train_val(
    raw_df: pd.DataFrame,
    target_col: str = TARGET_COL,
    val_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Розбиває сирий датафрейм на тренувальну і валідаційну частини.

    Використовує стратифікований розподіл (`stratify`) за цільовою
    колонкою, щоб частка клієнтів, які пішли (Exited=1), була
    однаковою і в train, і у val - інакше випадковий розподіл міг би
    "перекосити" вибірки, особливо коли класи незбалансовані
    (як у нашому випадку: ~20% Exited=1).

    Args:
        raw_df: сирий датафрейм з train.csv, включно з цільовою колонкою.
        target_col: назва цільової колонки, за якою стратифікуємо.
        val_size: частка даних, яка піде у валідаційний набір (0-1).
        random_state: фіксоване значення для відтворюваності розбиття.

    Returns:
        Кортеж (train_df, val_df).
    """
    train_df, val_df = train_test_split(
        raw_df,
        test_size=val_size,
        random_state=random_state,
        stratify=raw_df[target_col],
    )
    return train_df, val_df


def select_input_target_cols(
    df: pd.DataFrame,
    input_cols: List[str],
    target_col: str = TARGET_COL,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Розділяє датафрейм на ознаки (X) та ціль (y).

    Args:
        df: датафрейм (train_df або val_df).
        input_cols: перелік колонок, які підуть у X.
        target_col: назва колонки з ціллю.

    Returns:
        Кортеж (inputs, targets), де inputs - копія df[input_cols],
        targets - копія df[target_col].
    """
    inputs = df[input_cols].copy()
    targets = df[target_col].copy()
    return inputs, targets


def identify_numeric_categorical_cols(
    inputs_df: pd.DataFrame,
) -> Tuple[List[str], List[str]]:
    """Автоматично визначає числові та категоріальні колонки.

    Числові колонки - усе, що має тип np.number (int, float).
    Категоріальні - колонки типу object (текстові/рядкові дані).

    Args:
        inputs_df: датафрейм з ознаками (без цільової колонки).

    Returns:
        Кортеж (numeric_cols, categorical_cols) - списки назв колонок.
    """
    numeric_cols = inputs_df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = inputs_df.select_dtypes("object").columns.tolist()
    return numeric_cols, categorical_cols


def fit_encoder(train_inputs: pd.DataFrame, categorical_cols: List[str]) -> OneHotEncoder:
    """Навчає (fit) one-hot encoder на категоріальних колонках трену.

    `drop='if_binary'` означає: якщо категоріальна колонка має рівно
    два унікальних значення (як `Gender`: Male/Female), encoder
    створить лише ОДНУ колонку (0/1) замість двох - це прибирає
    надлишкову (лінійно залежну) інформацію.

    `handle_unknown='ignore'` захищає від помилки, якщо у нових
    даних (наприклад test.csv) раптом трапиться категорія, якої не
    було у train - тоді просто всі one-hot колонки для цього рядка
    будуть нулями, замість того щоб код впав з помилкою.

    Args:
        train_inputs: тренувальні ознаки (лише train, без val!) -
            щоб уникнути витоку інформації (data leakage) з
            валідаційного набору.
        categorical_cols: список категоріальних колонок для кодування.

    Returns:
        Навчений OneHotEncoder.
    """
    encoder = OneHotEncoder(sparse_output=False, drop="if_binary", handle_unknown="ignore")
    encoder.fit(train_inputs[categorical_cols])
    return encoder


def apply_encoder(
    inputs_df: pd.DataFrame,
    encoder: OneHotEncoder,
    categorical_cols: List[str],
) -> pd.DataFrame:
    """Застосовує вже навчений encoder до датафрейму і повертає новий df.

    Оригінальні категоріальні колонки видаляються, натомість
    додаються нові one-hot колонки з назвами на кшталт
    `Geography_Germany`, `Gender_Male` тощо.

    Args:
        inputs_df: датафрейм, до якого застосовуємо кодування
            (train, val або нові дані).
        encoder: OneHotEncoder, навчений функцією `fit_encoder`.
        categorical_cols: список категоріальних колонок, які кодуємо.

    Returns:
        Новий датафрейм з доданими one-hot колонками замість
        оригінальних категоріальних.
    """
    inputs_df = inputs_df.copy()
    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
    encoded_values = encoder.transform(inputs_df[categorical_cols])
    inputs_df[encoded_cols] = encoded_values
    inputs_df = inputs_df.drop(columns=categorical_cols)
    return inputs_df


def fit_scaler(train_inputs: pd.DataFrame, numeric_cols: List[str]) -> StandardScaler:
    """Навчає StandardScaler лише на числових колонках тренувального набору.

    StandardScaler віднімає середнє і ділить на стандартне відхилення,
    тобто приводить кожну числову ознаку до розподілу із середнім 0
    і стандартним відхиленням 1. Це важливо для моделей, чутливих до
    масштабу ознак (наприклад логістична регресія), але НЕ є
    обов'язковим для дерев рішень - вони порівнюють значення ознаки
    самі із собою (пороги), тому масштаб не впливає на побудову
    дерева. Саме тому цей крок зроблено опціональним (параметр
    `scaler_numeric` у `preprocess_data`).

    Args:
        train_inputs: тренувальні ознаки (лише train!).
        numeric_cols: список числових колонок для масштабування.

    Returns:
        Навчений StandardScaler.
    """
    scaler = StandardScaler()
    scaler.fit(train_inputs[numeric_cols])
    return scaler


def apply_scaler(
    inputs_df: pd.DataFrame,
    scaler: StandardScaler,
    numeric_cols: List[str],
) -> pd.DataFrame:
    """Застосовує вже навчений scaler до числових колонок датафрейму.

    Args:
        inputs_df: датафрейм, до якого застосовуємо масштабування.
        scaler: StandardScaler, навчений функцією `fit_scaler`.
        numeric_cols: список числових колонок для масштабування.

    Returns:
        Новий датафрейм з масштабованими числовими колонками.
    """
    inputs_df = inputs_df.copy()
    inputs_df[numeric_cols] = scaler.transform(inputs_df[numeric_cols])
    return inputs_df


def preprocess_data(
    raw_df: pd.DataFrame,
    scaler_numeric: bool = True,
    val_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
    List[str],
    Optional[StandardScaler],
    OneHotEncoder,
]:
    """Повний цикл попередньої обробки сирих даних для тренування моделі.

    Кроки:
        1. Обираємо колонки для роботи (прибираємо id, CustomerId,
           Surname - технічні/неінформативні ідентифікатори).
        2. Стратифіковано розбиваємо дані на train/val.
        3. Визначаємо числові та категоріальні колонки.
        4. One-hot кодуємо категоріальні колонки (encoder навчається
           лише на train, потім застосовується і до train, і до val -
           так ми уникаємо витоку інформації з валідаційного набору).
        5. (опційно) Масштабуємо числові колонки через StandardScaler
           - вимикається параметром `scaler_numeric=False`, бо для
           дерев рішень масштабування не потрібне.

    Args:
        raw_df: сирий датафрейм (як зчитаний з train.csv), обов'язково
            містить колонку `Exited` - цільову змінну.
        scaler_numeric: чи масштабувати числові ознаки StandardScaler.
            Для дерев рішень можна залишити False - результат моделі
            не зміниться, зате менше зайвих обчислень.
        val_size: частка даних для валідаційного набору.
        random_state: фіксоване значення для відтворюваності.

    Returns:
        Кортеж:
            X_train (pd.DataFrame) - оброблені тренувальні ознаки,
            train_targets (pd.Series) - тренувальна ціль,
            X_val (pd.DataFrame) - оброблені валідаційні ознаки,
            val_targets (pd.Series) - валідаційна ціль,
            input_cols (List[str]) - вихідний перелік "сирих" колонок,
                які пішли в X (до кодування/масштабування),
            scaler (StandardScaler | None) - навчений scaler, або None,
                якщо scaler_numeric=False,
            encoder (OneHotEncoder) - навчений one-hot encoder.
    """
    input_cols = [
        col for col in raw_df.columns if col not in COLUMNS_TO_DROP + [TARGET_COL]
    ]

    train_df, val_df = split_train_val(raw_df, TARGET_COL, val_size, random_state)

    train_inputs, train_targets = select_input_target_cols(train_df, input_cols, TARGET_COL)
    val_inputs, val_targets = select_input_target_cols(val_df, input_cols, TARGET_COL)

    numeric_cols, categorical_cols = identify_numeric_categorical_cols(train_inputs)

    encoder = fit_encoder(train_inputs, categorical_cols)
    train_inputs = apply_encoder(train_inputs, encoder, categorical_cols)
    val_inputs = apply_encoder(val_inputs, encoder, categorical_cols)

    scaler = None
    if scaler_numeric:
        scaler = fit_scaler(train_inputs, numeric_cols)
        train_inputs = apply_scaler(train_inputs, scaler, numeric_cols)
        val_inputs = apply_scaler(val_inputs, scaler, numeric_cols)

    return train_inputs, train_targets, val_inputs, val_targets, input_cols, scaler, encoder


def preprocess_new_data(
    new_df: pd.DataFrame,
    input_cols: List[str],
    scaler: Optional[StandardScaler],
    encoder: OneHotEncoder,
    scaler_numeric: bool = True,
) -> pd.DataFrame:
    """Обробляє нові дані (наприклад test.csv) уже навченими scaler/encoder.

    Використовується перед передбаченням моделі на нових даних:
    ми НЕ навчаємо (`fit`) scaler/encoder заново, а лише
    застосовуємо (`transform`) ті, що вже були навчені на train у
    функції `preprocess_data`. Це критично важливо - інакше нові дані
    оброблялися б за іншою "логікою", ніж дані, на яких навчалась
    модель, і передбачення були б некоректними.

    Args:
        new_df: нові сирі дані (наприклад, зчитаний test.csv).
        input_cols: перелік "сирих" колонок, отриманий з `preprocess_data`.
        scaler: навчений StandardScaler з `preprocess_data`, або None,
            якщо при навчанні `scaler_numeric=False`.
        encoder: навчений OneHotEncoder з `preprocess_data`.
        scaler_numeric: чи застосовувати масштабування (має збігатися
            з тим, що використовувалось у `preprocess_data`).

    Returns:
        Оброблений датафрейм ознак, готовий для подачі в `model.predict`.
    """
    new_inputs = new_df[input_cols].copy()
    numeric_cols, categorical_cols = identify_numeric_categorical_cols(new_inputs)

    new_inputs = apply_encoder(new_inputs, encoder, categorical_cols)

    if scaler_numeric:
        if scaler is None:
            raise ValueError(
                "scaler_numeric=True, але scaler=None. "
                "Переконайтесь, що при виклику preprocess_data() також було scaler_numeric=True."
            )
        new_inputs = apply_scaler(new_inputs, scaler, numeric_cols)

    return new_inputs
