# функция генерирования новых строк для балансировки классов
import numpy as np
def generate_balanced_data(X, y, k_neighbors=5, random_state=42):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)

    # if X.ndim != 2 or y.ndim != 1 or len(X) != len(y) or len(y) == 0:
    #     raise ValueError("Нужны непустая таблица X и одна метка y для каждой строки.")
    # if X.shape[1] == 0 or not np.isfinite(X).all():
    #     raise ValueError("Признаки должны быть числовыми, без пропусков и бесконечностей.")
    # if not isinstance(k_neighbors, (int, np.integer)) or k_neighbors < 1:
    #     raise ValueError("k_neighbors должен быть положительным целым числом.")

    rng = np.random.default_rng(random_state)
    classes, counts = np.unique(y, return_counts=True)
    target_count = counts.max()
    new_rows = []
    new_labels = []

    for label, count in zip(classes, counts):
        # Берём только реальные обучающие строки текущего класса.
        class_rows = X[y == label]
        n_generate = target_count - count

        for _ in range(n_generate):
            # Случайно выбираем исходную строку.
            base = class_rows[rng.integers(len(class_rows))]

            # Считаем расстояния до строк того же класса.
            distances = np.linalg.norm(class_rows - base, axis=1)
            # Сама строка и её точные копии не могут быть вторым родителем.
            distances[distances == 0] = np.inf
            available = np.isfinite(distances).sum()
            if available == 0:
                raise ValueError(
                    f"В классе {label!r} нет двух различных примеров для генерации."
                )

            # Выбираем одного из k ближайших соседей.
            k = min(k_neighbors, available)
            nearest_indices = np.argsort(distances)[:k]
            neighbor = class_rows[rng.choice(nearest_indices)]

            # Создаём промежуточную строку одним коэффициентом для всех признаков.
            gap = rng.random()
            new_row = base + gap * (neighbor - base)
            new_rows.append(new_row)
            new_labels.append(label)

    # Если классы уже равны, добавлять строки не требуется.
    if not new_rows:
        return X.copy(), y.copy()

    X_balanced = np.vstack([X, np.asarray(new_rows)])
    y_balanced = np.concatenate([y, np.asarray(new_labels, dtype=y.dtype)])
    return X_balanced, y_balanced