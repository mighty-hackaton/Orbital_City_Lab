"""Функція побудови фіч для моделі прогнозу швидкості (v3, з курсом).
Використовувати ІДЕНТИЧНО до того, як рахувалось при тренуванні —
інакше model.predict() дасть сміттєвий результат.

Модель повертає ДЕЛЬТУ швидкості (км/год), не абсолютне значення:
    predicted_speed = last_speed + model.predict([features])[0]
"""
import numpy as np

MIN_SPEED_FOR_HEADING_KMH = 3.0

FEATURE_NAMES = [
    "v1", "v2", "v3", "v4", "v5",
    "d1", "d2", "d3", "d4",
    "mean_v", "std_v", "trend_slope",
    "turn1", "turn2", "turn3", "turn4",
    "mean_abs_turn", "max_abs_turn",
    "cum_turn", "last_turn",
]


def angle_diff(a2, a1):
    """Різниця кутів у діапазоні [-180, 180]."""
    return (a2 - a1 + 180) % 360 - 180


def build_features(speed_window, heading_window):
    """speed_window, heading_window: по 5 останніх значень
    (від найстарішого до найновішого). heading_window бери з поля
    "heading" останніх 5 UDP-пакетів. Повертає список з 20 чисел
    у порядку FEATURE_NAMES.
    """
    sw = list(speed_window)
    hw = list(heading_window)

    diffs = [sw[i + 1] - sw[i] for i in range(len(sw) - 1)]
    mean_v = float(np.mean(sw))
    std_v = float(np.std(sw))
    trend_slope = float(np.polyfit(range(len(sw)), sw, 1)[0])

    turns = []
    for i in range(len(hw) - 1):
        if sw[i] < MIN_SPEED_FOR_HEADING_KMH or sw[i + 1] < MIN_SPEED_FOR_HEADING_KMH:
            turns.append(0.0)
        else:
            turns.append(angle_diff(hw[i + 1], hw[i]))

    abs_turns = [abs(t) for t in turns]
    mean_abs_turn = float(np.mean(abs_turns))
    max_abs_turn = float(np.max(abs_turns))
    cum_turn = float(np.sum(abs_turns))
    last_turn = float(turns[-1])

    return sw + diffs + [mean_v, std_v, trend_slope] + turns + [
        mean_abs_turn, max_abs_turn, cum_turn, last_turn
    ]
