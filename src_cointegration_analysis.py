"""
Коинтеграционный анализ демографической динамики РФ
Метод Энгла–Грейнджера, регрессия, тест Дики–Фуллера, прогноз.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Настройки
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 5)

# Создание папок для результатов
OUTPUT_DIR = Path("../outputs")
FIGURES_DIR = OUTPUT_DIR / "figures"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
FIGURES_DIR.mkdir(exist_ok=True, parents=True)

# -------------------------------------------------------------
# 1. Загрузка данных
# -------------------------------------------------------------
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df['year'] = df['year'].astype(int)
    return df

# -------------------------------------------------------------
# 2. Регрессия и тест коинтеграции
# -------------------------------------------------------------
def engle_granger_step(y, x):
    """Возвращает модель, остатки, ADF-статистику для остатков."""
    X_const = sm.add_constant(x)
    model = sm.OLS(y, X_const).fit()
    residuals = model.resid
    adf_stat, pvalue, _, _, crit_vals, _ = adfuller(residuals, autolag='AIC')
    return model, residuals, adf_stat, pvalue, crit_vals

# -------------------------------------------------------------
# 3. Прогноз
# -------------------------------------------------------------
def forecast(model, new_births):
    a = model.params['const']
    b = model.params['births']
    return a + b * new_births

# -------------------------------------------------------------
# 4. Графики
# -------------------------------------------------------------
def save_plots(df, model, residuals):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Население
    axes[0,0].plot(df['year'], df['population'], marker='o', color='green')
    axes[0,0].set_title('Динамика населения РФ (2015–2024)')
    axes[0,0].set_ylabel('млн чел.')

    # Рождаемость и смертность
    axes[0,1].plot(df['year'], df['births'], marker='s', label='Рождаемость')
    axes[0,1].plot(df['year'], df['deaths'], marker='^', label='Смертность')
    axes[0,1].set_title('Рождаемость и смертность')
    axes[0,1].legend()

    # Коинтеграционное поле
    axes[1,0].scatter(df['births'], df['population'], label='Факт')
    axes[1,0].plot(df['births'], model.fittedvalues, 'r', label='Регрессия')
    axes[1,0].set_xlabel('Рождаемость, млн')
    axes[1,0].set_ylabel('Население, млн')
    axes[1,0].legend()

    # Остатки
    axes[1,1].plot(df['year'], residuals, marker='d', color='purple')
    axes[1,1].axhline(y=0, color='black', linestyle='--')
    axes[1,1].set_title('Остатки модели')

    plt.tight_layout()
    fig_path = FIGURES_DIR / "cointegration_plots.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    return fig_path

# -------------------------------------------------------------
# 5. Главная функция
# -------------------------------------------------------------
def main(csv_file):
    print("Загрузка данных...")
    df = load_data(csv_file)

    Y = df['population']
    X = df['births']

    print("Оценка модели Энгла–Грейнджера...")
    model, residuals, adf_stat, adf_p, crit_vals = engle_grainger_step(Y, X)

    # Вывод в консоль и в файл
    summary_lines = []
    summary_lines.append("="*60)
    summary_lines.append("КОИНТЕГРАЦИОННЫЙ АНАЛИЗ ДЕМОГРАФИИ РФ")
    summary_lines.append("="*60)
    summary_lines.append(f"Период: {df['year'].min()}–{df['year'].max()}")
    summary_lines.append(f"Уравнение: Население = {model.params['const']:.2f} + ({model.params['births']:.2f}) × Рождаемость")
    summary_lines.append(f"R² = {model.rsquared:.4f} | p-value (F) = {model.f_pvalue:.5f}")
    summary_lines.append("\nТест Дики–Фуллера для остатков:")
    summary_lines.append(f"  ADF-статистика = {adf_stat:.4f}")
    summary_lines.append(f"  p-value = {adf_p:.4f}")
    summary_lines.append("  Критические значения:")
    for k, v in crit_vals.items():
        summary_lines.append(f"    {k}: {v:.4f}")
    if adf_p < 0.05:
        summary_lines.append("\nВывод: остатки стационарны → коинтеграция присутствует.")
    else:
        summary_lines.append("\nВывод: остатки нестационарны → коинтеграция отсутствует.")

    # Прогноз
    births_future = 1.18
    forecast_pop = forecast(model, births_future)
    summary_lines.append(f"\nПрогноз на 1 период: при рождаемости {births_future} млн → население = {forecast_pop:.2f} млн")

    # Сохраняем текстовый отчёт
    report_path = OUTPUT_DIR / "results_summary.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(summary_lines))

    # Печатаем в консоль
    print("\n".join(summary_lines))

    # Рисуем и сохраняем графики
    fig_path = save_plots(df, model, residuals)
    print(f"\nГрафики сохранены: {fig_path}")
    print(f"Текстовый отчёт: {report_path}")

if __name__ == "__main__":
    # Путь к CSV относительно папки src
    data_file = Path("../data/rf_demography_2015_2024.csv")
    if not data_file.exists():
        sys.exit(f"Ошибка: файл {data_file} не найден. Убедитесь, что структура проекта соблюдена.")
    main(data_file)