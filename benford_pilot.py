import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
data_dir = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)
# Фиксируем, чтобы симуляции можно было повторить
rng = np.random.default_rng(42)
digits = np.arange(1, 10)
# Вероятности первых цифр от 1 до 9 по Бенфорду
probs = np.log10(1 + 1 / digits)

def first_digit(x):
    x = np.asarray(x, dtype=float)
    # Убираем пропуски, бесконечности и нули
    x = np.abs(x[np.isfinite(x) & (x != 0)])
    return (x / 10 ** np.floor(np.log10(x))).astype(int)

def check(values, name):
    d = first_digit(values)
    n = len(d)
    count = np.array([(d == k).sum() for k in digits])
    freq = count / n
    mad = np.mean(np.abs(freq - probs))
    return {
        'name': name,
        'n': int(n),
        'obs': freq.round(4).tolist(),
        'mad': round(float(mad), 4),
    }

def plot(items, file, title):
    k = len(items)
    fig, axes = plt.subplots(1, k, figsize=(4.2 * k, 3.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, r in zip(axes, items):
        ax.bar(digits, r['obs'], color='#9db8d9', label='данные')
        ax.plot(digits, probs, 'o-', color='#c0392b', ms=4, label='Бенфорд')
        ax.set_title(f"{r['name']}\nn={r['n']}, MAD={r['mad']:.3f}", fontsize=9)
        ax.set_xticks(digits)
        ax.set_xlabel('первая цифра')
    axes[0].set_ylabel('доля')
    axes[0].legend(fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / file, dpi=160)
    plt.close(fig)

# Реальные данне
city = pd.read_csv(data_dir / 'city.csv')
pop = pd.to_numeric(city['population'], errors='coerce')
years = pd.to_numeric(city['foundation_year'], errors='coerce')
# Из ВВП исключаем перечисленное чтобы не смешивать страны с регионами.
gdp = pd.read_csv(data_dir / 'gdp.csv')
groups = {
    'AFE',
    'AFW',
    'ARB',
    'CEB',
    'CSS',
    'EAP',
    'EAR',
    'EAS',
    'ECA',
    'ECS',
    'EMU',
    'EUU',
    'FCS',
    'HIC',
    'HPC',
    'IBD',
    'IBT',
    'IDA',
    'IDB',
    'IDX',
    'INX',
    'LAC',
    'LCN',
    'LDC',
    'LIC',
    'LMC',
    'LMY',
    'LTE',
    'MEA',
    'MIC',
    'MNA',
    'NAC',
    'OED',
    'OSS',
    'PRE',
    'PSS',
    'PST',
    'SAS',
    'SSA',
    'SSF',
    'SST',
    'TEA',
    'TEC',
    'TLA',
    'TMN',
    'TSA',
    'TSS',
    'UMC',
    'WLD',
}
gdp = gdp[~gdp['Country Code'].isin(groups)]
# Последний год в файле может содержать данные не для всех стран.
year = int(gdp['Year'].max())
gdp_values = gdp[gdp['Year'] == year]['Value']
sp = pd.read_csv(data_dir / 'constituents-financials.csv')
cap = pd.to_numeric(sp['Market Cap'], errors='coerce')
ebitda = pd.to_numeric(sp['EBITDA'], errors='coerce')
# Здесь исследуем только положительную EBITDA.
ebitda = ebitda[ebitda > 0]
rows = [
    check(pop.dropna(), 'Население городов РФ'),
    check(gdp_values, f'ВВП стран, {year}'),
    check(cap.dropna(), 'Капитализация S&P 500'),
    check(ebitda, 'EBITDA S&P 500'),
    check(years.dropna(), 'Год основания городов РФ'),
]
plot(rows[:2], 'fig1_controls.png', 'Первые цифры населения и ВВП')
plot(rows[2:4], 'fig2_finance.png', 'Финансовые показатели (S&P 500)')
plot(rows[4:], 'fig3_counter.png', 'Первые цифры годов основания')

# 2. Отдельные траектории: на каждом шаге умножаем значение на случайный множитель.
runs, steps = (20000, 60)
x = np.ones(runs)
mads = []
for t in range(steps + 1):
    d = first_digit(x)
    freq = np.array([(d == k).mean() for k in digits])
    mads.append(float(np.mean(np.abs(freq - probs))))
    x = x * np.exp(rng.normal(0.05, 0.25, size=runs))
fig, ax = plt.subplots(figsize=(5.5, 3.2))
ax.plot(range(steps + 1), mads, color='#2c3e50')
ax.set_xlabel('шаг (например, год роста)')
ax.set_ylabel('MAD от Бенфорда')
ax.set_title('Распределение первых цифр при случайном росте', fontsize=9)
fig.tight_layout()
fig.savefig(out_dir / 'fig4_multiplicative.png', dpi=160)
plt.close(fig)

# 3. Казино выигрывает, если произведение начинается с 1, 2 или 3
N = 200000
# Логарифм до округления равномерен. Округление делает стратегию приближённой.
casino = np.floor(10 ** (3 + rng.random(N)))
players = {
    'всегда 2': np.full(N, 2.0),
    'всегда 5': np.full(N, 5.0),
    'равномерно 1000–9999': rng.integers(1000, 10000, N).astype(float),
    'равномерно 1–9': rng.integers(1, 10, N).astype(float),
}
wins = {}
for name, y in players.items():
    d = first_digit(casino * y)
    wins[name] = round(float(np.mean(d <= 3)), 4)
# Для сравнения оба игрока выбирают целые числа равномерно.
uniform = rng.integers(1000, 10000, N).astype(float)
uniform_win = round(float(np.mean(first_digit(uniform * rng.integers(1000, 10000, N)) <= 3)), 4)


# Сохраняем числа отдельно от графиков.
result = {
    'real': rows,
    'multiplicative': {'mad_start': mads[0], 'mad_end': mads[-1]},
    'casino_win_share': wins,
    'casino_win_if_uniform_vs_uniform': uniform_win,
    'benford_P_1to3': round(float(probs[:3].sum()), 4),
    'gdp_year': year,
}
(out_dir / 'results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
print(json.dumps(result, ensure_ascii=False, indent=2))
