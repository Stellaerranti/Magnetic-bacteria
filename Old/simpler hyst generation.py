import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Параметры цепочек и магнетита
num_chains = 5  # Количество цепочек
max_grains_per_chain = 20  # Максимальное число зерен в цепочке
field_steps = 100  # Число шагов по внешнему полю
Bc_max = 0.2  # Максимальная коэрцитивная сила (T)
Bu_max = 0.05  # Максимальное взаимодействующее поле (T)


def llg_solver(m, H_eff, alpha=0.1, gamma=1.76e11, dt=1e-12, steps=100):
    """
    Решает уравнение Ландау-Лифшица-Гильберта для одного зерна.

    m: начальное направление магнетизации (нормализованный вектор)
    H_eff: эффективное магнитное поле (в Теслах)
    alpha: параметр демпфирования
    gamma: гиромагнитное отношение
    dt: шаг времени
    steps: количество итераций
    """
    for _ in range(steps):
        mxH = np.cross(m, H_eff)
        m += -gamma * (1 / (1 + alpha**2)) * (mxH + alpha * np.cross(m, mxH)) * dt
        m /= np.linalg.norm(m)  # нормализация вектора
    return m


def simulate_grain_response(Hc, Hu):
    """
    Упрощенная модель отклика зерна на внешнее поле.
    Hc: коэрцитивная сила
    Hu: взаимодействующее поле
    """
    return np.exp(-(Hc**2 + Hu**2) / 0.01)

# Генерация цепочек и расчет FORC
Hc_values = np.linspace(0, Bc_max, field_steps)
Hu_values = np.linspace(-Bu_max, Bu_max, field_steps)
Hc_grid, Hu_grid = np.meshgrid(Hc_values, Hu_values)
forc_diagram = np.zeros_like(Hc_grid, dtype=np.float64)

for chain in tqdm(range(num_chains), desc="Simulating Chains"):
    num_grains = np.random.randint(5, max_grains_per_chain)
    m_chain = np.array([0, 0, 1], dtype=np.float64)  # начальное направление магнетизации
    for grain in range(num_grains):
        Hc_grain = np.random.uniform(0, Bc_max)
        Hu_grain = np.random.uniform(-Bu_max, Bu_max)

        # Генерация эффективного поля для текущего зерна
        H_eff = np.array([Hu_grain, 0, Hc_grain], dtype=np.float64)
        m_chain = llg_solver(m_chain, H_eff)

        # Рассчитываем вклад от текущего зерна
        forc_diagram += simulate_grain_response(Hc_grid - Hc_grain, Hu_grid - Hu_grain)

# Нормализация диаграммы
forc_diagram /= forc_diagram.max()

# Визуализация FORC-диаграммы
plt.figure(figsize=(8, 6))
plt.contourf(Hc_grid, Hu_grid, forc_diagram, levels=50, cmap="coolwarm")
plt.colorbar(label="FORC Intensity")
plt.xlabel("Hc (T)")
plt.ylabel("Hu (T)")
plt.title("Simulated FORC Diagram for Magnetite Chains (LLG Model)")
plt.show()