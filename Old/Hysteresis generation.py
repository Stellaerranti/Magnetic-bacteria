import numpy as np
import matplotlib.pyplot as plt

# Константы
gamma = 1.76e7  # Гиромагнитное отношение (рад/(с·Тл))
kappa = 0.1     # Коэффициент демпфирования
Ms = 450e3      # Насыщенная намагниченность (А/м)
H_max = 1e-3    # Максимальное внешнее поле (Тл)
steps = 100     # Число шагов для поля
dt = 1e-12      # Временной шаг (с)
num_steps = 1000  # Число временных шагов
D = 40e-9       # Диаметр частиц (м)
T_en = 5e-9     # Толщина немагнитной оболочки (м)

# Параметры ансамбля
num_chains = 5  # Число цепочек в ансамбле
N_p_range = (5, 20)  # Диапазон числа частиц в цепочках
a_range = (1.5 * D, 3 * D)  # Диапазон расстояний между центрами

# Генерация параметров цепочек
np.random.seed(42)  # Для воспроизводимости
N_p_list = np.random.randint(N_p_range[0], N_p_range[1] + 1, num_chains)
a_list = np.random.uniform(a_range[0], a_range[1], num_chains)

# Функция для расчета эффективного поля
def effective_field(H_ext, M, dipole_field):
    return H_ext + dipole_field - kappa * M / Ms

# Функция для расчета дипольного поля
def dipole_field(M_chain, a, N_p):
    H_dip = np.zeros(3)
    for i in range(N_p):
        for j in range(N_p):
            if i != j:
                r = np.array([0, 0, (i - j) * a])  # Расстояние между частицами
                r_norm = np.linalg.norm(r)
                H_dip += (3 * np.dot(M_chain[j], r) * r / r_norm**5 - M_chain[j] / r_norm**3)
    return H_dip / (4 * np.pi * 1e-7)  # Преобразование в Тл

# Генерация петли гистерезиса для каждой цепочки
H_vals = np.linspace(-H_max, H_max, steps)
M_vals_total = np.zeros(steps * 2)

for chain_idx in range(num_chains):
    N_p = N_p_list[chain_idx]
    a = a_list[chain_idx]
    
    M_chain = np.zeros((N_p, 3))  # Магнитные моменты частиц
    M_chain[:, 2] = Ms           # Начальная намагниченность вдоль оси Z
    M_vals_chain = []
    
    for H in H_vals:
        H_ext = np.array([0, 0, H])  # Внешнее поле вдоль оси Z
        for _ in range(num_steps):
            dipole_field_chain = np.zeros(3)
            for i in range(N_p):
                dipole_field_chain += dipole_field(M_chain, a, N_p)
            for i in range(N_p):
                H_eff = effective_field(H_ext, M_chain[i], dipole_field_chain)
                # Уравнение Ландау-Лифшица
                dM_dt = -gamma * np.cross(M_chain[i], H_eff) - kappa * gamma * np.cross(M_chain[i], np.cross(M_chain[i], H_eff))
                M_chain[i] += dM_dt * dt
                M_chain[i] = M_chain[i] / np.linalg.norm(M_chain[i]) * Ms  # Нормализация
        M_z = np.sum(M_chain[:, 2]) / N_p  # Средняя намагниченность вдоль оси Z
        M_vals_chain.append(M_z)
    
    # Отражение петли гистерезиса для цепочки
    M_vals_chain = np.concatenate([M_vals_chain, -np.array(M_vals_chain)])
    M_vals_total += M_vals_chain / num_chains  # Усреднение по ансамблю

# График петли гистерезиса ансамбля
H_vals_total = np.concatenate([H_vals, H_vals[::-1]])
plt.figure(figsize=(8, 6))
plt.plot(H_vals_total, M_vals_total, label=f"Ensemble Hysteresis Loop ({num_chains} chains)")
plt.xlabel("External Field H (T)")
plt.ylabel("Magnetization M (A/m)")
plt.title("Hysteresis Loop Simulation for Ensemble of Chains")
plt.grid()
plt.legend()
plt.show()