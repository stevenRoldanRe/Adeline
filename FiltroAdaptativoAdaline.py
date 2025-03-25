import numpy as np
import matplotlib.pyplot as plt

# Generamos una señal senoidal con ruido
np.random.seed(42)
t = np.linspace(0, 2*np.pi, 100)  # Tiempo
signal = np.sin(t)  # Señal original
noise = 0.3 * np.random.randn(100)  # Ruido
noisy_signal = signal + noise  # Señal con ruido

# Parámetros del ADALINE
lr = 0.01  # Tasa de aprendizaje
epochs = 50  # Número de iteraciones
w = np.random.randn()  # Peso inicial
b = np.random.randn()  # Bias inicial

# Entrenamiento del ADALINE
for _ in range(epochs):
    for i in range(len(t)):
        y_pred = w * t[i] + b  # Salida de ADALINE
        error = noisy_signal[i] - y_pred  # Cálculo del error
        w += lr * error * t[i]  # Ajuste del peso
        b += lr * error  # Ajuste del bias

# Generamos la señal filtrada
filtered_signal = w * t + b

# Graficamos resultados
plt.figure(figsize=(10, 5))
plt.plot(t, noisy_signal, label="Señal con ruido", linestyle="dotted")
plt.plot(t, signal, label="Señal original", linewidth=2)
plt.plot(t, filtered_signal, label="Señal filtrada (ADALINE)", linewidth=2)
plt.legend()
plt.title("Filtro Adaptativo con ADALINE")
plt.xlabel("Tiempo")
plt.ylabel("Amplitud")
plt.show()
