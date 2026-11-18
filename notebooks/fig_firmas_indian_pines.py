"""Genera document/figuras/firmas-indian-pines.pdf (Figura de firmas espectrales del protocolo).

Datos: Indian Pines (AVIRIS), repositorio de escenas hiperespectrales de la EHU
(data/Indian_pines_corrected.mat, 200 bandas, y data/Indian_pines_gt.mat).
Ejecutar desde la raíz del repositorio: python3 notebooks/fig_firmas_indian_pines.py
"""
import numpy as np, scipy.io as sio, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

X = sio.loadmat("data/Indian_pines_corrected.mat")["indian_pines_corrected"].astype(float)
gt = sio.loadmat("data/Indian_pines_gt.mat")["indian_pines_gt"]
clases = ["", "Alfalfa", "Maíz sin labranza", "Maíz labranza mínima", "Maíz", "Pasto-pastizal",
          "Pasto-árboles", "Pasto segado", "Heno en hileras", "Avena", "Soya sin labranza",
          "Soya labranza mínima", "Soya limpia", "Trigo", "Bosque", "Edificios-pasto-árboles",
          "Torres de acero"]

plt.rcParams.update({"font.size": 8})
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.5), gridspec_kw={"width_ratios": [1.4, 1]})
cmap = plt.get_cmap("tab20")

# (a) Firma media de cada clase etiquetada
for c in range(1, 17):
    ax[0].plot(X[gt == c].mean(0), lw=0.9, color=cmap(c - 1), label=clases[c])
ax[0].set_xlabel("Índice de banda"); ax[0].set_ylabel("Valor medido medio")
ax[0].set_title("(a) Firma media por clase", fontsize=8)

# (b) Firma media del maíz (clase 4) y su doble: iluminación doble simulada
m = X[gt == 4].mean(0)
ax[1].plot(m, color="black", lw=1, label="Maíz, $\\mathbf{x}$")
ax[1].plot(2 * m, color="black", lw=1, ls="--", label="$2\\mathbf{x}$")
ax[1].set_xlabel("Índice de banda"); ax[1].set_title("(b) Firma del maíz y su doble", fontsize=8)
ax[1].legend(fontsize=7, frameon=False)

fig.legend(*ax[0].get_legend_handles_labels(), loc="lower center", ncol=6, fontsize=6, frameon=False)
fig.tight_layout(rect=(0, 0.15, 1, 1))
fig.savefig("document/figuras/firmas-indian-pines.pdf")

cos = m @ (2 * m) / (np.linalg.norm(m) * np.linalg.norm(2 * m))
print("Distancia euclidiana x vs 2x:", round(float(np.linalg.norm(m - 2 * m))))
print("Ángulo espectral (rad):", float(np.arccos(np.clip(cos, -1, 1))))
