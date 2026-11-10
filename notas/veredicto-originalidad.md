# Veredicto de originalidad

*Borrador del 10 de noviembre de 2026, a partir de las fichas del grupo 6. Para revisar y enviar a la Dra. Li.*

## 1. Comparación con los trabajos más cercanos

| | EMS-GCN (Zhang 2022) | FG-DGCN (Li 2023) | MSGTN (Wang 2025) | Li et al. (2026) | **Mi propuesta** |
|---|---|---|---|---|---|
| Tarea | Clasificación | Clasificación | Clasificación | Agrupamiento | Clasificación |
| Clasificador | GCN mixhop | GCN dinámica | GCN multiescala + transformer | — (ADMM desplegado) | GCN semisupervisada |
| Superpíxeles | Aprendidos (SSN) | Aprendidos (SSN) | Aprendidos (SSN, pirámide) | Aprendidos (SSN) | Aprendidos (SSN) |
| Señal para aprenderlos | Entropía cruzada + reconstrucción de **etiquetas** + compacidad espacial | Entropía cruzada + reconstrucción de **etiquetas** | Solo entropía cruzada | **Sin etiquetas:** autorrepresentación + compacidad espectral (≈ EV) + consistencia local | **Sin etiquetas:** EV explícita + compacidad |
| Término espectral sin etiquetas | No | No | No | Sí, como término auxiliar, sin nombrarlo EV | Sí, es el criterio central |
| Distancia de asignación | Euclidiana² (características + posición) | Euclidiana (características) | Euclidiana (características)* | Euclidiana² (espectro + posición) | **Angular (tipo SAM)** |
| Etiquetas por clase | 30 | ≥ 100 (200 en GF-5 y Chris) | 1–5 % (1 a 123 en IP) | 0 | **5** |
| Mide ASA, BR o EV | Por verificar | No | No | No | **Sí, y su relación con la OA** |
| Análisis por tipo de escena | Por verificar | No | No | No | **Sí** |
| Conjuntos en común con los míos | IP, PU | Ninguno | IP, PU, SA, HongHu | SA | — |

\* La ecuación de asignación de MSGTN está por confirmar en el PDF original.

## 2. Veredicto

**El hueco sigue abierto, pero es más estrecho de lo que dice el protocolo actual.** Hay que corregir dos afirmaciones:

1. «Los algoritmos actuales forman los superpíxeles mediante reglas fijas» ya no es cierto. Tres trabajos de clasificación (EMS-GCN, FG-DGCN y MSGTN) aprenden superpíxeles diferenciables dentro de la red, aunque **siempre con la señal de las etiquetas**.
2. Tampoco puedo decir que soy el primero en aprender superpíxeles para HSI con una pérdida espectral sin etiquetas. **Li et al. (2026) ya incluye un término equivalente a la EV**, aunque para agrupamiento, como un término auxiliar y sin medirlo.

**Hueco reformulado (2–3 oraciones):**

> Los métodos que aprenden superpíxeles para clasificar imágenes hiperespectrales con GCN lo hacen con la señal de las etiquetas (entropía cruzada o reconstrucción de etiquetas), que con muy pocas muestras por clase es escasa, y ninguno mide la calidad de la segmentación que produce. Por otra parte, Galarza (2026) mostró que la varianza explicada (EV) de los superpíxeles predice la exactitud de clasificación mejor que el ASA. Ningún trabajo usa la EV como criterio explícito y sin etiquetas para aprender los superpíxeles de un clasificador semisupervisado, ni evalúa si maximizarla se traduce en mayor exactitud con 5 etiquetas por clase.

**Lo que sostiene la originalidad**, en orden de solidez:

1. **El criterio y su justificación:** la EV como objetivo explícito, motivado por el hallazgo empírico de Galarza (2026), para alimentar una GCN semisupervisada. Li et al. (2026) optimizan algo equivalente, pero sin esa motivación, para otra tarea, y su ablación muestra que ese término solo no mejora el agrupamiento de forma consistente (Salinas baja de 78,90 a 77,86 %).
2. **La afinidad angular (tipo SAM) en la asignación diferenciable:** ninguno de los cuatro la usa; todos usan la distancia euclidiana.
3. **La evaluación:** ninguno mide ASA, BR o EV de los superpíxeles aprendidos ni su relación con la OA, ni analiza por tipo de escena.
4. **El régimen de etiquetas:** 5 por clase, más extremo que el de todos los trabajos de clasificación revisados (30, ≥100 o 1–5 %).

**Riesgo que debo reconocer:** la ablación de Li et al. (2026) sugiere que maximizar solo un término tipo EV puede no mejorar la tarea final. Mi diseño experimental debe permitir que la respuesta sea «no»: la hipótesis se contrasta comparando contra SLIC y PCNN-SAM y midiendo la relación EV–OA.

## 3. Qué cambia en el protocolo

**Planteamiento del problema (sección 3)**
- Sustituir la frase de las «reglas fijas» por una que reconozca los superpíxeles aprendidos con etiquetas (citando EMS-GCN, FG-DGCN y MSGTN) y diga qué les falta.
- Citar a Li et al. (2026) como el único trabajo con una señal espectral sin etiquetas, y aclarar la diferencia: agrupamiento, término auxiliar, sin medir su efecto.
- Usar la cita de Li et al. (2026): los superpíxeles diferenciables se han aplicado «mainly in supervised tasks where gradients are derived from ground-truth labels».

**Objetivos (sección 4)**
- Objetivos 2 y 3 (afinidad angular y pérdida EV): se mantienen; son el núcleo original. En el 3, reconocer que la pérdida de reconstrucción viene de SSN y que lo nuevo es reconstruir el espectro sin etiquetas, maximizando la EV.
- Objetivo 5 (relación entre métricas y OA): reforzarlo, porque ningún trabajo lo hace con superpíxeles aprendidos.
- Objetivo 7 (extremo a extremo con etiquetas): mantenerlo como **exploratorio**, porque es lo que ya hacen EMS-GCN, FG-DGCN y MSGTN, y compararlo contra ellos.
- Objetivo 8 (HPC): sin cambios.

**Estado del arte (sección 8)**
- «Trabajos relacionados C»: un párrafo por cada uno de los cuatro (los borradores están en las fichas) y un cierre con el hueco reformulado.
- Cuadro comparativo: usar las columnas de la tabla de la sección 1.

## 4. Pendientes antes de enviarlo

- [ ] EMS-GCN: verificar en el PDF la OA en Indian Pines, si reporta métricas de segmentación y si hay otros tamaños de entrenamiento.
- [ ] MSGTN: confirmar en el PDF original en inglés la ecuación de asignación píxel–superpíxel.
- [ ] Li et al. (2026): confirmar en el código si F usa la asignación dura y si existe la versión final de ICASSP con cambios respecto al preprint.
- [ ] Revisar la ficha de cui2024superpixel (falta en el archivo de fichas).
