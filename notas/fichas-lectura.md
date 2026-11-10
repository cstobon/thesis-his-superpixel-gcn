# Fichas de lectura

Una ficha por artículo, en el orden de la pestaña Lecturas (grupo 6 primero). Llena los campos con tus palabras; deja «—» si no aplica.

- **Prioridad A**: ficha completa (pasadas 1 y 2).
- **Prioridad B**: ficha completa si haces la pasada 2; si no, solo «Problema», «Superpíxeles», «Señal» y «Para el protocolo».
- **Prioridad C**: solo «Dato que citarás».
- **Grupo 8 (HPC)**: usa los campos de costo computacional.

Los campos son las columnas del cuadro comparativo (sección 8 del protocolo): llenar la ficha ya es llenar el cuadro.

**Leyenda.** Superpíxeles: F = fijos (SLIC, ERS, PCNN-SAM…), A = aprendidos (SSN, FCN…), — = no usa. Señal: clasificación (necesita etiquetas), sin etiquetas (reconstrucción, EV, compacidad…), — = no aplica.

---

## 6 · Superpíxeles aprendidos en HSI: leer primero

*Definen si tu hueco sigue abierto. Van al veredicto del 7 nov y a «Trabajos relacionados C».*

### li2026joint · Prioridad A

Li et al. (2026). Joint Superpixel and Self-Representation Learning for Scalable Hyperspectral Image Clustering. *ICASSP 2026 - 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*. <http://dx.doi.org/10.1109/ICASSP55912.2026.11464647>

> Por qué está en la lista: 2026: superpíxeles diferenciables entrenados sin etiquetas para HSI, pero con fines de agrupamiento (clustering), no de clasificación con GCN.

- **Pasada 1 (9 de noviembre de 2026):** 
- **Pasada 2 (10 de noviembre de 2026, PDF arXiv 2509.24027v1, 5 pp.):** 
- **¿Amenaza tu originalidad? (sí / no / parcial, y por qué):** 
Parcial, y más cercana de lo que parece. Igual que mi propuesta, aprende superpíxeles diferenciables (tipo SSN) sin etiquetas sobre el espectro completo. Su pérdida de superpíxeles incluye un término de «compacidad espectral», (1/N) Σ‖X'ᵢ − Fᵢ‖² (Ec. 5), que es el error de reconstruir cada píxel con la media de su superpíxel; minimizarlo equivale prácticamente a maximizar la varianza explicada (EV = 1 − SSE intra/SST), aunque ellos no lo nombran así. Las diferencias que sí sostienen mi propuesta: (1) la EV es para mí el criterio central y explícito, mientras que en ellos ese término va mezclado con la consistencia local, la autorrepresentación y el ruido (Ec. 16); (2) ellos resuelven agrupamiento (clustering) por subespacios con una red ADMM desplegada, no clasificación semisupervisada con GCN; (3) no evalúan en Indian Pines; (4) su distancia es euclidiana, no angular, y no usan un extractor de características (trabajan sobre X' = X + δ); (5) no relacionan la calidad de los superpíxeles con la exactitud final, que es el hallazgo de Galarza (2026) en el que me apoyo. Consecuencia para el veredicto: ya no puedo afirmar que soy el primero en aprender superpíxeles diferenciables para HSI con una pérdida espectral sin etiquetas; el hueco debe formularse sobre la EV como criterio explícito para clasificación semisupervisada con GCN y pocas etiquetas.
- **Problema que resuelve:** 
El costo computacional y de memoria del agrupamiento por subespacios en HSI (crece cuadráticamente con el número de píxeles), y que los superpíxeles se generan como preprocesamiento independiente del agrupamiento, lo que produce particiones desalineadas con el objetivo (Sec. I). El agrupamiento se motiva por la falta de etiquetas en HSI.
- **Cómo forma los superpíxeles (F / A / —, y con qué método):** 
A. Asignación suave diferenciable tipo SSN: softmax de −d/τ sobre los 9 superpíxeles más cercanos (Ec. 3) y centroides como promedio ponderado (Ec. 4). Cada superpíxel aprende su propio parámetro de compacidad wⱼ, que equilibra la distancia espectral y la espacial (Sec. II-B). El número de superpíxeles M es el máximo entre una estimación por bordes y la cota 50C/R (Sec. III-A).
- **Señal con la que los aprende (clasificación / sin etiquetas / —, y qué pérdida):** 
Sin etiquetas. L_all = α·L_rep + L_spixel + L_noise (Ec. 16): L_rep = autorrepresentación dispersa (reconstrucción ‖ŜC − Ŝ‖², ℓ1 y entropía, Ec. 15) calculada por una red ADMM desplegada sobre los centroides; L_spixel = compacidad espectral + consistencia local (Ec. 5). Detalle: F se construye con el mapa de etiquetas duras L (cada píxel toma el centroide de su superpíxel), así que el gradiente del término espectral pasa por los centroides S y por δ, no por la asignación suave P [confirmar en el código]; L_noise regulariza el residuo δ (Ec. 17). Ojo: α se ajusta por conjunto de datos mirando la exactitud (Sec. III-C, Fig. 2) y M usa el número de clases C, así que no es del todo libre de etiquetas.
- **Espectro completo o PCA:** 
Espectro completo, X' = X + δ con un residuo aprendible δ (Sec. II-A); no menciona PCA.
- **Distancia (euclidiana / angular / otra):** 
Euclidiana al cuadrado, ponderada entre lo espectral y lo espacial: dᵢⱼ = wⱼ‖X'ᵢ − Sⱼ‖² + (1 − wⱼ)‖r(X'ᵢ) − r(Sⱼ)‖² (Ec. 2).
- **Etiquetas por clase:** 0 (no supervisado).
- **Conjuntos de datos:** Trento, Salinas y Urban (Tabla I). No usa Indian Pines.
- **OA en Indian Pines (y con cuántas etiquetas):** — (no lo evalúa). Como referencia, OA de agrupamiento: Salinas 80,96 %, Trento 90,21 %, Urban 84,52 % (Tabla II).
- **Código:** https://github.com/lxlscut/superpixel_ssc
- **Métricas de los superpíxeles (¿reporta ASA, BR o EV, o solo OA, AA y κ?):** 
Ninguna métrica de segmentación: solo OA, NMI y κ del agrupamiento (Tablas II y III). No reporta el valor de su término espectral ni de la EV.
- **¿Analiza por tipo de escena? ¿Publica su código?:** 
No hay un análisis por tipo de escena; solo comenta que Salinas es más sensible a α que Trento (Fig. 2). Código: https://github.com/lxlscut/superpixel_ssc
- **Dato clave de la ablación (Tabla III):** M1 = superpíxeles iniciales sin entrenar; M2 = superpíxeles entrenados solo con L_spixel + L_noise (es decir, con el término tipo EV, sin la autorrepresentación). M2 frente a M1: Salinas **77,86 vs 78,90 %** (baja), Trento 87,92 vs 85,09 % (sube), Urban 81,11 vs 80,95 % (igual). Optimizar solo el término espectral no mejoró el agrupamiento de forma consistente; la ganancia aparece con la optimización conjunta (Salinas 80,96 %). **Riesgo para mi tesis:** maximizar la EV por sí sola podría no traducirse en mejor OA; mi experimento debe medirlo (por eso EV, ASA y BR contra OA, y la comparación con SLIC y PCNN-SAM).
- **Cita útil (Sec. I):** «While prior works like Superpixel Sampling Networks (SSN) introduced differentiable superpixel frameworks, their application has been mainly in supervised tasks where gradients are derived from ground-truth labels.» Respalda que la señal sin etiquetas es el punto abierto.
- **Limitación frente a tu propuesta:** 
Los superpíxeles se optimizan para agrupar por subespacios, no para alimentar un clasificador: no hay GCN ni aprendizaje semisupervisado, y no se sabe si sus particiones sirven como nodos de un grafo para clasificar. Además, no evalúa en Indian Pines y sus hiperparámetros (α, M) dependen de información de las etiquetas.
- **Para el protocolo (en qué sección lo citas y la idea en una oración):** 
Trabajos relacionados C y planteamiento del problema: los marcos de superpíxeles diferenciables se han aplicado principalmente a tareas supervisadas, donde el gradiente proviene de las etiquetas, y llevarlos a entornos sin etiquetas exige otra señal de gradiente (Sec. I); Li et al. usan la autorrepresentación, mientras que mi propuesta usa la varianza explicada.

### wang2025multiscale · Prioridad A

Wang et al. (2025). Multiscale Graph Transformer Network With Dynamic Superpixel Pyramid for Hyperspectral Image Classification. *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing*. <http://dx.doi.org/10.1109/JSTARS.2025.3570267>

> Por qué está en la lista: 2025: superpíxeles diferenciables en varias escalas y un transformer de grafos; lo prueba en IP, PU, SA y WHU-Hi-HongHu (tus datos).

> Nota: leí la traducción automática al español (es-ES). Las citas textuales y los números deben tomarse del original en inglés. En el artículo el método se llama **MSGTN** (no MGTN).

- **Pasada 1 (8 de noviembre del 2026):** 
- **Pasada 2 (10 de noviembre de 2026, PDF completo, 19 pp.):** 
- **¿Amenaza tu originalidad? (sí / no / parcial, y por qué):** 
Parcial, la más baja de los tres de clasificación (EMS-GCN > FG-DGCN > MSGTN). MSGTN combina superpíxeles diferenciables tipo SSN con GCN multiescala y un transformer para clasificar HSI, el mismo esqueleto que mi propuesta y con 4 de mis 6 conjuntos. Verificado en el PDF: la **única pérdida es la entropía cruzada** de clasificación (sec. IV-C, final): no tiene pérdida de reconstrucción (ni de etiquetas ni espectral) ni de compacidad. Es el caso más puro de «superpíxeles aprendidos solo con la señal de las etiquetas». La amenaza no sube.
- **Problema que resuelve:** 
Los objetos del terreno en las HSI aparecen a escalas espaciales diversas y los métodos tradicionales extraen características a una escala fija, por lo que no se adaptan a estructuras espaciales variables. Además, los métodos convencionales se centran en rasgos espectrales o espaciales locales y descuidan el contexto global. MSGTN extrae características multiescala (GCN sobre grafos adaptativos) y modela dependencias globales entre superpíxeles (transformer) (resumen e introducción).
- **Cómo forma los superpíxeles (F / A / —, y con qué método):**
A. Segmentación tipo SSN («inspirándonos en la SSN [38]», sec. IV-A) en el espacio de características: una CNN de 3 capas (64, 128 y 64 canales; conv + GroupNorm + ReLU) extrae F, se inicializan los centros con la característica media alrededor de cada píxel, y se calcula la asociación suave Q con los **9 superpíxeles vecinos**, con **10 iteraciones** (Algoritmo 1). Pirámide de K escalas con distinto número de superpíxeles M (Tabla V): UP [512, 256, 128, 64]; IP [1024, 768, 640, 576, 528]; SA [512, 256, 128, 64]; WHHH [768, 512, 384, 320]. Cada escala define un grafo (GCN de 2 capas, ELU, conexiones residuales); los K grafos se concatenan, pasan por un transformer (2–3 codificadores, d' = 64) y Q regresa la clasificación a los píxeles.
- **Señal con la que los aprende (clasificación / sin etiquetas / —, y qué pérdida):** 
Clasificación. **Solo entropía cruzada** sobre los píxeles etiquetados («el rendimiento del modelo se evalúa mediante la función de pérdida de entropía cruzada», sec. IV-C). Sin reconstrucción ni compacidad, a diferencia de SSN, EMS-GCN y FG-DGCN. Entrenamiento extremo a extremo: 200 épocas, lr 1e-3 con recocido coseno y calentamiento lineal, 5 repeticiones.
- **Espectro completo o PCA:** 
Espectro completo: la CNN recibe X ∈ R^{H×W×B} con todas las bandas (103 UP, 220 IP, 224 SA, 270 WHHH); no menciona PCA. La segmentación se hace sobre las características aprendidas de 64 canales, no sobre el espectro original.
- **Distancia (euclidiana / angular / otra):** 
Asignación píxel–superpíxel: tipo SSN entre la característica F_p y el centro S_q [la ecuación es una imagen que no salió en la extracción de texto: verificar en el PDF original si es euclidiana al cuadrado y si incluye coordenadas]. Adyacencia del grafo: núcleo gaussiano sobre la distancia euclidiana entre las características medias de los superpíxeles, σ = 10, grafo denso normalizado (sec. IV-A). No usa distancia angular.
- **Etiquetas por clase:** 
Porcentaje por clase, no un número fijo: 1 % en UP, SA y WHHH; 5 % en IP (Tablas I–IV). En IP eso va de **1 (Avena) y 3 (Alfalfa)** a 123 muestras por clase; en UP de 10 a 187. Sensibilidad con 1–5 % por clase (Fig. 11).
- **Conjuntos de datos:** University of Pavia (UP), Indian Pines (IP), Salinas Valley (SA) y WHU-Hi-HongHu (WHHH): 4 de mis 6.
- **OA en Indian Pines (y con cuántas etiquetas):** 96,94 % con el 5 % por clase (Tabla VII). Otras: UP 99,10 %, SA 99,41–99,43 % y WHHH 97,96 %, todas con el 1 %. Ojo: el resumen dice 99,01 % (UP) y 97,95 % (WHHH); las tablas y la conclusión dicen 99,10 % y 97,96 %. Cita las tablas.
- **Métricas de los superpíxeles (¿reporta ASA, BR o EV, o solo OA, AA y κ?):** 
Solo OA, AA y κ. No mide la calidad de los superpíxeles; solo los muestra visualmente por escala (Fig. 13). Costo (Tabla X): 21,00 GFLOPs, 0,40 M parámetros, 25,25 s de entrenamiento y 0,07 s de inferencia. Hardware: 2× RTX 4090, PyTorch 2.1.0, **precisión mixta con NVIDIA Apex (FusedAdam)**: útil para el grupo 8 (HPC).
- **¿Analiza por tipo de escena? ¿Publica su código?:** 
No hay un análisis por tipo de escena; los resultados se reportan por conjunto. Código: https://github.com/w-tingting/MSGTN
- **Dato clave de la ablación (Tablas XI–XIV):** reemplazar los superpíxeles aprendidos por SLIC («El nuestro (SLIC)») baja la OA **15,04 (UP), 2,84 (IP), 7,27 (SA) y 16,81 (WHHH) puntos**. Contrasta con FG-DGCN, donde la diferencia fue menor de 1 punto. Ojo al citarlo: no dicen cómo configuraron SLIC (número de superpíxeles, sobre qué bandas), y una sola escala de SLIC contra una pirámide aprendida no es una comparación equivalente. Para tu tesis: la ganancia de aprender superpíxeles varía mucho entre artículos y ninguno mide si se debe a la calidad de la segmentación (ASA, BR o EV), justo lo que tú sí mides.
- **Trabajo futuro (sec. VII):** arquitecturas ligeras (atención aproximada, poda) y «la integración de estrategias de aprendizaje autosupervisado o de aprendizaje con pocos ejemplos puede reducir la dependencia de los datos etiquetados». Los autores reconocen la dependencia de las etiquetas, que es tu hueco.
- **Limitación frente a tu propuesta:** 
(1) Los superpíxeles se aprenden solo con la entropía cruzada: su calidad depende por completo de las etiquetas (1–5 % por clase, muchas más que 5) y no hay ningún criterio sin etiquetas que garantice la fidelidad espectral (EV). (2) Sin compacidad ni distancia angular. (3) No evalúa la calidad de los superpíxeles ni la relación entre las métricas de segmentación y la OA. (4) Su ablación contra SLIC no aísla el efecto de la segmentación.
- **Para el protocolo (en qué sección lo citas y la idea en una oración):** 
Trabajos relacionados C y planteamiento. Borrador del párrafo: Wang et al. (2025) proponen MSGTN, que construye una pirámide de superpíxeles diferenciables tipo SSN en el espacio de características, modela cada escala con una GCN y fusiona los superpíxeles de todas las escalas con un transformer. La segmentación se entrena de extremo a extremo solo con la entropía cruzada de clasificación, sin ningún término que mida la calidad de los superpíxeles, y se evalúa con 1 a 5 % de muestras etiquetadas por clase en Pavia University, Indian Pines, Salinas y WHU-Hi-HongHu. Aunque su ablación reporta pérdidas de hasta 16,8 puntos de OA al sustituir la pirámide aprendida por SLIC, el trabajo no mide la calidad de la segmentación (ASA, BR o EV), por lo que no es posible saber si la ganancia proviene de mejores superpíxeles o de la arquitectura multiescala. Los propios autores señalan como trabajo futuro reducir la dependencia de los datos etiquetados, que es el problema que esta tesis aborda al aprender los superpíxeles sin etiquetas mediante la EV.

### li2023feature · Prioridad A

Li et al. (2023). Feature-guided dynamic graph convolutional network for wetland hyperspectral image classification. *International Journal of Applied Earth Observation and Geoinformation*. <http://dx.doi.org/10.1016/j.jag.2023.103485>

> Por qué está en la lista: Módulo aprendible que genera superpíxeles adaptativos más una adyacencia dinámica. Revisa con qué pérdida aprende los superpíxeles.

- **Pasada 1 (9 de noviembre de 2026, solo resumen):** 
- **Pasada 2 (10 de noviembre de 2026, PDF completo, 16 pp.):** 
- **¿Amenaza tu originalidad? (sí / no / parcial, y por qué):** 
Parcial, menor que EMS-GCN. FG-DGCN ya combina un módulo de superpíxeles aprendible (tipo SSN) con una GCN para clasificar HSI con pocas muestras, que es el esqueleto de mi propuesta. Verificado en el PDF: la pérdida total es solo L_cro (Ec. 16, entropía cruzada) + L_rec (Ec. 17, reconstrucción de las **etiquetas semánticas** Y con Q̃Q̂ᵀY). Ambas usan solo los m píxeles etiquetados. No hay ningún término sin etiquetas: ni espectral ni de compacidad. La amenaza no sube. Mi EV usa la misma estructura de reconstrucción, pero con el espectro X en lugar de Y y sobre todos los píxeles, sin etiquetas.
- **Problema que resuelve:** 
Las GCN para clasificar humedales con pocas muestras dependen de superpíxeles generados con procesos predefinidos y parámetros fijados a mano (p. ej. SLIC), y al construir el grafo se ignora la relación espacial entre superpíxeles; ambas cosas limitan la clasificación (resumen e introducción, p. 2).
- **Cómo forma los superpíxeles (F / A / —, y con qué método):** 
A. Dos bloques (sec. 2.2): (1) realce de características por píxel: conv 1×1@64, dos A-Conv 3×3@32 (convolución + autoatención) y conv 3×3@128 con conexiones tipo DenseNet (Tabla 4); (2) generación de superpíxeles tipo SSN: rejilla regular inicial S⁰ con el promedio de características, asociación suave Q_ij = exp(−γ‖X_i − S_j‖) con los **8 superpíxeles vecinos** (γ aprendible, Ec. 9), centros S_j = promedio ponderado por Q (Ec. 10), 5 iteraciones. No inicializa con SLIC. Escala de superpíxeles: 100, 100 y 50 (GF-5, Chris y GT) y 100 en KSC; el artículo no aclara si es el número de superpíxeles o de píxeles por superpíxel [verificar en el código]. Al final, Q proyecta las características de la GCN de vuelta a los píxeles (Ec. 15).
- **Señal con la que los aprende (clasificación / sin etiquetas / —, y qué pérdida):** 
Clasificación. L = L_cro + L_rec (sec. 2.4; el texto no da pesos). L_cro = entropía cruzada sobre los m píxeles etiquetados (Ec. 16). L_rec = entropía cruzada entre las etiquetas one-hot Y y su reconstrucción píxel → superpíxel → píxel Q̃Q̂ᵀY (Ec. 17; Q̃ normalizada por filas, Q̂ por columnas): premia los superpíxeles puros en clase (tipo ASA diferenciable), igual que SSN y EMS-GCN. **Sin término de compacidad**, a diferencia de EMS-GCN. Entrenamiento extremo a extremo con Adam (lr 0,0005), 500 épocas, 2 capas GCN.
- **Espectro completo o PCA:** 
Espectro completo: la red recibe X_in ∈ R^{H×W×C} con todas las bandas (150 en GF-5, 18 en Chris, 126 en GT, 224 en KSC); no menciona PCA. La asignación se calcula sobre las características aprendidas de 128 canales, no sobre el espectro original.
- **Distancia (euclidiana / angular / otra):** 
Euclidiana (norma, sin elevar al cuadrado según la Ec. 9) entre características aprendidas del píxel y del superpíxel. **No concatena coordenadas** (EMS-GCN sí): lo espacial solo entra por la restricción a 8 vecinos. En el grafo usa similitud coseno entre nodos (Ec. 12, adyacencia espectral dinámica) más una adyacencia espacial binaria de superpíxeles con borde común ponderada por una «importancia» aprendida (Ecs. 11 y 13). Ojo: la coseno está en el grafo, no en la asignación píxel–superpíxel.
- **Etiquetas por clase:** 200 por clase en GF-5 y Chris; 5 % por clase en GT (de 6 a 1645 muestras); 20 por clase en KSC (Tablas 1–3, sec. 3.6). Sensibilidad (Tabla 10): 100–300 por clase o 1–10 %. Nunca baja de 100 por clase en los conjuntos principales, muy lejos de mis 5.
- **Conjuntos de datos:** GF-5 (462×617, 150 bandas, 8 clases), Chris (510×511, 18 bandas, 6 clases), GT (dron, 590×512, 126 bandas, 9 clases): los tres son humedales del delta del río Amarillo; más KSC (614×512, 224 bandas, 13 clases) como generalización. Ninguno coincide con mis 6 conjuntos.
- **OA en Indian Pines (y con cuántas etiquetas):** — (no lo evalúa). Referencia: OA 83,28 % GF-5, 88,25 % Chris, 97,85 % GT y 99,73 % KSC (Tablas 6, 7, 8 y 11).
- **Métricas de los superpíxeles (¿reporta ASA, BR o EV, o solo OA, AA y κ?):** 
Solo OA, AA y κ. No mide la calidad de los superpíxeles (ni ASA, ni BR, ni EV). Reporta FLOPs, parámetros y tiempos de entrenamiento y prueba en una RTX 3090 (Tablas 12 y 13; útil para el grupo 8 de HPC).
- **¿Analiza por tipo de escena? ¿Publica su código?:** 
No hay un análisis por tipo de escena; solo comenta casos (el río Amarillo queda en un solo superpíxel, sec. 3.3.1). Código: https://github.com/MengQiao-stu/Elsevier-FG-DGCN-master
- **Dato clave de la ablación (Tabla 9):** cambiar SLIC por los superpíxeles aprendidos mejora la OA solo **+0,38 % (GF-5), +0,80 % (Chris) y +0,47 % (GT)**, y eso con 200 etiquetas por clase. Argumento para mi tesis: aprender los superpíxeles con la señal de las etiquetas aporta poco incluso con muchas etiquetas; con 5 por clase la señal es todavía más débil.
- **Cita útil (introducción, p. 2):** los autores descartan los métodos no supervisados porque «they do not rely on the true labels, which is not robust». Mi tesis argumenta lo contrario para la etapa de segmentación: un criterio sin etiquetas (EV) es más robusto cuando hay solo 5 etiquetas por clase.
- **Limitación frente a tu propuesta:** 
(1) Los superpíxeles se aprenden solo con la señal de las etiquetas (L_cro + L_rec), así que su calidad depende de muchas muestras (≥100 por clase); no hay un criterio sin etiquetas que garantice la fidelidad espectral (EV). (2) No hay compacidad ni coordenadas en la asignación. (3) La distancia de asignación es euclidiana sobre características aprendidas, no angular. (4) No evalúa la calidad de los superpíxeles ni la relación entre las métricas de segmentación y la OA. (5) Solo humedales y ninguno de mis 6 conjuntos.
- **Para el protocolo (en qué sección lo citas y la idea en una oración):** 
Trabajos relacionados C y planteamiento. Borrador del párrafo: Li et al. (2023) proponen FG-DGCN, que sustituye a SLIC por un módulo de superpíxeles aprendible tipo SSN y añade una adyacencia dinámica que combina similitudes espectrales y espaciales entre superpíxeles. Los superpíxeles se ajustan con dos pérdidas supervisadas: la entropía cruzada de clasificación y una pérdida de reconstrucción de las etiquetas semánticas, ambas calculadas solo sobre los píxeles etiquetados. Su ablación muestra que reemplazar SLIC por los superpíxeles aprendidos eleva la OA menos de un punto porcentual, aun con 200 muestras por clase. El trabajo no evalúa la calidad de los superpíxeles ni incluye un criterio sin etiquetas que preserve la información espectral, el hueco que esta tesis aborda con la EV.

### zhang2022ems · Prioridad A

Zhang et al. (2022). EMS-GCN: An End-to-End Mixhop Superpixel-Based Graph Convolutional Network for Hyperspectral Image Classification. *IEEE Transactions on Geoscience and Remote Sensing*. <http://dx.doi.org/10.1109/TGRS.2022.3163326>

> Por qué está en la lista: El más cercano a tu tesis: superpíxeles diferenciables dentro de una GCN, entrenados de extremo a extremo. Define si tu hueco sigue abierto.

- **Pasada 1 (9 de noviembre de 2026, resumen + código oficial):** 
- **Pasada 2 (fecha):** 
- **¿Amenaza tu originalidad? (sí / no / parcial, y por qué):** 
Parcial (la más alta hasta ahora). EMS-GCN ya hace lo que describe el título de mi tesis: superpíxeles diferenciables (SLIC diferenciable tipo SSN) dentro de una GCN, entrenados de extremo a extremo para clasificar HSI. La diferencia que sostiene mi propuesta es la señal: en EMS-GCN los superpíxeles se aprenden con entropía cruzada sobre las etiquetas más una pérdida de reconstrucción de etiquetas y una compacidad solo espacial (posiciones). No hay ningún término sin etiquetas que mida la fidelidad espectral de los superpíxeles; mi propuesta los aprende con la varianza explicada (EV), que es justamente ese término.
- **Problema que resuelve:** 
Las GCN basadas en superpíxeles para HSI usan una segmentación fija hecha como preprocesamiento (p. ej., SLIC), separada de la clasificación, así que los errores de segmentación se arrastran y los bordes no se pueden corregir. EMS-GCN integra una segmentación diferenciable para refinar los bordes durante el entrenamiento y una GCN mixhop que capta información local (dentro del superpíxel) y de largo alcance (entre superpíxeles) (resumen).
- **Cómo forma los superpíxeles (F / A / —, y con qué método):** 
A. SLIC diferenciable (DSLIC, basado en SSN/ESCNet): una CNN multiescala (convoluciones 1×1, 3×3 y 5×5, 128 canales) extrae características por píxel, se les concatenan las coordenadas escaladas (η_pos = 1,8) y se hacen 5 iteraciones de asignación suave (softmax) sobre los 9 superpíxeles vecinos de una rejilla inicial. Número de superpíxeles = H·W/escala, con escala = 100 por defecto (≈ 210 en Indian Pines) (código: EMSGCN.py, ssn.py).
- **Señal con la que los aprende (clasificación / sin etiquetas / —, y qué pérdida):** 
Clasificación. L = CE + 0,1·L_recon + 0,001·L_compact (código: main.py). CE = entropía cruzada de los píxeles etiquetados. L_recon = entropía cruzada entre las etiquetas y su reconstrucción píxel → superpíxel → píxel con la matriz de asignación Q (depende de etiquetas). L_compact = MSE entre las coordenadas de cada píxel y las de su superpíxel (sin etiquetas, pero solo espacial). [Verificar en el PDF que las ecuaciones coinciden con el código.]
- **Espectro completo o PCA:** 
Espectro completo estandarizado (StandardScaler por banda), sin PCA; la CNN lo proyecta a 128 canales antes de segmentar (código).
- **Distancia (euclidiana / angular / otra):** 
Euclidiana al cuadrado entre características CNN + posición del píxel y del superpíxel (kernel calc_assoc). En el grafo: la adyacencia binaria entre superpíxeles vecinos se calcula una sola vez y los pesos de las aristas se aprenden como sigmoid(HHᵀ) con softmax, tipo atención (GCNLayer).
- **Etiquetas por clase:** 30 por clase por defecto (15 si la clase tiene menos de 30), muestreo aleatorio y opción de muestreo disjunto; 5 repeticiones con semillas distintas (código). [Verificar en el PDF otros tamaños de entrenamiento.]
- **Conjuntos de datos:** Indian Pines y Pavia University en el código. [Verificar en el PDF si hay más.]
- **OA en Indian Pines (y con cuántas etiquetas):** [Verificar en el PDF, tabla de resultados; configuración por defecto: 30 por clase.]
- **Código:** https://github.com/immortal13/EMS-GCN-hyperspectral-image-classification (basado en ESCNet y CEGCN; requiere CUDA).
- **Métricas de los superpíxeles (¿reporta ASA, BR o EV, o solo OA, AA y κ?):** 
- **¿Analiza por tipo de escena? ¿Publica su código?:** 
- **Limitación frente a tu propuesta:** 
Los superpíxeles dependen de las etiquetas: con pocas muestras, la señal para mover los bordes es escasa, y la única regularización sin etiquetas es espacial (compacidad), no espectral. Además, la adyacencia del grafo se fija con la segmentación inicial (la rejilla) y no se actualiza cuando cambian los superpíxeles (código: get_A con la variable global A).
- **Para el protocolo (en qué sección lo citas y la idea en una oración):** 
Trabajos relacionados C (antecedente directo) y justificación: Zhang et al. (2022) muestran que integrar superpíxeles diferenciables en una GCN de extremo a extremo mejora la clasificación de HSI frente a la segmentación fija, pero los aprenden con señal supervisada; queda abierto aprenderlos con un criterio espectral sin etiquetas como la EV.

### liu2024dual · Prioridad B

Liu et al. (2024). Dual-Stage Hyperspectral Image Classification Model with Spectral Supertoken. *Computer Vision – ECCV 2024*. <http://dx.doi.org/10.1007/978-3-031-72754-2_21>

> Por qué está en la lista: ECCV 2024: «supertokens» espectrales, una agrupación aprendida parecida a los superpíxeles. Sirve para mostrar la tendencia.

- **Pasada 1 (9 de noviembre de 2026, texto completo arXiv 2407.07307v1):** 
- **Pasada 2 (fecha):** 
- **¿Amenaza tu originalidad? (sí / no / parcial, y por qué):** 
No. DSTC agrupa píxeles espectralmente similares en «supertokens» (inspirados en superpíxeles) y los clasifica con un ViT, no con una GCN. La agrupación es un k-means suave desplegado dentro de la red, entrenada solo con entropía cruzada sobre etiquetas suaves (CPSL); no hay ningún término sin etiquetas ni varianza explicada. Sirve como evidencia de la tendencia a aprender la agrupación dentro de la red, no como competidor directo.
- **Problema que resuelve:** 
Los modelos píxel a píxel ignoran la similitud entre píxeles vecinos, lo que produce bordes borrosos o incorrectos entre clases y etiquetas inconsistentes dentro de regiones continuas con pequeñas variaciones espectrales. Además, SLIC es iterativo, se paraleliza mal en GPU y no garantiza etiquetas consistentes por región (Sec. 1).
- **Cómo forma los superpíxeles (F / A / —, y con qué método):** 
A (supertokens). Un codificador tipo UNet (ResNet18, PVTv2-b1 o Swin-tiny) da características densas F_D; la asociación píxel–centro usa F_D más proyecciones lineales aprendibles del espectro crudo y de su primera derivada espectral (Ec. 5). Centros iniciales tipo SLIC, actualización de k-means suave (Ec. 6) y asignación final por argmax. Se agrupa dentro de parches de 16×16 con M = 4 centros por parche (Sec. 4.1); los supertokens no tienen que ser regiones conexas.
- **Señal con la que los aprende (clasificación / sin etiquetas / —, y qué pérdida):** 
Clasificación. Única pérdida: entropía cruzada de los supertokens contra una etiqueta suave basada en la proporción de clases dentro de cada token (CPSL, Ecs. 11–12). Entrenamiento de extremo a extremo; no hay pérdida de agrupamiento, compacidad ni reconstrucción.
- **Espectro completo o PCA:** 
Espectro completo (sin PCA), más sus derivadas espectrales de primer orden (Ecs. 3–5).
- **Distancia (euclidiana / angular / otra):** 
Euclidiana al cuadrado dentro de un núcleo exponencial: A = exp(−‖F_D + I_a + I'_a − P‖²) (Ec. 5).
- **Etiquetas por clase:** 
No usa un número fijo por clase. En IP, KSC y UP: 10 % para entrenar y 90 % para probar, con parches de 9×9 solapados (Sec. 4.3); no dice si la partición es disjunta, así que puede haber fuga entre entrenamiento y prueba. En WHU-OHS: imágenes completas de 512×512 con anotación densa.
- **Conjuntos de datos:** WHU-OHS (7795 imágenes, 32 bandas, 24 clases), Indian Pines, Kennedy Space Center (KSC) y Pavia University; HS-SOD solo en el material suplementario.
- **OA en Indian Pines (y con cuántas etiquetas):** 98,75 % (AA 96,15 %, κ 98,57 %) con 10 % de muestras (Tabla 3). Ojo: el régimen es muy distinto al de pocas etiquetas, y el 10 % sobre parches solapados infla la comparación.
- **Código:** https://github.com/laprf/DSTC
- **Métricas de los superpíxeles (¿reporta ASA, BR o EV, o solo OA, AA y κ?):** 
- **¿Analiza por tipo de escena? ¿Publica su código?:** 
- **Limitación frente a tu propuesta:** 
La agrupación se aprende solo con la señal de clasificación y requiere muchas etiquetas (10 % o anotación densa); no se evalúa con pocas muestras por clase. Los autores reconocen que la ventaja es menor en IP, KSC y UP, donde trabajan con parches pequeños (Sec. 4.3). No modela relaciones entre regiones con un grafo.
- **Para el protocolo (en qué sección lo citas y la idea en una oración):** 
Trabajos relacionados C (tendencia): Liu et al. (2024) agrupan píxeles en supertokens espectrales aprendidos de extremo a extremo para clasificar HSI, lo que confirma la tendencia a integrar la agrupación tipo superpíxel en la red; pero la aprenden solo con etiquetas y con un transformer, no con un criterio sin etiquetas ni una GCN.

---

## Antecedente directo

*Se cita en el planteamiento, los antecedentes y la metodología.*

### galarza2026 · Prioridad A

Galarza (2026). Tesis de referencia: PCNN-SAM y PCNN-GCN (título completo en tu PDF). *Tesis de maestría (PDF local)*.

> Por qué está en la lista: Antecedente directo: el hallazgo de que la EV predice la OA es la base de tu pregunta. Agrégala a Zotero a mano, con el tipo «Tesis».

- **Pasada 2 (fecha):** 
- **Hallazgos que citas (con página y tabla):** 
- **Trabajo futuro que retomas (TF1, TF3, TF4):** 
- **Números exactos para el planteamiento (EV, OA, correlación):** 
- **Para el protocolo (en qué sección lo citas):** 

---

## 1 · Imágenes hiperespectrales y pocas etiquetas

*Marco teórico (HSI y SAM) y primer párrafo del planteamiento.*

### jia2021survey · Prioridad A

Jia et al. (2021). A survey: Deep learning for hyperspectral image classification with few labeled samples. *Neurocomputing*. <http://dx.doi.org/10.1016/j.neucom.2021.03.035>

> Por qué está en la lista: Revisión del problema de las pocas muestras etiquetadas en HSI: justifica el escenario de 5 etiquetas por clase.

- **Pasada 1 (fecha):** 
- **Pasada 2 (fecha):** 
- **¿Amenaza tu originalidad? (sí / no / parcial, y por qué):** 
- **Problema que resuelve:** 
- **Cómo forma los superpíxeles (F / A / —, y con qué método):** 
- **Señal con la que los aprende (clasificación / sin etiquetas / —, y qué pérdida):** 
- **Espectro completo o PCA:** 
- **Distancia (euclidiana / angular / otra):** 
- **Etiquetas por clase:** 
- **Conjuntos de datos:** 
- **OA en Indian Pines (y con cuántas etiquetas):** 
- **Métricas de los superpíxeles (¿reporta ASA, BR o EV, o solo OA, AA y κ?):** 
- **¿Analiza por tipo de escena? ¿Publica su código?:** 
- **Limitación frente a tu propuesta:** 
- **Para el protocolo (en qué sección lo citas y la idea en una oración):** 

### li2019deep · Prioridad A

Li et al. (2019). Deep Learning for Hyperspectral Image Classification: An Overview. *IEEE Transactions on Geoscience and Remote Sensing*. <http://dx.doi.org/10.1109/TGRS.2019.2907932>

> Por qué está en la lista: Panorama del aprendizaje profundo en HSI; fuente principal del contexto del planteamiento.

- **Pasada 1 (fecha):** 
- **Pasada 2 (fecha):** 
- **¿Amenaza tu originalidad? (sí / no / parcial, y por qué):** 
- **Problema que resuelve:** 
- **Cómo forma los superpíxeles (F / A / —, y con qué método):** 
- **Señal con la que los aprende (clasificación / sin etiquetas / —, y qué pérdida):** 
- **Espectro completo o PCA:** 
- **Distancia (euclidiana / angular / otra):** 
- **Etiquetas por clase:** 
- **Conjuntos de datos:** 
- **OA en Indian Pines (y con cuántas etiquetas):** 
- **Métricas de los superpíxeles (¿reporta ASA, BR o EV, o solo OA, AA y κ?):** 
- **¿Analiza por tipo de escena? ¿Publica su código?:** 
- **Limitación frente a tu propuesta:** 
- **Para el protocolo (en qué sección lo citas y la idea en una oración):** 

### sahu2026hyperspectral · Prioridad B

Sahu y Panda (2026). Hyperspectral image classification: A systematic survey of traditional methods, deep networks, and emerging paradigms. *Signal Processing: Image Communication*. <http://dx.doi.org/10.1016/j.image.2026.117682>

> Por qué está en la lista: Revisión de 2026: la más reciente; útil para citar el estado actual del área.

- **Pasada 1 (fecha):** 
- **Pasada 2 (fecha):** 
- **¿Amenaza tu originalidad? (sí / no / parcial, y por qué):** 
- **Problema que resuelve:** 
- **Cómo forma los superpíxeles (F / A / —, y con qué método):** 
- **Señal con la que los aprende (clasificación / sin etiquetas / —, y qué pérdida):** 
- **Espectro completo o PCA:** 
- **Distancia (euclidiana / angular / otra):** 
- **Etiquetas por clase:** 
- **Conjuntos de datos:** 
- **OA en Indian Pines (y con cuántas etiquetas):** 
- **Métricas de los superpíxeles (¿reporta ASA, BR o EV, o solo OA, AA y κ?):** 
- **¿Analiza por tipo de escena? ¿Publica su código?:** 
- **Limitación frente a tu propuesta:** 
- **Para el protocolo (en qué sección lo citas y la idea en una oración):** 

### ahmad2022hyperspectral · Prioridad B

Ahmad et al. (2022). Hyperspectral Image Classification—Traditional to Deep Models: A Survey for Future Prospects. *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing*. <http://dx.doi.org/10.1109/JSTARS.2021.3133021>

> Por qué está en la lista: Revisión de métodos tradicionales y profundos; ubica a las GCN dentro del área.

- **Pasada 1 (fecha):** 
- **Pasada 2 (fecha):** 
- **¿Amenaza tu originalidad? (sí / no / parcial, y por qué):** 
- **Problema que resuelve:** 
- **Cómo forma los superpíxeles (F / A / —, y con qué método):** 
- **Señal con la que los aprende (clasificación / sin etiquetas / —, y qué pérdida):** 
- **Espectro completo o PCA:** 
- **Distancia (euclidiana / angular / otra):** 
- **Etiquetas por clase:** 
- **Conjuntos de datos:** 
- **OA en Indian Pines (y con cuántas etiquetas):** 
- **Métricas de los superpíxeles (¿reporta ASA, BR o EV, o solo OA, AA y κ?):** 
- **¿Analiza por tipo de escena? ¿Publica su código?:** 
- **Limitación frente a tu propuesta:** 
- **Para el protocolo (en qué sección lo citas y la idea en una oración):** 

### bioucasdias2013hyperspectral · Prioridad B

Bioucas-Dias et al. (2013). Hyperspectral Remote Sensing Data Analysis and Future Challenges. *IEEE Geoscience and Remote Sensing Magazine*. <http://dx.doi.org/10.1109/MGRS.2013.2244672>

> Por qué está en la lista: Referencia clásica de qué es una HSI y sus retos; para la primera oración del planteamiento.

- **Pasada 1 (fecha):** 
- **Pasada 2 (fecha):** 
- **¿Amenaza tu originalidad? (sí / no / parcial, y por qué):** 
- **Problema que resuelve:** 
- **Cómo forma los superpíxeles (F / A / —, y con qué método):** 
- **Señal con la que los aprende (clasificación / sin etiquetas / —, y qué pérdida):** 
- **Espectro completo o PCA:** 
- **Distancia (euclidiana / angular / otra):** 
- **Etiquetas por clase:** 
- **Conjuntos de datos:** 
- **OA en Indian Pines (y con cuántas etiquetas):** 
- **Métricas de los superpíxeles (¿reporta ASA, BR o EV, o solo OA, AA y κ?):** 
- **¿Analiza por tipo de escena? ¿Publica su código?:** 
- **Limitación frente a tu propuesta:** 
- **Para el protocolo (en qué sección lo citas y la idea en una oración):** 

### fauvel2013advances · Prioridad B

Fauvel et al. (2013). Advances in Spectral-Spatial Classification of Hyperspectral Images. *Proceedings of the IEEE*. <http://dx.doi.org/10.1109/JPROC.2012.2197589>

> Por qué está en la lista: Clasificación espectral-espacial: por qué usar vecindad (superpíxeles) y no solo el espectro.

- **Pasada 1 (fecha):** 
- **Pasada 2 (fecha):** 
- **¿Amenaza tu originalidad? (sí / no / parcial, y por qué):** 
- **Problema que resuelve:** 
- **Cómo forma los superpíxeles (F / A / —, y con qué método):** 
- **Señal con la que los aprende (clasificación / sin etiquetas / —, y qué pérdida):** 
- **Espectro completo o PCA:** 
- **Distancia (euclidiana / angular / otra):** 
- **Etiquetas por clase:** 
- **Conjuntos de datos:** 
- **OA en Indian Pines (y con cuántas etiquetas):** 
- **Métricas de los superpíxeles (¿reporta ASA, BR o EV, o solo OA, AA y κ?):** 
- **¿Analiza por tipo de escena? ¿Publica su código?:** 
- **Limitación frente a tu propuesta:** 
- **Para el protocolo (en qué sección lo citas y la idea en una oración):** 

### zhong2020whu · Prioridad C

Zhong et al. (2020). WHU-Hi: UAV-borne hyperspectral with high spatial resolution (H2) benchmark datasets and classifier for precise crop identification based on deep convolutional neural network with CRF. *Remote Sensing of Environment*. <http://dx.doi.org/10.1016/j.rse.2020.112012>

> Por qué está en la lista: Artículo que presenta los conjuntos WHU-Hi (LongKou, HanChuan, HongHu).

- **Pasada 1 (fecha):** 
- **Dato que citarás (con página o tabla):** 
- **Para el protocolo (sección):** 

### paoletti2019deep · Prioridad C

Paoletti et al. (2019). Deep learning classifiers for hyperspectral imaging: A review. *ISPRS Journal of Photogrammetry and Remote Sensing*. <http://dx.doi.org/10.1016/j.isprsjprs.2019.09.006>

> Por qué está en la lista: Revisión de clasificadores profundos en HSI; cita de apoyo.

- **Pasada 1 (fecha):** 
- **Dato que citarás (con página o tabla):** 
- **Para el protocolo (sección):** 

### audebert2019deep · Prioridad C

Audebert et al. (2019). Deep Learning for Classification of Hyperspectral Data: A Comparative Review. *IEEE Geoscience and Remote Sensing Magazine*. <http://dx.doi.org/10.1109/MGRS.2019.2912563>

> Por qué está en la lista: Revisión comparativa con experimentos; cita de apoyo.

- **Pasada 1 (fecha):** 
- **Dato que citarás (con página o tabla):** 
- **Para el protocolo (sección):** 

### khan2018modern · Prioridad C

Khan et al. (2018). Modern Trends in Hyperspectral Image Analysis: A Review. *IEEE Access*. <http://dx.doi.org/10.1109/ACCESS.2018.2812999>

> Por qué está en la lista: Aplicaciones de las HSI (agricultura, minería, medicina…): llena la «(lista de aplicaciones)» del planteamiento.

- **Pasada 1 (fecha):** 
- **Dato que citarás (con página o tabla):** 
- **Para el protocolo (sección):** 

### kruse1993spectral · Prioridad C

Kruse et al. (1993). The spectral image processing system (SIPS)—interactive visualization and analysis of imaging spectrometer data. *Remote Sensing of Environment*. <http://dx.doi.org/10.1016/0034-4257(93)90013-N>

> Por qué está en la lista: Origen del Spectral Angle Mapper (SAM); cita obligada para tu afinidad angular.

- **Pasada 1 (fecha):** 
- **Dato que citarás (con página o tabla):** 
- **Para el protocolo (sección):** 

### congalton1991review · Prioridad C

Congalton (1991). A review of assessing the accuracy of classifications of remotely sensed data. *Remote Sensing of Environment*. <http://dx.doi.org/10.1016/0034-4257(91)90048-B>

> Por qué está en la lista: Métricas de exactitud en percepción remota (OA, κ); cita para la sección de evaluación.

- **Pasada 1 (fecha):** 
- **Dato que citarás (con página o tabla):** 
- **Para el protocolo (sección):** 

### hughes1968mean · Prioridad C

Hughes (1968). On the mean accuracy of statistical pattern recognizers. *IEEE Transactions on Information Theory*. <http://dx.doi.org/10.1109/TIT.1968.1054102>

> Por qué está en la lista: Fenómeno de Hughes: muchas bandas con pocas muestras degradan la clasificación.

- **Pasada 1 (fecha):** 
- **Dato que citarás (con página o tabla):** 
- **Para el protocolo (sección):** 

---

## 2 · Superpíxeles clásicos y su evaluación

*Marco teórico (superpíxeles y métricas) y líneas base.*

### barcelos2024comprehensive · Prioridad A

Barcelos et al. (2024). A Comprehensive Review and New Taxonomy on Superpixel Segmentation. *ACM Computing Surveys*. <http://dx.doi.org/10.1145/3652509>

> Por qué está en la lista: Revisión y taxonomía reciente de superpíxeles, incluidos los métodos con aprendizaje profundo.

- **Pasada 1 (fecha):** 
- **Pasada 2 (fecha):** 
- **¿Amenaza tu originalidad? (sí / no / parcial, y por qué):** 
- **Problema que resuelve:** 
- **Cómo forma los superpíxeles (F / A / —, y con qué método):** 
- **Señal con la que los aprende (clasificación / sin etiquetas / —, y qué pérdida):** 
- **Espectro completo o PCA:** 
- **Distancia (euclidiana / angular / otra):** 
- **Etiquetas por clase:** 
- **Conjuntos de datos:** 
- **OA en Indian Pines (y con cuántas etiquetas):** 
- **Métricas de los superpíxeles (¿reporta ASA, BR o EV, o solo OA, AA y κ?):** 
- **¿Analiza por tipo de escena? ¿Publica su código?:** 
- **Limitación frente a tu propuesta:** 
- **Para el protocolo (en qué sección lo citas y la idea en una oración):** 

### stutz2018superpixels · Prioridad A

Stutz et al. (2018). Superpixels: An evaluation of the state-of-the-art. *Computer Vision and Image Understanding*. <http://dx.doi.org/10.1016/j.cviu.2017.03.007>

> Por qué está en la lista: Comparativa de 28 algoritmos de superpíxeles con métricas estándar (ASA, BR, EV…): fuente de tus métricas.

- **Pasada 1 (fecha):** 
- **Pasada 2 (fecha):** 
- **¿Amenaza tu originalidad? (sí / no / parcial, y por qué):** 
- **Problema que resuelve:** 
- **Cómo forma los superpíxeles (F / A / —, y con qué método):** 
- **Señal con la que los aprende (clasificación / sin etiquetas / —, y qué pérdida):** 
- **Espectro completo o PCA:** 
- **Distancia (euclidiana / angular / otra):** 
- **Etiquetas por clase:** 
- **Conjuntos de datos:** 
- **OA en Indian Pines (y con cuántas etiquetas):** 
- **Métricas de los superpíxeles (¿reporta ASA, BR o EV, o solo OA, AA y κ?):** 
- **¿Analiza por tipo de escena? ¿Publica su código?:** 
- **Limitación frente a tu propuesta:** 
- **Para el protocolo (en qué sección lo citas y la idea en una oración):** 

### achanta2012slic · Prioridad A

Achanta et al. (2012). SLIC Superpixels Compared to State-of-the-Art Superpixel Methods. *IEEE Transactions on Pattern Analysis and Machine Intelligence*. <http://dx.doi.org/10.1109/TPAMI.2012.120>

> Por qué está en la lista: SLIC: tu línea base principal y la base de SSN.

- **Pasada 1 (fecha):** 
- **Pasada 2 (fecha):** 
- **¿Amenaza tu originalidad? (sí / no / parcial, y por qué):** 
- **Problema que resuelve:** 
- **Cómo forma los superpíxeles (F / A / —, y con qué método):** 
- **Señal con la que los aprende (clasificación / sin etiquetas / —, y qué pérdida):** 
- **Espectro completo o PCA:** 
- **Distancia (euclidiana / angular / otra):** 
- **Etiquetas por clase:** 
- **Conjuntos de datos:** 
- **OA en Indian Pines (y con cuántas etiquetas):** 
- **Métricas de los superpíxeles (¿reporta ASA, BR o EV, o solo OA, AA y κ?):** 
- **¿Analiza por tipo de escena? ¿Publica su código?:** 
- **Limitación frente a tu propuesta:** 
- **Para el protocolo (en qué sección lo citas y la idea en una oración):** 

### giraud2026superpixel · Prioridad B

Giraud y Clément (2026). Superpixel segmentation: A long-lasting ill-posed problem. *Computer Vision and Image Understanding*. <http://dx.doi.org/10.1016/j.cviu.2026.104831>

> Por qué está en la lista: 2026: discusión de por qué la segmentación en superpíxeles es un problema mal planteado y cómo evaluarla.

- **Pasada 1 (fecha):** 
- **Pasada 2 (fecha):** 
- **¿Amenaza tu originalidad? (sí / no / parcial, y por qué):** 
- **Problema que resuelve:** 
- **Cómo forma los superpíxeles (F / A / —, y con qué método):** 
- **Señal con la que los aprende (clasificación / sin etiquetas / —, y qué pérdida):** 
- **Espectro completo o PCA:** 
- **Distancia (euclidiana / angular / otra):** 
- **Etiquetas por clase:** 
- **Conjuntos de datos:** 
- **OA en Indian Pines (y con cuántas etiquetas):** 
- **Métricas de los superpíxeles (¿reporta ASA, BR o EV, o solo OA, AA y κ?):** 
- **¿Analiza por tipo de escena? ¿Publica su código?:** 
- **Limitación frente a tu propuesta:** 
- **Para el protocolo (en qué sección lo citas y la idea en una oración):** 

### chen2023local · Prioridad B

Chen et al. (2023). Local aggregation and global attention network for hyperspectral image classification with spectral-induced aligned superpixel segmentation. *Expert Systems with Applications*. <http://dx.doi.org/10.1016/j.eswa.2023.120828>

> Por qué está en la lista: Superpíxeles calculados sobre características espectrales profundas: lo más parecido a tu extractor espectral.

- **Pasada 1 (fecha):** 
- **Pasada 2 (fecha):** 
- **¿Amenaza tu originalidad? (sí / no / parcial, y por qué):** 
- **Problema que resuelve:** 
- **Cómo forma los superpíxeles (F / A / —, y con qué método):** 
- **Señal con la que los aprende (clasificación / sin etiquetas / —, y qué pérdida):** 
- **Espectro completo o PCA:** 
- **Distancia (euclidiana / angular / otra):** 
- **Etiquetas por clase:** 
- **Conjuntos de datos:** 
- **OA en Indian Pines (y con cuántas etiquetas):** 
- **Métricas de los superpíxeles (¿reporta ASA, BR o EV, o solo OA, AA y κ?):** 
- **¿Analiza por tipo de escena? ¿Publica su código?:** 
- **Limitación frente a tu propuesta:** 
- **Para el protocolo (en qué sección lo citas y la idea en una oración):** 

### jiang2018superpca · Prioridad B

Jiang et al. (2018). SuperPCA: A Superpixelwise PCA Approach for Unsupervised Feature Extraction of Hyperspectral Imagery. *IEEE Transactions on Geoscience and Remote Sensing*. <http://dx.doi.org/10.1109/TGRS.2018.2828029>

> Por qué está en la lista: Superpíxeles en HSI sobre la imagen reducida por PCA: ejemplo del flujo que criticas.

- **Pasada 1 (fecha):** 
- **Pasada 2 (fecha):** 
- **¿Amenaza tu originalidad? (sí / no / parcial, y por qué):** 
- **Problema que resuelve:** 
- **Cómo forma los superpíxeles (F / A / —, y con qué método):** 
- **Señal con la que los aprende (clasificación / sin etiquetas / —, y qué pérdida):** 
- **Espectro completo o PCA:** 
- **Distancia (euclidiana / angular / otra):** 
- **Etiquetas por clase:** 
- **Conjuntos de datos:** 
- **OA en Indian Pines (y con cuántas etiquetas):** 
- **Métricas de los superpíxeles (¿reporta ASA, BR o EV, o solo OA, AA y κ?):** 
- **¿Analiza por tipo de escena? ¿Publica su código?:** 
- **Limitación frente a tu propuesta:** 
- **Para el protocolo (en qué sección lo citas y la idea en una oración):** 

### ayres2024hierarchical · Prioridad C

Ayres et al. (2024). Hierarchical homogeneity-based superpixel segmentation: application to hyperspectral image analysis. *International Journal of Remote Sensing*. <http://dx.doi.org/10.1080/01431161.2024.2384098>

> Por qué está en la lista: Superpíxeles jerárquicos por homogeneidad espectral en HSI. Ojo: Taylor & Francis, fuera de tu lista de editoriales.

- **Pasada 1 (fecha):** 
- **Dato que citarás (con página o tabla):** 
- **Para el protocolo (sección):** 

### wang2017superpixel · Prioridad C

Wang et al. (2017). Superpixel segmentation: A benchmark. *Signal Processing: Image Communication*. <http://dx.doi.org/10.1016/j.image.2017.04.007>

> Por qué está en la lista: Otra comparativa de algoritmos de superpíxeles; cita de apoyo.

- **Pasada 1 (fecha):** 
- **Dato que citarás (con página o tabla):** 
- **Para el protocolo (sección):** 

### chen2017linear · Prioridad C

Chen et al. (2017). Linear Spectral Clustering Superpixel. *IEEE Transactions on Image Processing*. <http://dx.doi.org/10.1109/TIP.2017.2651389>

> Por qué está en la lista: LSC, una de tus líneas base.

- **Pasada 1 (fecha):** 
- **Dato que citarás (con página o tabla):** 
- **Para el protocolo (sección):** 

### fang2015classification · Prioridad C

Fang et al. (2015). Classification of Hyperspectral Images by Exploiting Spectral–Spatial Information of Superpixel via Multiple Kernels. *IEEE Transactions on Geoscience and Remote Sensing*. <http://dx.doi.org/10.1109/TGRS.2015.2445767>

> Por qué está en la lista: Superpíxeles más kernels múltiples en HSI: trabajo pionero de superpíxeles en HSI.

- **Pasada 1 (fecha):** 
- **Dato que citarás (con página o tabla):** 
- **Para el protocolo (sección):** 

### vandenbergh2014seeds · Prioridad C

Van den Bergh et al. (2014). SEEDS: Superpixels Extracted Via Energy-Driven Sampling. *International Journal of Computer Vision*. <http://dx.doi.org/10.1007/s11263-014-0744-2>

> Por qué está en la lista: SEEDS, una de tus líneas base.

- **Pasada 1 (fecha):** 
- **Dato que citarás (con página o tabla):** 
- **Para el protocolo (sección):** 

### liu2011entropy · Prioridad C

Liu et al. (2011). Entropy rate superpixel segmentation. *CVPR 2011*. <http://dx.doi.org/10.1109/CVPR.2011.5995323>

> Por qué está en la lista: ERS, una de tus líneas base.

- **Pasada 1 (fecha):** 
- **Dato que citarás (con página o tabla):** 
- **Para el protocolo (sección):** 

### johnson1999pcnn · Prioridad C

Johnson y Padgett (1999). PCNN models and applications. *IEEE Transactions on Neural Networks*. <http://dx.doi.org/10.1109/72.761706>

> Por qué está en la lista: Modelo PCNN, la base de PCNN-SAM de Galarza.

- **Pasada 1 (fecha):** 
- **Dato que citarás (con página o tabla):** 
- **Para el protocolo (sección):** 

---

## 3 · Redes neuronales de grafos

*Marco teórico (GCN).*

### kipf2017semisupervised · Prioridad A

Kipf y Welling (2017). Semi-Supervised Classification with Graph Convolutional Networks. *International Conference on Learning Representations (ICLR)*. <https://arxiv.org/abs/1609.02907>

> Por qué está en la lista: La GCN que usan Galarza y tu tesis. Ojo: ICLR (arXiv), fuera de tu lista; confirma con la Dra. que se acepta.

- **Pasada 1 (fecha):** 
- **Pasada 2 (fecha):** 
- **¿Amenaza tu originalidad? (sí / no / parcial, y por qué):** 
- **Problema que resuelve:** 
- **Cómo forma los superpíxeles (F / A / —, y con qué método):** 
- **Señal con la que los aprende (clasificación / sin etiquetas / —, y qué pérdida):** 
- **Espectro completo o PCA:** 
- **Distancia (euclidiana / angular / otra):** 
- **Etiquetas por clase:** 
- **Conjuntos de datos:** 
- **OA en Indian Pines (y con cuántas etiquetas):** 
- **Métricas de los superpíxeles (¿reporta ASA, BR o EV, o solo OA, AA y κ?):** 
- **¿Analiza por tipo de escena? ¿Publica su código?:** 
- **Limitación frente a tu propuesta:** 
- **Para el protocolo (en qué sección lo citas y la idea en una oración):** 

### wu2021comprehensive · Prioridad B

Wu et al. (2021). A Comprehensive Survey on Graph Neural Networks. *IEEE Transactions on Neural Networks and Learning Systems*. <http://dx.doi.org/10.1109/TNNLS.2020.2978386>

> Por qué está en la lista: Revisión de redes neuronales de grafos: marco teórico.

- **Pasada 1 (fecha):** 
- **Pasada 2 (fecha):** 
- **¿Amenaza tu originalidad? (sí / no / parcial, y por qué):** 
- **Problema que resuelve:** 
- **Cómo forma los superpíxeles (F / A / —, y con qué método):** 
- **Señal con la que los aprende (clasificación / sin etiquetas / —, y qué pérdida):** 
- **Espectro completo o PCA:** 
- **Distancia (euclidiana / angular / otra):** 
- **Etiquetas por clase:** 
- **Conjuntos de datos:** 
- **OA en Indian Pines (y con cuántas etiquetas):** 
- **Métricas de los superpíxeles (¿reporta ASA, BR o EV, o solo OA, AA y κ?):** 
- **¿Analiza por tipo de escena? ¿Publica su código?:** 
- **Limitación frente a tu propuesta:** 
- **Para el protocolo (en qué sección lo citas y la idea en una oración):** 

### bronstein2017geometric · Prioridad C

Bronstein et al. (2017). Geometric Deep Learning: Going beyond Euclidean data. *IEEE Signal Processing Magazine*. <http://dx.doi.org/10.1109/MSP.2017.2693418>

> Por qué está en la lista: Aprendizaje profundo geométrico: contexto de las GNN.

- **Pasada 1 (fecha):** 
- **Dato que citarás (con página o tabla):** 
- **Para el protocolo (sección):** 

---

## 4 · GCN con superpíxeles en HSI

*«Trabajos relacionados A» y la mayor parte del cuadro comparativo.*

### zhao2025review · Prioridad A

Zhao et al. (2025). A review of hyperspectral image classification based on graph neural networks. *Artificial Intelligence Review*. <http://dx.doi.org/10.1007/s10462-025-11169-y>

> Por qué está en la lista: Revisión de 2025 de GNN en HSI: comprueba con sus referencias que no te falte nada.

- **Pasada 1 (fecha):** 
- **Pasada 2 (fecha):** 
- **¿Amenaza tu originalidad? (sí / no / parcial, y por qué):** 
- **Problema que resuelve:** 
- **Cómo forma los superpíxeles (F / A / —, y con qué método):** 
- **Señal con la que los aprende (clasificación / sin etiquetas / —, y qué pérdida):** 
- **Espectro completo o PCA:** 
- **Distancia (euclidiana / angular / otra):** 
- **Etiquetas por clase:** 
- **Conjuntos de datos:** 
- **OA en Indian Pines (y con cuántas etiquetas):** 
- **Métricas de los superpíxeles (¿reporta ASA, BR o EV, o solo OA, AA y κ?):** 
- **¿Analiza por tipo de escena? ¿Publica su código?:** 
- **Limitación frente a tu propuesta:** 
- **Para el protocolo (en qué sección lo citas y la idea en una oración):** 

### hong2021graph · Prioridad A

Hong et al. (2021). Graph Convolutional Networks for Hyperspectral Image Classification. *IEEE Transactions on Geoscience and Remote Sensing*. <http://dx.doi.org/10.1109/TGRS.2020.3015157>

> Por qué está en la lista: miniGCN: GCN contra CNN en HSI y entrenamiento por mini-lotes; muy citado.

- **Pasada 1 (fecha):** 
- **Pasada 2 (fecha):** 
- **¿Amenaza tu originalidad? (sí / no / parcial, y por qué):** 
- **Problema que resuelve:** 
- **Cómo forma los superpíxeles (F / A / —, y con qué método):** 
- **Señal con la que los aprende (clasificación / sin etiquetas / —, y qué pérdida):** 
- **Espectro completo o PCA:** 
- **Distancia (euclidiana / angular / otra):** 
- **Etiquetas por clase:** 
- **Conjuntos de datos:** 
- **OA en Indian Pines (y con cuántas etiquetas):** 
- **Métricas de los superpíxeles (¿reporta ASA, BR o EV, o solo OA, AA y κ?):** 
- **¿Analiza por tipo de escena? ¿Publica su código?:** 
- **Limitación frente a tu propuesta:** 
- **Para el protocolo (en qué sección lo citas y la idea en una oración):** 

### liu2021cnn · Prioridad A

Liu et al. (2021). CNN-Enhanced Graph Convolutional Network With Pixel- and Superpixel-Level Feature Fusion for Hyperspectral Image Classification. *IEEE Transactions on Geoscience and Remote Sensing*. <http://dx.doi.org/10.1109/TGRS.2020.3037361>

> Por qué está en la lista: CEGCN: matriz de asociación píxel-superpíxel (la pieza que tu SSN vuelve suave). Tiene código público.

- **Pasada 1 (fecha):** 
- **Pasada 2 (fecha):** 
- **¿Amenaza tu originalidad? (sí / no / parcial, y por qué):** 
- **Problema que resuelve:** 
- **Cómo forma los superpíxeles (F / A / —, y con qué método):** 
- **Señal con la que los aprende (clasificación / sin etiquetas / —, y qué pérdida):** 
- **Espectro completo o PCA:** 
- **Distancia (euclidiana / angular / otra):** 
- **Etiquetas por clase:** 
- **Conjuntos de datos:** 
- **OA en Indian Pines (y con cuántas etiquetas):** 
- **Métricas de los superpíxeles (¿reporta ASA, BR o EV, o solo OA, AA y κ?):** 
- **¿Analiza por tipo de escena? ¿Publica su código?:** 
- **Limitación frente a tu propuesta:** 
- **Para el protocolo (en qué sección lo citas y la idea en una oración):** 

### wan2020multiscale · Prioridad A

Wan et al. (2020). Multiscale Dynamic Graph Convolutional Network for Hyperspectral Image Classification. *IEEE Transactions on Geoscience and Remote Sensing*. <http://dx.doi.org/10.1109/TGRS.2019.2949180>

> Por qué está en la lista: MDGCN: GCN sobre superpíxeles SLIC con grafos dinámicos y multiescala; referencia clásica del área.

- **Pasada 1 (fecha):** 
- **Pasada 2 (fecha):** 
- **¿Amenaza tu originalidad? (sí / no / parcial, y por qué):** 
- **Problema que resuelve:** 
- **Cómo forma los superpíxeles (F / A / —, y con qué método):** 
- **Señal con la que los aprende (clasificación / sin etiquetas / —, y qué pérdida):** 
- **Espectro completo o PCA:** 
- **Distancia (euclidiana / angular / otra):** 
- **Etiquetas por clase:** 
- **Conjuntos de datos:** 
- **OA en Indian Pines (y con cuántas etiquetas):** 
- **Métricas de los superpíxeles (¿reporta ASA, BR o EV, o solo OA, AA y κ?):** 
- **¿Analiza por tipo de escena? ¿Publica su código?:** 
- **Limitación frente a tu propuesta:** 
- **Para el protocolo (en qué sección lo citas y la idea en una oración):** 

### jia2024graph · Prioridad B

Jia et al. (2024). Graph-in-Graph Convolutional Network for Hyperspectral Image Classification. *IEEE Transactions on Neural Networks and Learning Systems*. <http://dx.doi.org/10.1109/TNNLS.2022.3182715>

> Por qué está en la lista: GiGCN: cada superpíxel es a su vez un grafo interno.

- **Pasada 1 (fecha):** 
- **Pasada 2 (fecha):** 
- **¿Amenaza tu originalidad? (sí / no / parcial, y por qué):** 
- **Problema que resuelve:** 
- **Cómo forma los superpíxeles (F / A / —, y con qué método):** 
- **Señal con la que los aprende (clasificación / sin etiquetas / —, y qué pérdida):** 
- **Espectro completo o PCA:** 
- **Distancia (euclidiana / angular / otra):** 
- **Etiquetas por clase:** 
- **Conjuntos de datos:** 
- **OA en Indian Pines (y con cuántas etiquetas):** 
- **Métricas de los superpíxeles (¿reporta ASA, BR o EV, o solo OA, AA y κ?):** 
- **¿Analiza por tipo de escena? ¿Publica su código?:** 
- **Limitación frente a tu propuesta:** 
- **Para el protocolo (en qué sección lo citas y la idea en una oración):** 

### yang2024semi · Prioridad B

Yang et al. (2024). Semi-Supervised Multiscale Dynamic Graph Convolution Network for Hyperspectral Image Classification. *IEEE Transactions on Neural Networks and Learning Systems*. <http://dx.doi.org/10.1109/TNNLS.2022.3212985>

> Por qué está en la lista: GCN dinámica multiescala semi-supervisada (TNNLS 2024).

- **Pasada 1 (fecha):** 
- **Pasada 2 (fecha):** 
- **¿Amenaza tu originalidad? (sí / no / parcial, y por qué):** 
- **Problema que resuelve:** 
- **Cómo forma los superpíxeles (F / A / —, y con qué método):** 
- **Señal con la que los aprende (clasificación / sin etiquetas / —, y qué pérdida):** 
- **Espectro completo o PCA:** 
- **Distancia (euclidiana / angular / otra):** 
- **Etiquetas por clase:** 
- **Conjuntos de datos:** 
- **OA en Indian Pines (y con cuántas etiquetas):** 
- **Métricas de los superpíxeles (¿reporta ASA, BR o EV, o solo OA, AA y κ?):** 
- **¿Analiza por tipo de escena? ¿Publica su código?:** 
- **Limitación frente a tu propuesta:** 
- **Para el protocolo (en qué sección lo citas y la idea en una oración):** 

### yu2024dspg · Prioridad B

Yu et al. (2024). dSPG: A New Discriminant Superpixel Graph Regularizer and Convolutional Network for Hyperspectral Image Classification. *IEEE Transactions on Geoscience and Remote Sensing*. <http://dx.doi.org/10.1109/TGRS.2024.3439434>

> Por qué está en la lista: Regularizador de grafo de superpíxeles discriminante (TGRS 2024).

- **Pasada 1 (fecha):** 
- **Pasada 2 (fecha):** 
- **¿Amenaza tu originalidad? (sí / no / parcial, y por qué):** 
- **Problema que resuelve:** 
- **Cómo forma los superpíxeles (F / A / —, y con qué método):** 
- **Señal con la que los aprende (clasificación / sin etiquetas / —, y qué pérdida):** 
- **Espectro completo o PCA:** 
- **Distancia (euclidiana / angular / otra):** 
- **Etiquetas por clase:** 
- **Conjuntos de datos:** 
- **OA en Indian Pines (y con cuántas etiquetas):** 
- **Métricas de los superpíxeles (¿reporta ASA, BR o EV, o solo OA, AA y κ?):** 
- **¿Analiza por tipo de escena? ¿Publica su código?:** 
- **Limitación frente a tu propuesta:** 
- **Para el protocolo (en qué sección lo citas y la idea en una oración):** 

### zou2024psformer · Prioridad B

Zou et al. (2024). PSFormer: Pyramid Superpixel Transformer for Hyperspectral Image Classification. *IEEE Transactions on Geoscience and Remote Sensing*. <http://dx.doi.org/10.1109/TGRS.2024.3468876>

> Por qué está en la lista: Transformer con pirámide de superpíxeles (TGRS 2024).

- **Pasada 1 (fecha):** 
- **Pasada 2 (fecha):** 
- **¿Amenaza tu originalidad? (sí / no / parcial, y por qué):** 
- **Problema que resuelve:** 
- **Cómo forma los superpíxeles (F / A / —, y con qué método):** 
- **Señal con la que los aprende (clasificación / sin etiquetas / —, y qué pérdida):** 
- **Espectro completo o PCA:** 
- **Distancia (euclidiana / angular / otra):** 
- **Etiquetas por clase:** 
- **Conjuntos de datos:** 
- **OA en Indian Pines (y con cuántas etiquetas):** 
- **Métricas de los superpíxeles (¿reporta ASA, BR o EV, o solo OA, AA y κ?):** 
- **¿Analiza por tipo de escena? ¿Publica su código?:** 
- **Limitación frente a tu propuesta:** 
- **Para el protocolo (en qué sección lo citas y la idea en una oración):** 

### li2023multistage · Prioridad B

Li et al. (2023). Multistage Superpixel-Guided Hyperspectral Image Classification With Sparse Graph Attention Networks. *IEEE Transactions on Geoscience and Remote Sensing*. <http://dx.doi.org/10.1109/TGRS.2023.3304716>

> Por qué está en la lista: Superpíxeles en varias etapas con atención de grafos dispersa (TGRS 2023).

- **Pasada 1 (fecha):** 
- **Pasada 2 (fecha):** 
- **¿Amenaza tu originalidad? (sí / no / parcial, y por qué):** 
- **Problema que resuelve:** 
- **Cómo forma los superpíxeles (F / A / —, y con qué método):** 
- **Señal con la que los aprende (clasificación / sin etiquetas / —, y qué pérdida):** 
- **Espectro completo o PCA:** 
- **Distancia (euclidiana / angular / otra):** 
- **Etiquetas por clase:** 
- **Conjuntos de datos:** 
- **OA en Indian Pines (y con cuántas etiquetas):** 
- **Métricas de los superpíxeles (¿reporta ASA, BR o EV, o solo OA, AA y κ?):** 
- **¿Analiza por tipo de escena? ¿Publica su código?:** 
- **Limitación frente a tu propuesta:** 
- **Para el protocolo (en qué sección lo citas y la idea en una oración):** 

### liu2022multilevel · Prioridad B

Liu et al. (2022). Multilevel Superpixel Structured Graph U-Nets for Hyperspectral Image Classification. *IEEE Transactions on Geoscience and Remote Sensing*. <http://dx.doi.org/10.1109/TGRS.2021.3112586>

> Por qué está en la lista: Superpíxeles en varios niveles con Graph U-Net; mismo autor que CEGCN.

- **Pasada 1 (fecha):** 
- **Pasada 2 (fecha):** 
- **¿Amenaza tu originalidad? (sí / no / parcial, y por qué):** 
- **Problema que resuelve:** 
- **Cómo forma los superpíxeles (F / A / —, y con qué método):** 
- **Señal con la que los aprende (clasificación / sin etiquetas / —, y qué pérdida):** 
- **Espectro completo o PCA:** 
- **Distancia (euclidiana / angular / otra):** 
- **Etiquetas por clase:** 
- **Conjuntos de datos:** 
- **OA en Indian Pines (y con cuántas etiquetas):** 
- **Métricas de los superpíxeles (¿reporta ASA, BR o EV, o solo OA, AA y κ?):** 
- **¿Analiza por tipo de escena? ¿Publica su código?:** 
- **Limitación frente a tu propuesta:** 
- **Para el protocolo (en qué sección lo citas y la idea en una oración):** 

### sellars2020superpixel · Prioridad B

Sellars et al. (2020). Superpixel Contracted Graph-Based Learning for Hyperspectral Image Classification. *IEEE Transactions on Geoscience and Remote Sensing*. <http://dx.doi.org/10.1109/TGRS.2019.2961599>

> Por qué está en la lista: Grafo «contraído» de superpíxeles con aprendizaje semi-supervisado: muy cercano al flujo de Galarza.

- **Pasada 1 (fecha):** 
- **Pasada 2 (fecha):** 
- **¿Amenaza tu originalidad? (sí / no / parcial, y por qué):** 
- **Problema que resuelve:** 
- **Cómo forma los superpíxeles (F / A / —, y con qué método):** 
- **Señal con la que los aprende (clasificación / sin etiquetas / —, y qué pérdida):** 
- **Espectro completo o PCA:** 
- **Distancia (euclidiana / angular / otra):** 
- **Etiquetas por clase:** 
- **Conjuntos de datos:** 
- **OA en Indian Pines (y con cuántas etiquetas):** 
- **Métricas de los superpíxeles (¿reporta ASA, BR o EV, o solo OA, AA y κ?):** 
- **¿Analiza por tipo de escena? ¿Publica su código?:** 
- **Limitación frente a tu propuesta:** 
- **Para el protocolo (en qué sección lo citas y la idea en una oración):** 

### qin2019spectral · Prioridad B

Qin et al. (2019). Spectral–Spatial Graph Convolutional Networks for Semisupervised Hyperspectral Image Classification. *IEEE Geoscience and Remote Sensing Letters*. <http://dx.doi.org/10.1109/LGRS.2018.2869563>

> Por qué está en la lista: Una de las primeras GCN semi-supervisadas espectral-espaciales para HSI.

- **Pasada 1 (fecha):** 
- **Pasada 2 (fecha):** 
- **¿Amenaza tu originalidad? (sí / no / parcial, y por qué):** 
- **Problema que resuelve:** 
- **Cómo forma los superpíxeles (F / A / —, y con qué método):** 
- **Señal con la que los aprende (clasificación / sin etiquetas / —, y qué pérdida):** 
- **Espectro completo o PCA:** 
- **Distancia (euclidiana / angular / otra):** 
- **Etiquetas por clase:** 
- **Conjuntos de datos:** 
- **OA en Indian Pines (y con cuántas etiquetas):** 
- **Métricas de los superpíxeles (¿reporta ASA, BR o EV, o solo OA, AA y κ?):** 
- **¿Analiza por tipo de escena? ¿Publica su código?:** 
- **Limitación frente a tu propuesta:** 
- **Para el protocolo (en qué sección lo citas y la idea en una oración):** 

### wang2026mixed · Prioridad C

Wang et al. (2026). Mixed KAN Convolutional Network With Superpixel-Based Multihead Graph Learning for Hyperspectral Image Classification. *IEEE Transactions on Geoscience and Remote Sensing*. <http://dx.doi.org/10.1109/TGRS.2026.3705787>

> Por qué está en la lista: 2026: grafos multicabeza sobre superpíxeles; muestra que el tema sigue activo.

- **Pasada 1 (fecha):** 
- **Dato que citarás (con página o tabla):** 
- **Para el protocolo (sección):** 

### yang2024dual · Prioridad C

Yang y Zhang (2024). A Dual-Branch Fusion of a Graph Convolutional Network and a Convolutional Neural Network for Hyperspectral Image Classification. *Sensors*. <http://dx.doi.org/10.3390/s24144760>

> Por qué está en la lista: Fusión de GCN y CNN en dos ramas (MDPI Sensors 2024).

- **Pasada 1 (fecha):** 
- **Dato que citarás (con página o tabla):** 
- **Para el protocolo (sección):** 

### zhao2023hyperspectral · Prioridad C

Zhao et al. (2023). Hyperspectral Image Classification With Multi-Attention Transformer and Adaptive Superpixel Segmentation-Based Active Learning. *IEEE Transactions on Image Processing*. <http://dx.doi.org/10.1109/TIP.2023.3287738>

> Por qué está en la lista: Segmentación adaptativa en superpíxeles con aprendizaje activo (TIP 2023).

- **Pasada 1 (fecha):** 
- **Dato que citarás (con página o tabla):** 
- **Para el protocolo (sección):** 

### wang2023pixel · Prioridad C

Wang et al. (2023). Pixel-Superpixel Level Multiscale Graph and Spectral–Spatial Representation Fusion Network for Hyperspectral Image Classification. *IEEE Geoscience and Remote Sensing Letters*. <http://dx.doi.org/10.1109/LGRS.2023.3323706>

> Por qué está en la lista: Grafos a nivel píxel y superpíxel combinados (GRSL 2023).

- **Pasada 1 (fecha):** 
- **Dato que citarás (con página o tabla):** 
- **Para el protocolo (sección):** 

### liu2023adaptive · Prioridad C

Liu et al. (2023). Adaptive Multi-Feature Fusion Graph Convolutional Network for Hyperspectral Image Classification. *Remote Sensing*. <http://dx.doi.org/10.3390/rs15235483>

> Por qué está en la lista: GCN con fusión adaptativa de características (MDPI Remote Sensing 2023).

- **Pasada 1 (fecha):** 
- **Dato que citarás (con página o tabla):** 
- **Para el protocolo (sección):** 

### ding2022graph · Prioridad C

Ding et al. (2022). Graph Sample and Aggregate-Attention Network for Hyperspectral Image Classification. *IEEE Geoscience and Remote Sensing Letters*. <http://dx.doi.org/10.1109/LGRS.2021.3062944>

> Por qué está en la lista: GraphSAGE con atención para HSI.

- **Pasada 1 (fecha):** 
- **Dato que citarás (con página o tabla):** 
- **Para el protocolo (sección):** 

### ding2022multi · Prioridad C

Ding et al. (2022). Multi-feature fusion: Graph neural network and CNN combining for hyperspectral image classification. *Neurocomputing*. <http://dx.doi.org/10.1016/j.neucom.2022.06.031>

> Por qué está en la lista: Fusión de GNN y CNN para HSI.

- **Pasada 1 (fecha):** 
- **Dato que citarás (con página o tabla):** 
- **Para el protocolo (sección):** 

### wan2021hyperspectral · Prioridad C

Wan et al. (2021). Hyperspectral Image Classification With Context-Aware Dynamic Graph Convolutional Network. *IEEE Transactions on Geoscience and Remote Sensing*. <http://dx.doi.org/10.1109/TGRS.2020.2994205>

> Por qué está en la lista: GCN dinámica sensible al contexto; mismo grupo que MDGCN.

- **Pasada 1 (fecha):** 
- **Dato que citarás (con página o tabla):** 
- **Para el protocolo (sección):** 

---

## 5 · Superpíxeles diferenciables y sin etiquetas

*Marco teórico (SSN) y «Trabajos relacionados B».*

### yang2020superpixel · Prioridad A

Yang et al. (2020). Superpixel Segmentation With Fully Convolutional Networks. *2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*. <http://dx.doi.org/10.1109/CVPR42600.2020.01398>

> Por qué está en la lista: Superpixel FCN: una sola red sin iteraciones; principal alternativa a SSN.

- **Pasada 1 (fecha):** 
- **Pasada 2 (fecha):** 
- **¿Amenaza tu originalidad? (sí / no / parcial, y por qué):** 
- **Problema que resuelve:** 
- **Cómo forma los superpíxeles (F / A / —, y con qué método):** 
- **Señal con la que los aprende (clasificación / sin etiquetas / —, y qué pérdida):** 
- **Espectro completo o PCA:** 
- **Distancia (euclidiana / angular / otra):** 
- **Etiquetas por clase:** 
- **Conjuntos de datos:** 
- **OA en Indian Pines (y con cuántas etiquetas):** 
- **Métricas de los superpíxeles (¿reporta ASA, BR o EV, o solo OA, AA y κ?):** 
- **¿Analiza por tipo de escena? ¿Publica su código?:** 
- **Limitación frente a tu propuesta:** 
- **Para el protocolo (en qué sección lo citas y la idea en una oración):** 

### suzuki2020superpixel · Prioridad A

Suzuki (2020). Superpixel Segmentation Via Convolutional Neural Networks with Regularized Information Maximization. *ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*. <http://dx.doi.org/10.1109/ICASSP40776.2020.9054140>

> Por qué está en la lista: Superpíxeles con CNN sin etiquetas: antecedente directo de tu entrenamiento no supervisado.

- **Pasada 1 (fecha):** 
- **Pasada 2 (fecha):** 
- **¿Amenaza tu originalidad? (sí / no / parcial, y por qué):** 
- **Problema que resuelve:** 
- **Cómo forma los superpíxeles (F / A / —, y con qué método):** 
- **Señal con la que los aprende (clasificación / sin etiquetas / —, y qué pérdida):** 
- **Espectro completo o PCA:** 
- **Distancia (euclidiana / angular / otra):** 
- **Etiquetas por clase:** 
- **Conjuntos de datos:** 
- **OA en Indian Pines (y con cuántas etiquetas):** 
- **Métricas de los superpíxeles (¿reporta ASA, BR o EV, o solo OA, AA y κ?):** 
- **¿Analiza por tipo de escena? ¿Publica su código?:** 
- **Limitación frente a tu propuesta:** 
- **Para el protocolo (en qué sección lo citas y la idea en una oración):** 

### jampani2018superpixel · Prioridad A

Jampani et al. (2018). Superpixel Sampling Networks. *Computer Vision – ECCV 2018*. <http://dx.doi.org/10.1007/978-3-030-01234-2_22>

> Por qué está en la lista: SSN: tu punto de partida. Se lee tres veces: hoy las pasadas 1 y 2 y en diciembre la tercera.

- **Pasada 1 (fecha):** 
- **Pasada 2 (fecha):** 
- **¿Amenaza tu originalidad? (sí / no / parcial, y por qué):** 
- **Problema que resuelve:** 
- **Cómo forma los superpíxeles (F / A / —, y con qué método):** 
- **Señal con la que los aprende (clasificación / sin etiquetas / —, y qué pérdida):** 
- **Espectro completo o PCA:** 
- **Distancia (euclidiana / angular / otra):** 
- **Etiquetas por clase:** 
- **Conjuntos de datos:** 
- **OA en Indian Pines (y con cuántas etiquetas):** 
- **Métricas de los superpíxeles (¿reporta ASA, BR o EV, o solo OA, AA y κ?):** 
- **¿Analiza por tipo de escena? ¿Publica su código?:** 
- **Limitación frente a tu propuesta:** 
- **Para el protocolo (en qué sección lo citas y la idea en una oración):** 

### eliasof2022rethinking · Prioridad B

Eliasof et al. (2022). Rethinking Unsupervised Neural Superpixel Segmentation. *2022 IEEE International Conference on Image Processing (ICIP)*. <http://dx.doi.org/10.1109/ICIP46576.2022.9897484>

> Por qué está en la lista: Continúa a Suzuki; compara la imagen con su versión «superpixelada» suave, una idea muy parecida a tu pérdida EV.

- **Pasada 1 (fecha):** 
- **Pasada 2 (fecha):** 
- **¿Amenaza tu originalidad? (sí / no / parcial, y por qué):** 
- **Problema que resuelve:** 
- **Cómo forma los superpíxeles (F / A / —, y con qué método):** 
- **Señal con la que los aprende (clasificación / sin etiquetas / —, y qué pérdida):** 
- **Espectro completo o PCA:** 
- **Distancia (euclidiana / angular / otra):** 
- **Etiquetas por clase:** 
- **Conjuntos de datos:** 
- **OA en Indian Pines (y con cuántas etiquetas):** 
- **Métricas de los superpíxeles (¿reporta ASA, BR o EV, o solo OA, AA y κ?):** 
- **¿Analiza por tipo de escena? ¿Publica su código?:** 
- **Limitación frente a tu propuesta:** 
- **Para el protocolo (en qué sección lo citas y la idea en una oración):** 

### zhu2021learning · Prioridad B

Zhu et al. (2021). Learning the Superpixel in a Non-iterative and Lifelong Manner. *2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*. <http://dx.doi.org/10.1109/CVPR46437.2021.00128>

> Por qué está en la lista: LNSNet: superpíxeles sin etiquetas y sin iteraciones.

- **Pasada 1 (fecha):** 
- **Pasada 2 (fecha):** 
- **¿Amenaza tu originalidad? (sí / no / parcial, y por qué):** 
- **Problema que resuelve:** 
- **Cómo forma los superpíxeles (F / A / —, y con qué método):** 
- **Señal con la que los aprende (clasificación / sin etiquetas / —, y qué pérdida):** 
- **Espectro completo o PCA:** 
- **Distancia (euclidiana / angular / otra):** 
- **Etiquetas por clase:** 
- **Conjuntos de datos:** 
- **OA en Indian Pines (y con cuántas etiquetas):** 
- **Métricas de los superpíxeles (¿reporta ASA, BR o EV, o solo OA, AA y κ?):** 
- **¿Analiza por tipo de escena? ¿Publica su código?:** 
- **Limitación frente a tu propuesta:** 
- **Para el protocolo (en qué sección lo citas y la idea en una oración):** 

### wang2021ainet · Prioridad C

Wang et al. (2021). AINet: Association Implantation for Superpixel Segmentation. *2021 IEEE/CVF International Conference on Computer Vision (ICCV)*. <http://dx.doi.org/10.1109/ICCV48922.2021.00699>

> Por qué está en la lista: AINet: mejora de la asociación píxel-superpíxel.

- **Pasada 1 (fecha):** 
- **Dato que citarás (con página o tabla):** 
- **Para el protocolo (sección):** 

### tu2018learning · Prioridad C

Tu et al. (2018). Learning Superpixels with Segmentation-Aware Affinity Loss. *2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition*. <http://dx.doi.org/10.1109/CVPR.2018.00066>

> Por qué está en la lista: Afinidades aprendidas para superpíxeles (SEAL).

- **Pasada 1 (fecha):** 
- **Dato que citarás (con página o tabla):** 
- **Para el protocolo (sección):** 

---

## 7 · Método experimental

*Metodología: comparación estadística entre métodos.*

### demsar2006statistical · Prioridad C

Demšar (2006). Statistical Comparisons of Classifiers over Multiple Data Sets. *Journal of Machine Learning Research*. <https://jmlr.org/papers/v7/demsar06a.html>

> Por qué está en la lista: Pruebas estadísticas (Wilcoxon, Friedman) para comparar métodos. Ojo: JMLR, fuera de tu lista.

- **Pasada 1 (fecha):** 
- **Dato que citarás (con página o tabla):** 
- **Para el protocolo (sección):** 

---

## 8 · HPC y costo computacional

*Objetivo 8 (TF1 de Galarza): «Trabajos relacionados D», la metodología de tiempos y la infraestructura.*

### quesadabarriuso2021gpu · Prioridad A

Quesada-Barriuso et al. (2021). GPU accelerated waterpixel algorithm for superpixel segmentation of hyperspectral images. *The Journal of Supercomputing*. <https://doi.org/10.1007/s11227-021-03666-y>

> Por qué está en la lista: El más cercano a tu objetivo 8: superpíxeles para HSI implementados en GPU. Anota qué paraleliza y cuánto acelera.

- **Pasada 1 (fecha):** 
- **Pasada 2 (fecha):** 
- **Problema que resuelve:** 
- **Componente que acelera (distancia espectral, segmentación, entrenamiento…):** 
- **Plataforma (GPU, multinúcleo, clúster, nube) y modelo de hardware:** 
- **Aceleración reportada (× frente a qué):** 
- **Cómo midió los tiempos (repeticiones, qué reporta):** 
- **Relación con tu objetivo 8:** 
- **Para el protocolo (en qué sección lo citas y la idea en una oración):** 

### hoefler2015scientific · Prioridad A

Hoefler y Belli (2015). Scientific benchmarking of parallel computing systems: twelve ways to tell the masses when reporting performance results. *Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis*. <https://doi.org/10.1145/2807591.2807644>

> Por qué está en la lista: Cómo medir tiempos y aceleración correctamente: repeticiones, mediana e intervalos. Va a la metodología del objetivo 8.

- **Pasada 1 (fecha):** 
- **Pasada 2 (fecha):** 
- **Problema que resuelve:** 
- **Componente que acelera (distancia espectral, segmentación, entrenamiento…):** 
- **Plataforma (GPU, multinúcleo, clúster, nube) y modelo de hardware:** 
- **Aceleración reportada (× frente a qué):** 
- **Cómo midió los tiempos (repeticiones, qué reporta):** 
- **Relación con tu objetivo 8:** 
- **Para el protocolo (en qué sección lo citas y la idea en una oración):** 

### plaza2011high · Prioridad A

Plaza et al. (2011). High Performance Computing for Hyperspectral Remote Sensing. *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing*. <https://doi.org/10.1109/jstars.2010.2095495>

> Por qué está en la lista: Fundamento: por qué el análisis hiperespectral necesita HPC (GPU, clústeres). Cita principal de «Trabajos relacionados D».

- **Pasada 1 (fecha):** 
- **Pasada 2 (fecha):** 
- **Problema que resuelve:** 
- **Componente que acelera (distancia espectral, segmentación, entrenamiento…):** 
- **Plataforma (GPU, multinúcleo, clúster, nube) y modelo de hardware:** 
- **Aceleración reportada (× frente a qué):** 
- **Cómo midió los tiempos (repeticiones, qué reporta):** 
- **Relación con tu objetivo 8:** 
- **Para el protocolo (en qué sección lo citas y la idea en una oración):** 

### wang2023cloud · Prioridad B

Wang et al. (2023). Cloud-Based Fine-Grained Parallel Optimization on CPU-GPU Heterogeneous Hyperspectral Image Superpixel Space-Spectrum Fusion Classification Algorithms. *IGARSS 2023 - 2023 IEEE International Geoscience and Remote Sensing Symposium*. <https://doi.org/10.1109/igarss52108.2023.10281686>

> Por qué está en la lista: Superpíxeles y clasificación de HSI paralelizados en CPU+GPU: antecedente directo del objetivo 8.

- **Pasada 1 (fecha):** 
- **Pasada 2 (fecha):** 
- **Problema que resuelve:** 
- **Componente que acelera (distancia espectral, segmentación, entrenamiento…):** 
- **Plataforma (GPU, multinúcleo, clúster, nube) y modelo de hardware:** 
- **Aceleración reportada (× frente a qué):** 
- **Cómo midió los tiempos (repeticiones, qué reporta):** 
- **Relación con tu objetivo 8:** 
- **Para el protocolo (en qué sección lo citas y la idea en una oración):** 

### haut2021distributed · Prioridad B

Haut et al. (2021). Distributed Deep Learning for Remote Sensing Data Interpretation. *Proceedings of the IEEE*. <https://doi.org/10.1109/jproc.2021.3063258>

> Por qué está en la lista: Aprendizaje profundo distribuido en percepción remota (grupo de Plaza).

- **Pasada 1 (fecha):** 
- **Pasada 2 (fecha):** 
- **Problema que resuelve:** 
- **Componente que acelera (distancia espectral, segmentación, entrenamiento…):** 
- **Plataforma (GPU, multinúcleo, clúster, nube) y modelo de hardware:** 
- **Aceleración reportada (× frente a qué):** 
- **Cómo midió los tiempos (repeticiones, qué reporta):** 
- **Relación con tu objetivo 8:** 
- **Para el protocolo (en qué sección lo citas y la idea en una oración):** 

### lu2021mtfc · Prioridad B

Lu et al. (2021). MTFC: A Multi-GPU Training Framework for Cube-CNN-Based Hyperspectral Image Classification. *IEEE Transactions on Emerging Topics in Computing*. <https://doi.org/10.1109/tetc.2020.3016978>

> Por qué está en la lista: Entrenamiento de clasificadores HSI con varias GPU.

- **Pasada 1 (fecha):** 
- **Pasada 2 (fecha):** 
- **Problema que resuelve:** 
- **Componente que acelera (distancia espectral, segmentación, entrenamiento…):** 
- **Plataforma (GPU, multinúcleo, clúster, nube) y modelo de hardware:** 
- **Aceleración reportada (× frente a qué):** 
- **Cómo midió los tiempos (repeticiones, qué reporta):** 
- **Relación con tu objetivo 8:** 
- **Para el protocolo (en qué sección lo citas y la idea en una oración):** 

### bennun2019demystifying · Prioridad B

Ben-Nun y Hoefler (2019). Demystifying Parallel and Distributed Deep Learning: An In-depth Concurrency Analysis. *ACM Computing Surveys*. <https://doi.org/10.1145/3320060>

> Por qué está en la lista: Teoría: paralelismo de datos y de modelo en aprendizaje profundo. Para el marco teórico del objetivo 8.

- **Pasada 1 (fecha):** 
- **Pasada 2 (fecha):** 
- **Problema que resuelve:** 
- **Componente que acelera (distancia espectral, segmentación, entrenamiento…):** 
- **Plataforma (GPU, multinúcleo, clúster, nube) y modelo de hardware:** 
- **Aceleración reportada (× frente a qué):** 
- **Cómo midió los tiempos (repeticiones, qué reporta):** 
- **Relación con tu objetivo 8:** 
- **Para el protocolo (en qué sección lo citas y la idea en una oración):** 

### lee2011recent · Prioridad B

Lee et al. (2011). Recent Developments in High Performance Computing for Remote Sensing: A Review. *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing*. <https://doi.org/10.1109/jstars.2011.2162643>

> Por qué está en la lista: Revisión de HPC en percepción remota: una cita para los antecedentes.

- **Pasada 1 (fecha):** 
- **Pasada 2 (fecha):** 
- **Problema que resuelve:** 
- **Componente que acelera (distancia espectral, segmentación, entrenamiento…):** 
- **Plataforma (GPU, multinúcleo, clúster, nube) y modelo de hardware:** 
- **Aceleración reportada (× frente a qué):** 
- **Cómo midió los tiempos (repeticiones, qué reporta):** 
- **Relación con tu objetivo 8:** 
- **Para el protocolo (en qué sección lo citas y la idea en una oración):** 

### li2020pytorch · Prioridad C

Li et al. (2020). PyTorch distributed: experiences on accelerating data parallel training. *Proceedings of the VLDB Endowment*. <https://doi.org/10.14778/3415478.3415530>

> Por qué está en la lista: PyTorch Distributed: la herramienta si usas varias GPU. Publicado en PVLDB (ACM).

- **Pasada 1 (fecha):** 
- **Dato que citarás (con página o tabla):** 
- **Para el protocolo (sección):** 

### chiang2019cluster · Prioridad C

Chiang et al. (2019). Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks. *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery &amp; Data Mining*. <https://doi.org/10.1145/3292500.3330925>

> Por qué está en la lista: Escalado de GCN a grafos grandes; sirve para justificar que tu grafo de superpíxeles no lo necesita.

- **Pasada 1 (fecha):** 
- **Dato que citarás (con página o tabla):** 
- **Para el protocolo (sección):** 

### wu2016parallel · Prioridad C

Wu et al. (2016). Parallel and Distributed Dimensionality Reduction of Hyperspectral Data on Cloud Computing Architectures. *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing*. <https://doi.org/10.1109/jstars.2016.2542193>

> Por qué está en la lista: Reducción de dimensión de HSI en la nube (cómputo distribuido).

- **Pasada 1 (fecha):** 
- **Dato que citarás (con página o tabla):** 
- **Para el protocolo (sección):** 

### bernabe2013hyperspectral · Prioridad C

Bernabé et al. (2013). Hyperspectral Unmixing on GPUs and Multi-Core Processors: A Comparison. *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing*. <https://doi.org/10.1109/jstars.2013.2254470>

> Por qué está en la lista: Ejemplo clásico de comparación GPU contra CPU multinúcleo en HSI.

- **Pasada 1 (fecha):** 
- **Dato que citarás (con página o tabla):** 
- **Para el protocolo (sección):** 
