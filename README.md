# Plataforma Cuantitativa de Riesgos y Finanzas Actuariales

Un ecosistema de análisis de riesgos cuantitativos y valuación actuarial de grado empresarial, implementado en Python (Streamlit) y Microsoft Excel. Esta plataforma cierra la brecha entre el modelado predictivo de crédito, la contabilidad regulatoria de pasivos laborales y las estrategias avanzadas de gestión de activos y pasivos (ALM).

Desarrollado bajo estándares institucionales (NIF D-3 / IAS 19) para servir como un portafolio técnico robusto en arquitectura de riesgos, validación de modelos y consultoría cuantitativa.

---

## 🛠️ Arquitectura de la Plataforma y Módulos Principales

### 1. Riesgo de Crédito y Probabilidad de Incumplimiento (PD)
* **Metodología:** Marco de clasificación binaria mediante Regresión Logística optimizada por Estimación de Máxima Verosimilitud (MLE).
* **Características Clave:** * Manejo avanzado del desbalanceo de clases matemáticamente penalizado (`class_weight='balanced'`).
  * Integración completa de pipelines de preprocesamiento de datos y escalado de variables (`StandardScaler`).
* **Riesgo de Modelo (Model Risk):** Métricas de evaluación diseñadas para comités de riesgos y auditoría interna, presentando **Curvas ROC-AUC** para evaluar el poder de discriminación y **Matrices de Confusión** para el control de errores Tipo I y Tipo II.

### 2. Valuación Actuarial de Pasivos Corporativos (NIF D-3 / IAS 19)
* **Metodología:** Motor de evaluación completamente vectorizado que implementa el **Método de la Unidad de Crédito Proyectada (PUC)** para valuar obligaciones contingentes a largo plazo.
* **Características Clave:**
  * **Modelos de Decrementos Múltiples:** Los riesgos se modelan bajo la teoría de riesgos competitivos, combinando tablas demográficas oficiales (mortalidad) con curvas empíricas de rotación corporativa ($q_x^{(w)}$).
  * **Valores de Conmutación ($D_x, N_x$):** Implementa atajos actuariales clásicos para colapsar proyecciones de flujo de efectivo de alta dimensionalidad en divisiones algebraicas discretas y auditables. Ideal para valuar rentas vitalicias diferidas en poblaciones de más de 10,000 empleados.
  * **Revelación Contable:** Divide el valor presente total de las obligaciones en la Obligación por Beneficios Definidos (DBO) para el balance general y el Costo Laboral del Año (CSC) para el estado de resultados.

### 3. Gestión de Activos y Pasivos (ALM) e Inmunización Estructural
* **Metodología:** Marco clásico de **Inmunización de Redington** diseñado para mitigar el riesgo de tasa de interés en los balances corporativos.
* **Características Clave:**
  * **Arquitectura de Sensibilidad:** Cálculo vectorizado de la **Duración de Macaulay**, **Duración Modificada** y **Convexidad** tanto para activos como para pasivos.
  * **Optimización de Portafolios:** Solucionador de ecuaciones lineales que calcula dinámicamente los pesos óptimos ($w_1, w_2$) de instrumentos de deuda soberana (ej. Mbonos a 10 y 30 años) para empatar la duración objetivo del pasivo.
  * **Control de Descalce:** Restricciones programadas para alertar sobre descalces de plazos y evitar posiciones estructuralmente imposibles de cubrir sin apalancamiento.

### 4. Dashboard Interactivo en Streamlit (Producción)
* **Arquitectura:** Aplicación web modular (`app.py`) construida para proveer a directivos (CFOs) y auditores externos de un entorno de pruebas interactivo.
* **Capacidades:** Análisis de sensibilidad y estrés en tiempo real sobre supuestos macroeconómicos (tasas de descuento, inflación salarial, edad de retiro) con actualización instantánea de los pasivos actuariales y los pesos del portafolio de cobertura.

---

## 📂 Estructura del Repositorio

```text
├── app.py                      # Aplicación Web Interactiva (Streamlit)
├── notebooks/                  # Investigación, prototipos y pruebas matemáticas
│   ├── credit_risk_pd.ipynb    # Modelado de PD y optimización MLE
│   └── actuarial_models.ipynb  # Matrices de decremento y algoritmos ALM
├── excel/                      # Papeles de trabajo actuariales auditables
│   └── Actuarial_Valuation_PUC_10k.xlsx  # Modelo corporativo (10,000 empleados)
└── README.md                   # Documentación del proyecto y marco teórico
```

---

## 🚀 Requisitos Técnicos e Instalación

Para ejecutar el dashboard interactivo de manera local, asegúrate de tener Python 3.9+ instalado y sigue estos pasos:

1. Clona el repositorio:
   ```bash
   git clone https://github.com/TU_USUARIO/TU_REPOSITORIO.git
   cd TU_REPOSITORIO
   ```

2. Instala las dependencias:
   ```bash
   pip install streamlit pandas numpy matplotlib seaborn scikit-learn scipy
   ```

3. Ejecuta la aplicación en Streamlit:
   ```bash
   streamlit run app.py
   ```

---

## 📊 Resumen Teórico y Ecuaciones Incorporadas

* **Fórmula de Salario Proyectado:**
  $$S_T = S_0 	imes (1 + i)^{T-t}$$

* **Valor Presente Actuarial (Valores de Conmutación):**
  $$VPO = \text{Pensión Anual} 	imes rac{N_{\text{Retiro}}}{D_{\text{Edad Actual}}}$$

* **Duración de Macaulay ($D_{\text{Mac}}$):**
  $$D_{\text{Mac}} = rac{\sum t 	imes PV(CF_t)}{\text{Total } PV}$$

* **Optimización de Pesos (Redington):**
  $$w_1 = rac{D_2 - D_L}{D_2 - D_1}, \quad w_2 = 1 - w_1$$
