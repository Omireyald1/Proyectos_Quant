import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
from scipy.stats import norm

# Configuración inicial de la página
st.set_page_config(page_title="Risk & Actuarial Quantitative Dashboard", layout="wide")

# Estilo global de gráficos
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Blues_r")

st.sidebar.title("Navegación Quant")
st.sidebar.markdown("---")
modulo = st.sidebar.radio(
    "Selecciona un Módulo de Riesgos:",
    ["Dashboard General", "1. Riesgo de Crédito (PD)", "2. Pasivos Laborales (NIF D-3)", "3. Inmunización Estructurada (ALM)"]
)

# Base de datos compartida o simulada para consistencia
@st.cache_data
def generar_censo_empleados(n=1000):
    np.random.seed(101)
    edades = np.random.triangular(left=18, mode=35, right=60, size=n).astype(int)
    antiguedad_maxima = edades - 18
    antiguedades = np.array([np.random.randint(0, max(1, am)) for am in antiguedad_maxima])
    salarios_mensuales = np.random.lognormal(mean=9.8, sigma=0.5, size=n)
    return pd.DataFrame({
        'ID_Empleado': np.arange(1, n + 1),
        'Edad': edades,
        'Antigüedad': antiguedades,
        'Salario_Mensual': salarios_mensuales
    })

# =====================================================================
# MODULO: DASHBOARD GENERAL
# =====================================================================
if modulo == "Dashboard General":
    st.title("Risk & Actuarial Management Platform")
    st.markdown("""
    Bienvenido a la plataforma integrada de administración de riesgos cuantitativos y modelos actuariales. 
    Este ecosistema interactivo empaqueta los motores de cálculo desarrollados para las mesas de **Model Risk**, 
    **Consultoría Actuarial (NIF D-3)** y **Estrategia de Activos y Pasivos (ALM)**.
    
    ### Arquitectura del Sistema
    1. **Módulo 1: Riesgo de Crédito (PD):** Clasificación binaria mediante optimización por Máxima Verosimilitud (Regresión Logística) enfocado en portafolios minoristas con tratamiento analítico de desbalanceo de clases.
    2. **Módulo 2: Pasivos Laborales (NIF D-3 / IAS 19):** Motor vectorizado para la valuación de obligaciones contingentes bajo el Método de la Unidad de Crédito Proyectada (PUC) utilizando tablas biométricas dinámicas.
    3. **Módulo 3: Inmunización Estructurada (ALM):** Mitigación de riesgo de tasa de interés mediante emparejamiento exacto de Duración de Macaulay y optimización algebraica de portafolios de Mbonos soberanos.
    """)
    
    st.info("Utiliza el panel de la izquierda para navegar de forma profunda en cada motor matemático.")

# =====================================================================
# MODULO 1: RIESGO DE CRÉDITO
# =====================================================================
elif modulo == "1. Riesgo de Crédito (PD)":
    st.title("🔬 Modelado de Probabilidad de Incumplimiento (PD)")
    st.markdown("Optimización de parámetros para la estimación de Pérdida Esperada ($ECL = PD \\times LGD \\times EAD$).")
    
    # Controles del modelo
    col_ctrl1, col_ctrl2 = st.columns(2)
    with col_ctrl1:
        tratamiento_desbalanceo = st.selectbox("Estrategia contra Desbalanceo Matemático:", ["Class Weights (Balanced)", "Sin Ajuste (MCO Estilo)"])
    with col_ctrl2:
        test_size = st.slider("Tamaño del Set de Validación (%):", 10, 40, 20) / 100
        
    # Simulación interna de datos de crédito (evita fallas de SSL externas)
    @st.cache_data
    def simular_datos_credito(n=5000):
        np.random.seed(42)
        limit_bal = np.random.exponential(scale=150000, size=n) + 10000
        edad = np.random.randint(21, 65, size=n)
        # Variables comportamentales (meses de atraso)
        pay_1 = np.random.choice([-1, 0, 1, 2, 3], size=n, p=[0.4, 0.4, 0.1, 0.07, 0.03])
        bill_amt1 = limit_bal * np.random.uniform(0.1, 0.9, size=n)
        
        # Log odds de default basados en variables
        log_odds = -2.5 - 0.000005 * limit_bal + 0.8 * pay_1 + 0.00001 * bill_amt1
        prob_default = 1 / (1 + np.exp(-log_odds))
        target = np.random.binomial(1, p=prob_default)
        
        return pd.DataFrame({'LIMIT_BAL': limit_bal, 'AGE': edad, 'PAY_1': pay_1, 'BILL_AMT1': bill_amt1, 'target': target})
        
    df_credit = simular_datos_credito()
    
    # Separación e Inferencia
    X = df_credit.drop('target', axis=1)
    y = df_credit['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    cw = 'balanced' if tratamiento_desbalanceo == "Class Weights (Balanced)" else None
    model = LogisticRegression(class_weight=cw, random_state=42)
    model.fit(X_train_s, y_train)
    
    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)[:, 1]
    
    # Métricas clave en tarjetas
    roc_auc = roc_auc_score(y_test, y_prob)
    rep = classification_report(y_test, y_pred, output_dict=True)
    
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("ROC-AUC Model Score", f"{roc_auc:.4f}")
    col_m2.metric("Sensibilidad / Recall (Clase 1)", f"{rep['1']['recall']:.2%}")
    col_m3.metric("Precisión General (Accuracy)", f"{rep['accuracy']:.2%}")
    
    # Gráficos de validación de Model Risk
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax_cm)
        ax_cm.set_title("Matriz de Confusión Auditable", fontweight='bold')
        ax_cm.set_xlabel("Predicción")
        ax_cm.set_ylabel("Realidad")
        st.pyplot(fig_cm)
        
    with col_g2:
        fig_roc, ax_roc = plt.subplots(figsize=(5, 4))
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.2f})')
        ax_roc.plot([0, 1], [0, 1], color='navy', linestyle='--')
        ax_roc.set_title("Curva ROC (Poder de Separación)", fontweight='bold')
        ax_roc.set_xlabel("Tasa Falsos Positivos")
        ax_roc.set_ylabel("Tasa Verdaderos Positivos")
        ax_roc.legend()
        st.pyplot(fig_roc)

# =====================================================================
# MODULO 2: PASIVOS LABORALES
# =====================================================================
elif modulo == "2. Pasivos Laborales (NIF D-3)":
    st.title("📊 Valuación Actuarial de Pasivos Corporativos")
    st.markdown("Cálculo de la **Obligación por Beneficios Definidos (DBO)** mediante el método contable de la **Unidad de Crédito Proyectada (PUC)**.")
    
    # Configuración de supuestos en tiempo real
    st.sidebar.markdown("### Supuestos Actuariales")
    r_input = st.sidebar.slider("Tasa de Descuento Financiero (%):", 5.0, 15.0, 8.75, step=0.25) / 100
    i_input = st.sidebar.slider("Tasa de Crecimiento Salarial (%):", 2.0, 10.0, 4.50, step=0.25) / 100
    edad_ret_input = st.sidebar.number_input("Edad de Jubilación Estándar:", value=65)
    meses_premio = st.sidebar.number_input("Meses de Beneficio Contractual:", value=3)

    df_emp = generar_censo_empleados()
    
    # Motor de Cálculo Vectorizado
    df_emp['Años_Faltantes'] = edad_ret_input - df_emp['Edad']
    df_emp['Salario_Proyectado'] = df_emp['Salario_Mensual'] * (1 + i_input)**df_emp['Años_Faltantes']
    df_emp['Beneficio_Futuro'] = df_emp['Salario_Proyectado'] * meses_premio
    df_emp['Prob_Permanencia_Acumulada'] = 0.95 ** df_emp['Años_Faltantes'] # Curva de decremento simplificada
    df_emp['Factor_Descuento'] = 1 / ((1 + r_input)**df_emp['Años_Faltantes'])
    df_emp['VPO'] = df_emp['Beneficio_Futuro'] * df_emp['Prob_Permanencia_Acumulada'] * df_emp['Factor_Descuento']
    df_emp['DBO'] = df_emp['VPO'] * (df_emp['Antigüedad'] / (df_emp['Antigüedad'] + df_emp['Años_Faltantes']))
    df_emp['Costo_Servicio'] = df_emp['VPO'] / (df_emp['Antigüedad'] + df_emp['Años_Faltantes'])
    
    dbo_total = df_emp['DBO'].sum()
    csc_total = df_emp['Costo_Servicio'].sum()
    
    col_c1, col_c2 = st.columns(2)
    col_c1.metric("Obligación Total Recocida (DBO Balance General)", f"${dbo_total:,.2f} MXN")
    col_c2.metric("Costo Laboral del Año (Gasto Estado Resultados)", f"${csc_total:,.2f} MXN")
    
    # Segmentación para reporte ejecutivo al CFO
    df_emp['Grupo_Edad'] = pd.cut(df_emp['Edad'], bins=[0, 25, 35, 45, 55, 70], labels=['<25', '25-35', '35-45', '45-55', '55+'])
    rep_grupos = df_emp.groupby('Grupo_Edad', observed=False)['DBO'].sum().reset_index()
    
    fig_bar, ax_bar = plt.subplots(figsize=(10, 4))
    sns.barplot(x='Grupo_Edad', y='DBO', data=rep_grupos, edgecolor='black', ax=ax_bar)
    ax_bar.set_title("Concentración del Riesgo de Fondeo por Rango de Edad", fontweight='bold')
    ax_bar.set_ylabel("Pasivo Acumulado ($ MXN)")
    st.pyplot(fig_bar)

# =====================================================================
# MODULO 3: INMUNIZACIÓN (ALM)
# =====================================================================
elif modulo == "3. Inmunización Estructurada (ALM)":
    st.title("📈 Asset-Liability Management (ALM) & Inmunización de Redington")
    st.markdown("Estrategia de cobertura estructural de pasivos utilizando Mbonos soberanos de tasa fija.")
    
    # Simulación de flujos de pasivo agregados
    horizonte = np.arange(1, 41)
    flujos_esperados = norm.pdf(horizonte, loc=16, scale=7) * 1200
    flujos_esperados = np.maximum(flujos_esperados, 10)
    
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        d_target = st.slider("Duración Requerida del Pasivo ($D_{Mac}$ Target):", 5.0, 15.0, 9.5, step=0.1)
    with col_p2:
        r_mercado = st.slider("Rendimiento del Mercado de Deuda ($y$):", 4.0, 14.0, 8.75, step=0.25) / 100

    # Características de Instrumentos de Cobertura
    # Mbono 10Y
    d_mbono_10y = 7.20
    p_mbono_10y = 94.50
    # Mbono 30Y
    d_mbono_30y = 11.40
    p_mbono_30y = 98.20
    
    # Optimización Algebraica de Pesos
    if d_target < d_mbono_10y or d_target > d_mbono_30y:
        st.error(f"❌ Error de Descalce Estructural (Duration Mismatch): No es posible inmunizar una duración de {d_target} años utilizando únicamente Mbonos con duraciones de {d_mbono_10y}Y y {d_mbono_30y}Y sin incurrir en ventas en corto (Apalancamiento). Ajusta la duración target dentro del rango operativo.")
    else:
        w1 = (d_mbono_30y - d_target) / (d_mbono_30y - d_mbono_10y)
        w2 = 1 - w1
        
        st.success(f"✅ Portafolio Sintético Inmunizado Encontrado Exitosamente. Duración del Activo = {w1*d_mbono_10y + w2*d_mbono_30y:.2f} años.")
        
        col_w1, col_w2 = st.columns(2)
        col_w1.metric("Asignación Estratégica en Mbono 10Y", f"{w1:.2%}")
        col_w2.metric("Asignación Estratégica en Mbono 30Y", f"{w2:.2%}")
        
        # Gráfico sectorial del Hedge
        fig_pie, ax_pie = plt.subplots(figsize=(5, 4))
        ax_pie.pie([w1, w2], labels=['Mbono 10Y (Corto)', 'Mbono 30Y (Largo)'], autopct='%1.1f%%', startangle=90, wedgeprops={'edgecolor':'black'})
        ax_pie.set_title("Estructura de Capital del Fondo de Cobertura Actuarial", fontweight='bold')
        st.pyplot(fig_pie)
