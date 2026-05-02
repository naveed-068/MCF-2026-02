"""
Aquí va a estar la pagina de Streamlit (st) donde se mostrará el proyecto ya en 
formato de pagina web.
"""
#%%

"""
Aplicación Streamlit para Análisis de Riesgo del Paladio (PA=F)
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime
from scipy import stats
from scipy.stats import norm, t, kurtosis, skew
#import warnings
#warnings.filterwarnings('ignore')

#Función auxiliar para la extracción de las series
def extract_value(data):
    """
    Extrae un valor numérico de Series/DataFrame/array
    """
    if data is None:
        return 0
    if hasattr(data, 'values'):
        data = data.values
    if hasattr(data, 'shape'):
        if data.shape == ():
            return float(data)
        elif len(data.shape) == 1:
            return float(data[0]) if len(data) > 0 else 0
        elif len(data.shape) >= 2:
            return float(data[0, 0]) if data.shape[0] > 0 and data.shape[1] > 0 else 0
    if hasattr(data, '__float__'):
        return float(data)
    if hasattr(data, 'iloc'):
        return extract_value(data.iloc[0])
    return 0


# FUNCIONES DE CÁLCULO (Todo lo del proyecto)
def descargar_datos(ticker, start_date, end_date):
    """Descarga datos de Yahoo Finance"""
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

def calcular_rendimientos(data):
    """Calcula rendimientos diarios"""
    if isinstance(data['Close'], pd.DataFrame):
        retornos = data['Close'].iloc[:, 0].pct_change().dropna() * 100
    else:
        retornos = data['Close'].pct_change().dropna() * 100
    return retornos

def Expected_Shortfall_normal(alpha, mu, sigma):
    """ES paramétrico normal"""
    z_alpha = norm.ppf(alpha)
    return mu - sigma * (norm.pdf(z_alpha) / alpha)

def Expected_Shortfall_tstudent(alpha, df, loc, scale):
    """ES paramétrico t-student"""
    t_alpha = t.ppf(alpha, df)
    pdf_t_alpha = t.pdf(t_alpha, df)
    ES_std = -pdf_t_alpha / alpha * ((df + t_alpha**2) / (df - 1))
    return loc + scale * ES_std

def rolling_VaR_historico(retornos, window=252):
    """Rolling VaR histórico"""
    resultados = []
    for i in range(window, len(retornos)):
        ventana = retornos.iloc[i-window:i]
        var_95 = np.percentile(ventana, 5)
        var_99 = np.percentile(ventana, 1)
        es_95 = ventana[ventana <= var_95].mean()
        es_99 = ventana[ventana <= var_99].mean()
        
        resultados.append({
            'Fecha': retornos.index[i],
            'Retorno_Real': retornos.iloc[i],
            'VaR_95%': var_95,
            'VaR_99%': var_99,
            'ES_95%': es_95,
            'ES_99%': es_99
        })
    return pd.DataFrame(resultados)

def VaR_volatilidad_movil(retornos, window=252):
    """VaR con volatilidad móvil"""
    q_95 = norm.ppf(0.95)
    q_99 = norm.ppf(0.99)
    volatilidad = retornos.rolling(window=window).std()
    VaR_95 = -q_95 * volatilidad.shift(1)
    VaR_99 = -q_99 * volatilidad.shift(1)
    
    resultados = pd.DataFrame({
        'Fecha': retornos.index,
        'Retorno_Real': retornos,
        'VaR_95%': VaR_95,
        'VaR_99%': VaR_99
    })
    return resultados.dropna()

def monte_carlo_var_es(rendimientos, n_simulaciones=10000):
    """Monte Carlo VaR y ES"""
    mu = extract_value(rendimientos.mean())
    sigma = extract_value(rendimientos.std())
    df, loc, scale = t.fit(rendimientos)
    
    sim_norm = np.random.normal(mu, sigma, n_simulaciones)
    sim_t = t.rvs(df, loc, scale, n_simulaciones)
    
    resultados = []
    for alpha in [0.05, 0.025, 0.01]:
        var_norm = np.percentile(sim_norm, alpha * 100)
        es_norm = sim_norm[sim_norm <= var_norm].mean()
        var_t = np.percentile(sim_t, alpha * 100)
        es_t = sim_t[sim_t <= var_t].mean()
        
        resultados.append({
            'Confianza': f"{(1-alpha)*100}%",
            'VaR_Normal': var_norm,
            'ES_Normal': es_norm,
            'VaR_TStudent': var_t,
            'ES_TStudent': es_t
        })
    return pd.DataFrame(resultados)

# ============================================
# CONFIGURACIÓN DE STREAMLIT
# ============================================
st.set_page_config(
    page_title="Análisis de Riesgo - Paladio",
    layout="wide"
)

st.title(" Análisis de Riesgo del Paladio (PA=F)")
st.markdown("---")

# Sidebar (Barrita de al lado)
with st.sidebar:
    st.header(" Configuración")
    
    ticker = st.text_input("Símbolo del activo", value="PA=F")
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Fecha inicio", datetime(2010, 1, 1))
    with col2:
        end_date = st.date_input("Fecha fin", datetime.now())
    
    window = st.number_input("Ventana rolling (días)", min_value=50, max_value=500, value=252)
    
    st.markdown("---")
    load_data = st.button("Cargar Datos", type="primary", use_container_width=True)

# Cargamos los datos
@st.cache_data
def cargar_datos(ticker, start, end):
    try:
        data = descargar_datos(ticker, start, end)
        if data.empty:
            return None
        return data
    except Exception as e:
        st.error(f"Error al cargar datos: {e}")
        return None

if load_data or 'data' not in st.session_state:
    with st.spinner("Cargando datos..."):
        data = cargar_datos(ticker, start_date, end_date)
        if data is not None:
            st.session_state.data = data
            st.session_state.retornos = calcular_rendimientos(data)
            st.success(f"Datos cargados: {len(st.session_state.retornos)} días")
            st.session_state.datos_cargados = True


# Main Menu
if st.session_state.get('datos_cargados', False):
    retornos = st.session_state.retornos
    data = st.session_state.data
    
    # Métricas
    st.subheader("Resumen del Activo")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        precio = extract_value(data['Close'].iloc[-1])
        st.metric("Precio Actual", f"${precio:.2f}")
    
    with col2:
        ret_mean = extract_value(retornos.mean())
        st.metric("Rendimiento Promedio", f"{ret_mean:.4f}%")
    
    #with col3:
        #ret_std = extract_value(retornos.std())
       #st.metric("Volatilidad", f"{ret_std:.4f}%")
    
    with col4:
        st.metric("Datos Históricos", f"{len(retornos)} días")
    
    # Tabs
    tabs = st.tabs([
        " Rendimientos",
        " VaR y ES Estático",
        " Rolling Window",
        " Violaciones",
        " Monte Carlo",
        " Volatilidad Móvil"
    ])
    
    
    # Apartado 1: RENDIMIENTOS
    
    with tabs[0]:
        st.header("Análisis de Rendimientos")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=retornos.index, y=retornos.values,
            mode='lines', name='Rendimientos Diarios',
            line=dict(color='blue', width=1)
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        fig.update_layout(title="Rendimientos Diarios del Paladio",
                         xaxis_title="Fecha", yaxis_title="Rendimiento (%)",
                         height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Estadísticas
        media_val = extract_value(retornos.mean())
        mediana_val = extract_value(retornos.median())
        std_val = extract_value(retornos.std())
        max_val = extract_value(retornos.max())
        min_val = extract_value(retornos.min())
        k = extract_value(kurtosis(retornos.dropna()))
        s = extract_value(skew(retornos.dropna()))
        
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            st.metric("Media", f"{media_val:.4f}%")
            st.metric("Mediana", f"{mediana_val:.4f}%")
        with col_b:
            st.metric("Varianza", f"{std_val:.4f}%")
            st.metric("Máximo", f"{max_val:.4f}%")
        with col_c:
            st.metric("Mínimo", f"{min_val:.4f}%")
            st.metric("Curtosis", f"{k:.4f}")
            st.metric("Sesgo", f"{s:.4f}")
        
        # Interpretación
        st.subheader(" Interpretación")
        if k > 3:
            st.info(" **Curtosis leptocúrtica**: Distribución con colas pesadas. Los eventos extremos son más probables que en una distribución normal.")
        elif k < 3:
            st.info(" **Curtosis platicúrtica**: Distribución con colas ligeras.")
        else:
            st.info(" **Curtosis mesocúrtica**: Similar a distribución normal.")
        
        if s > 0:
            st.info(" **Sesgo positivo**: Cola derecha más larga (rendimientos positivos extremos).")
        elif s < 0:
            st.info(" **Sesgo negativo**: Cola izquierda más larga (pérdidas extremas).")
    
    # Apartado 2: VAR y ES
    with tabs[1]:
        st.header("VaR y ES ")
        
        metodo = st.selectbox("Selecciona el método", 
                              ["Histórico", "Normal (Paramétrico)", "t-Student (Paramétrico)"])
        
        mu = extract_value(retornos.mean())
        sigma = extract_value(retornos.std())
        df_t, loc_t, scale_t = t.fit(retornos)
        
        alphas = [0.05, 0.025, 0.01]
        niveles = ["95%", "97.5%", "99%"]
        resultados_medidas = []
        
        if metodo == "Histórico":
            from numpy import percentile
            for alpha, nivel in zip(alphas, niveles):
                var = np.percentile(retornos, alpha * 100)
                es = retornos[retornos <= var].mean()
                resultados_medidas.append({"Nivel": nivel, "VaR": extract_value(var), "ES": extract_value(es)})
        elif metodo == "Normal (Paramétrico)":
            for alpha, nivel in zip(alphas, niveles):
                var = norm.ppf(alpha, mu, sigma)
                es = Expected_Shortfall_normal(alpha, mu, sigma)
                resultados_medidas.append({"Nivel": nivel, "VaR": var, "ES": es})
        else:
            for alpha, nivel in zip(alphas, niveles):
                var = t.ppf(alpha, df_t, loc_t, scale_t)
                es = Expected_Shortfall_tstudent(alpha, df_t, loc_t, scale_t)
                resultados_medidas.append({"Nivel": nivel, "VaR": var, "ES": es})
        
        df_resultados = pd.DataFrame(resultados_medidas)
        df_resultados['VaR (%)'] = df_resultados['VaR'].apply(lambda x: f"{x:.4f}%")
        df_resultados['ES (%)'] = df_resultados['ES'].apply(lambda x: f"{x:.4f}%")
        
        st.dataframe(df_resultados[['Nivel', 'VaR (%)', 'ES (%)']], use_container_width=True)
        
        # Gráfico
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(niveles))
        width = 0.35
        ax.bar(x - width/2, df_resultados['VaR'], width, label='VaR', color='blue', alpha=0.7)
        ax.bar(x + width/2, df_resultados['ES'], width, label='ES', color='red', alpha=0.7)
        ax.set_xlabel('Nivel de Confianza')
        ax.set_ylabel('Valor (%)')
        ax.set_title(f'Comparación VaR vs ES - {metodo}')
        ax.set_xticks(x)
        ax.set_xticklabels(niveles)
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    # Apartado 3
    with tabs[2]:
        st.header("Rolling VaR y ES - Método Histórico")
        
        with st.spinner("Calculando rolling statistics..."):
            resultados_roll = rolling_VaR_historico(retornos, window=window)
        
        st.write(f"**Total de predicciones:** {len(resultados_roll)}")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=resultados_roll['Fecha'], y=resultados_roll['Retorno_Real'],
                                mode='lines', name='Retornos Reales', line=dict(color='gray', width=0.8), opacity=0.5))
        fig.add_trace(go.Scatter(x=resultados_roll['Fecha'], y=resultados_roll['VaR_95%'],
                                mode='lines', name='VaR 95%', line=dict(color='blue', width=1.5, dash='dash')))
        fig.add_trace(go.Scatter(x=resultados_roll['Fecha'], y=resultados_roll['VaR_99%'],
                                mode='lines', name='VaR 99%', line=dict(color='red', width=1.5, dash='dash')))
        fig.add_trace(go.Scatter(x=resultados_roll['Fecha'], y=resultados_roll['ES_95%'],
                                mode='lines', name='ES 95%', line=dict(color='lightblue', width=1, dash='dot')))
        fig.add_trace(go.Scatter(x=resultados_roll['Fecha'], y=resultados_roll['ES_99%'],
                                mode='lines', name='ES 99%', line=dict(color='lightcoral', width=1, dash='dot')))
        fig.update_layout(title=f'Rolling VaR y ES - Ventana de {window} días',
                         xaxis_title='Fecha', yaxis_title='Rendimiento (%)', height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Últimas 10 predicciones")
        st.dataframe(resultados_roll.tail(10).round(2), use_container_width=True)
    
    # Apartado 4: VIOLACIONES
    with tabs[3]:
        st.header("Análisis de Violaciones del VaR")
        
        with st.spinner("Calculando violaciones..."):
            resultados_roll = rolling_VaR_historico(retornos, window=window)
            viol_95 = (resultados_roll['Retorno_Real'] < resultados_roll['VaR_95%']).sum()
            viol_99 = (resultados_roll['Retorno_Real'] < resultados_roll['VaR_99%']).sum()
            total = len(resultados_roll)
            pct_95 = viol_95 / total * 100
            pct_99 = viol_99 / total * 100
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Violaciones VaR 95%", viol_95, delta=f"{pct_95:.2f}% (Esperado: 5%)")
            if abs(pct_95 - 5) <= 0.5:
                st.success("Evaluación: Excelente")
            elif abs(pct_95 - 5) <= 1:
                st.warning("Evaluación: Aceptable")
            else:
                st.error("Evaluación: Deficiente")
        
        with col2:
            st.metric("Violaciones VaR 99%", viol_99, delta=f"{pct_99:.2f}% (Esperado: 1%)")
            if abs(pct_99 - 1) <= 0.2:
                st.success("Evaluación: Excelente")
            elif abs(pct_99 - 1) <= 0.5:
                st.warning("Evaluación: Aceptable")
            else:
                st.error("Evaluación: Deficiente")
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        axes[0].plot(resultados_roll['Fecha'], resultados_roll['Retorno_Real'], color='gray', alpha=0.5, linewidth=0.8)
        axes[0].plot(resultados_roll['Fecha'], resultados_roll['VaR_95%'], color='blue', linewidth=1.5, linestyle='--')
        viol_plot = resultados_roll[resultados_roll['Retorno_Real'] < resultados_roll['VaR_95%']]
        axes[0].scatter(viol_plot['Fecha'], viol_plot['Retorno_Real'], color='red', s=20, alpha=0.7)
        axes[0].set_title('Violaciones del VaR 95%')
        axes[0].legend(['Retornos', 'VaR 95%', 'Violaciones'])
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(resultados_roll['Fecha'], resultados_roll['Retorno_Real'], color='gray', alpha=0.5, linewidth=0.8)
        axes[1].plot(resultados_roll['Fecha'], resultados_roll['VaR_99%'], color='red', linewidth=1.5, linestyle='--')
        viol_plot = resultados_roll[resultados_roll['Retorno_Real'] < resultados_roll['VaR_99%']]
        axes[1].scatter(viol_plot['Fecha'], viol_plot['Retorno_Real'], color='darkred', s=20, alpha=0.7)
        axes[1].set_title('Violaciones del VaR 99%')
        axes[1].legend(['Retornos', 'VaR 99%', 'Violaciones'])
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
    

    # Apartado de las Simulaciones MONTE CARLO
    with tabs[4]:
        st.header("Simulación Monte Carlo")
        
        n_sim = st.slider("Número de simulaciones", min_value=1000, max_value=50000, value=10000, step=1000)
        
        with st.spinner(f"Ejecutando {n_sim} simulaciones..."):
            resultados_mc = monte_carlo_var_es(retornos, n_sim)
        
        st.dataframe(resultados_mc.round(4), use_container_width=True)
        
        # Gráfico de distribuciones
        mu = extract_value(retornos.mean())
        sigma = extract_value(retornos.std())
        sim_norm = np.random.normal(mu, sigma, n_sim)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        axes[0].hist(sim_norm, bins=50, color='blue', alpha=0.6, edgecolor='black')
        axes[0].axvline(x=np.percentile(sim_norm, 5), color='red', linestyle='--', label='VaR 95%')
        axes[0].set_title('Distribución Normal Simulada')
        axes[0].legend()
        
        df_t, loc_t, scale_t = t.fit(retornos)
        sim_t = t.rvs(df_t, loc_t, scale_t, n_sim)
        axes[1].hist(sim_t, bins=50, color='green', alpha=0.6, edgecolor='black')
        axes[1].axvline(x=np.percentile(sim_t, 5), color='red', linestyle='--', label='VaR 95%')
        axes[1].set_title('Distribución t-Student Simulada')
        axes[1].legend()
        
        plt.tight_layout()
        st.pyplot(fig)
    

    # Apartado 6 VOLATILIDAD MÓVIL
    with tabs[5]:
        st.header("VaR con Volatilidad Móvil")
        st.markdown("**Fórmula:** $VaR(1-\\alpha) = q_\\alpha \\times \\sigma_{252,t}$")
        st.markdown("Donde $q_{0.95}=1.6449$ y $q_{0.99}=2.3263$")
        
        with st.spinner("Calculando..."):
            resultados_vol = VaR_volatilidad_movil(retornos, window=window)
            viol_95 = (resultados_vol['Retorno_Real'] < resultados_vol['VaR_95%']).sum()
            viol_99 = (resultados_vol['Retorno_Real'] < resultados_vol['VaR_99%']).sum()
            total = len(resultados_vol)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Violaciones VaR 95%", viol_95, delta=f"{viol_95/total*100:.2f}% (Esperado: 5%)")
        with col2:
            st.metric("Violaciones VaR 99%", viol_99, delta=f"{viol_99/total*100:.2f}% (Esperado: 1%)")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=resultados_vol['Fecha'], y=resultados_vol['Retorno_Real'],
                                mode='lines', name='Retornos Reales', line=dict(color='gray', width=0.8), opacity=0.5))
        fig.add_trace(go.Scatter(x=resultados_vol['Fecha'], y=resultados_vol['VaR_95%'],
                                mode='lines', name='VaR 95% (Vol Mov)', line=dict(color='blue', width=1.5, dash='dash')))
        fig.add_trace(go.Scatter(x=resultados_vol['Fecha'], y=resultados_vol['VaR_99%'],
                                mode='lines', name='VaR 99% (Vol Mov)', line=dict(color='red', width=1.5, dash='dash')))
        fig.update_layout(title=f'VaR con Volatilidad Móvil - Ventana {window} días',
                         xaxis_title='Fecha', yaxis_title='Rendimiento (%)', height=500)
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.caption("Desarrollado para el Análisis de Riesgo del Paladio | Métodos Cuantitativos")