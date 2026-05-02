"""
En este codigo se crearán funciones para poder calcular lo que nos pide el proyecto.
Primero comencemos importando las librerias necesarias para el proyecto.
"""
#%%
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import datetime as dt
from scipy import stats
from scipy.stats import kurtosis, skew, shapiro, norm, t

#Funcion para descargar los datos de las acciones.
def descargar_datos(ticker, start_date, end_date):
    """
    Esta función descarga los datos de las acciones de Yahoo Finance.

    Args:
        ticker (String): Es el nombre clave del activo financiero al que se analizará, por ejemplo: "MSFT" Microsoft
        start_date (String): Es la fecha de inicio de la evaluacion
        end_date (String): Es la fecha hasta donde se va a evaluar el activo

    Returns:
        dataframe de pandas:Los datos que se descargarón de Yahoo Finance del activo financiero.
    """
    data = yf.download(ticker, start = start_date , end = end_date)
    #print(data.to_string())
    return data

#Función para el calculo de rendimientos por día

def rendimientos(data):
    """
    Función que se encarga de calcular los rendimientos del activo financiero
    Args:
        df (_dataframe_): Va a sacar la diferencia de los rendimientos entre dos días
    """
    rendimientos = data['Close'].pct_change().dropna()
    return rendimientos

#Curtosis
def curtosis(rendimientos):
    """
    Esta función se va a encargar de calcular la curtosis de los rendimientos.
    Recordemos que la Curtosis es una medida que tanto se ajustan 

    Args:
        rendimientos (array): Son los rendimientos diarios del activo financiero
    """
    Curt = kurtosis(rendimientos)
    return Curt
#Sesgo
def sesgo(rendimientos):
    """
    Esta función se va encargar de calcular el sesgo de los rendimientos.
    Recordemos que el sesgo va a ser una medida de

    Args:
        rendimientos (Dataframe): Los rendimientos diarios del activo financiero
    """
    sesg = skew(rendimientos)
    return sesg

#VaR
def Value_at_Risk(rendimientos, alpha):
    """
    Esta función se encargara de calcular el VaR de los rendimientos

    Args:
        alpha (float): El porcentaje del cuantil al que quieres llegar, es un numero entre 0 y 1 
        
        rendimientos (Dataframe): Los rendimientos diarios del activo financiero
    """
    if alpha < 1 and alpha > 0:
        VaR = np.percentile(rendimientos, (1-alpha)*100)
    else:
        return "Hay que elegir un alpha en el intervalo (0,1)"
    return VaR

def Expected_Shortfall(rendimientos, alpha):
    """
    Esta función se encargará de calcular el Spected_Shortfall a un porcentaje alpha
    de los rendimientos

    Args:
        rendimientos (Dataframe): Son los rendimientos del activo financiero
        alpha (float): El porcentaje alpha al que se quiere calcular, es un número entre 0 y 1
    """
    if alpha > 0 and alpha < 1:
        VaR = np.percentile(rendimientos, (1-alpha)*100)
        CVaR = rendimientos[rendimientos <= VaR].mean()
        return CVaR
    else:
        return "Hay que elegir un alpha que este entre (0,1)"
    

def Expected_Shortfall_tstudent(alpha, df, loc, scale):

    """
    Función para calcular el Expected Shortfall de una distribución t-student
    
    Args:
        alpha(float): Nivel de significancia
        df(float): Grados de libertad
        loc(float): Parámetro de locación
        scale(float): Parámetro de escala
    """
    
    # Valor crítico t para el alpha dado
    t_alpha = t.ppf(alpha, df)
    
    # Función de densidad en t_alpha
    pdf_t_alpha = t.pdf(t_alpha, df)
    
    # Fórmula paramétrica del ES para t-student
    # ES = loc + scale * [ -f(t_alpha)/α * ((df + t_alpha^2)/(df - 1)) ]
    ES_std = -pdf_t_alpha / alpha * ((df + t_alpha**2) / (df - 1))
    ES = loc + scale * ES_std
    
    return ES

def Expected_Shortfall_normal(alpha, mu, sigma):
      """
      Funcion para calcular el Expected Shortfall de una normal con media mu y varianza sigma

      Args:
          alpha (float): Alpha al que se quiere calcular el ES
          mu (_type_): La media de la normal
          sigma (_type_): La Varianza de la normal

      Returns:
          float: Se obtiene como resultado el CVaR 
      """
      z_alpha = stats.norm.ppf(alpha)  # Esto es el VaR
      ES = mu - sigma * (stats.norm.pdf(z_alpha) / alpha) 
      return ES

"""
def Rolling_Window_Recursive(retornos, i, Serie_T, window, alpha):
    
    Esta función me  va a permitir poder predecir los valores que tomara la
    medida de riesgo al día i

    Args:
        i (int): Días de los cuales quieres hacer la predicción
        Serie_T (Dataframe): Serie de Tiempo de donde se hará la predicción.
    input("Dime cuantos días más quisieras la predicción")
    j = +1
    
"""

def calcular_VaR_ES_historico(ventana):
    """
    Funcion que me va a ayudar a calcular el VaR y ES historicos

    Args:
        ventana (float):La ventana de días con la que se hae el Rollin Window

    Returns:
        Float: Regresa el VaR al 95, 99 y ES95 Y ES99
    """
    var_95 = np.percentile(ventana, 5)
    var_99 = np.percentile(ventana, 1)
    es_95 = ventana[ventana <= var_95].mean()
    es_99 = ventana[ventana <= var_99].mean()
    return var_95, var_99, es_95, es_99

def tabla_violaciones(df, nombre_metodo):
    """Calcula y muestra las violaciones para un método"""
    total = len(df)
    
    violaciones = {
        'Métrica': ['VaR 95%', 'VaR 99%', 'ES 95%', 'ES 99%'],
        'Violaciones': [
            df['Violacion_VaR_95'].sum(),
            df['Violacion_VaR_99'].sum(),
            df['Violacion_ES_95'].sum(),
            df['Violacion_ES_99'].sum()
        ],
        'Porcentaje (%)': [
            df['Violacion_VaR_95'].sum() / total * 100,
            df['Violacion_VaR_99'].sum() / total * 100,
            df['Violacion_ES_95'].sum() / total * 100,
            df['Violacion_ES_99'].sum() / total * 100
        ],
        'Esperado (%)': [5, 1, 5, 1]
    }
    
    return pd.DataFrame(violaciones)
def Expected_Shortfall_normal(alpha, mu, sigma):
    """
    Función para calcular el Expected Shortfall de una normal
    """
    from scipy import stats
    z_alpha = stats.norm.ppf(alpha)
    ES = mu - sigma * (stats.norm.pdf(z_alpha) / alpha)
    return ES

def Expected_Shortfall_tstudent(alpha, df, loc, scale):
    """
    Función para calcular el Expected Shortfall de una t-student
    """
    from scipy.stats import t
    t_alpha = t.ppf(alpha, df)
    pdf_t_alpha = t.pdf(t_alpha, df)
    ES_std = -pdf_t_alpha / alpha * ((df + t_alpha**2) / (df - 1))
    ES = loc + scale * ES_std
    return ES

def rolling_VaR_historico(retornos, window=252):
    """
    Calcula rolling VaR histórico
    """
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
    """
    VaR con volatilidad móvil
    """
    from scipy.stats import norm
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
    """
    Simulación Monte Carlo
    """
    from scipy.stats import norm, t
    mu = rendimientos.mean()
    sigma = rendimientos.std()
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