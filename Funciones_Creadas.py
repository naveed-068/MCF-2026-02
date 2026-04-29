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
