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
import scipy.stats
from scipy.stats import kurtosis, skew, shapiro, norm

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
        rendimientos (_type_): _description_
    """
    sesg = skew(rendimientos)
    return sesg
    