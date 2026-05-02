# MCF-2026-02

Aquí se va a describir todo lo que se hizo en el proyecto

Objetivo del Proyecto

El objetivo principal de este proyecto fue realizar un análisis de riesgo completo del activo financiero Paladio (PA=F) utilizando los metodos vistos en clase. Por lo que se busco calcular las principales medidas de riesgo como el Value at Risk (VaR) y el Expected Shortfall (ES) mediante enfoques paramétricos, históricos y de simulación Monte Carlo. Además, se desarrolló una aplicación web interactiva en Streamlit para visualizar los resultados de forma accesible y dinámica.

Descripción del Activo (¿Por qué lo eligimos)

Elegimos como activo financiero al Paladio ya que es utilizado principalmente en la industria automotriz para convertidores catalíticos. Es un activo conocido por su alta volatilidad y sensibilidad a factores económicos y geopolíticos.

Metodología de Trabajo (¿Cómo se trabajo?)

El proyecto se estructuró en dos partes principales:

Desarrollo en Notebook: Se creó un notebook de Jupyter donde se realizaron todas las pruebas, cálculos y visualizaciones iniciales. Aquí se definieron y validaron las funciones necesarias para el análisis.

Aplicación en Streamlit: Con base en el notebook, se construyó una aplicación web interactiva que permite a cualquier usuario cargar datos, ajustar parámetros y visualizar los resultados de riesgo sin necesidad de conocer el código subyacente.

Funciones Implementadas

Se creó un módulo separado llamado Funciones_Creadas.py que contiene todas las funciones reutilizables, incluyendo:

Descarga de datos desde Yahoo Finance.

Cálculo de rendimientos diarios.

Cálculo de estadísticas descriptivas como curtosis y sesgo.

Cálculo de VaR y ES mediante métodos histórico, normal paramétrico y t-Student paramétrico.

Implementación de rolling windows para análisis dinámico.

Simulación de Monte Carlo.

Cálculo de VaR con volatilidad móvil.

Análisis de Rendimientos

El primer paso fue calcular los rendimientos diarios del Paladio. Se obtuvo una curtosis aproximadamente de 9.12, lo que indica una distribución leptocúrtica, es decir, con colas más pesadas que una distribución normal. Esto significa que el Paladio tiene una mayor probabilidad de experimentar eventos extremos, tanto positivos como negativos. El sesgo resultó ligeramente negativo, sugiriendo una leve asimetría hacia pérdidas extremas.

Resultados del VaR y ES Estáticos

Se calcularon el VaR y el ES para tres niveles de confianza: 95%, 97.5% y 99%, utilizando tres métodos diferentes.

Con el método histórico, se obtuvo que, por ejemplo, con un 95% de confianza, la pérdida máxima diaria esperada es de aproximadamente 3.5%, mientras que el ES (pérdida promedio cuando se supera el VaR) es de alrededor de 4.1%.

Con el método paramétrico asumiendo normalidad, los valores fueron ligeramente diferentes, dando un VaR del 95% de aproximadamente 3.8% y un ES de 4.8%.

El método t-Student arrojó resultados intermedios, con un VaR del 95% de 3.4% y un ES de 5.2%. Este método es preferible porque captura mejor las colas pesadas de la distribución del Paladio.

Análisis con Rolling Window

Se implementó un análisis con ventana móvil de 252 días para evaluar cómo evolucionan el VaR y el ES a lo largo del tiempo. Se generaron aproximadamente 3835 predicciones. Se analizaron las violaciones, es decir, los días en que la pérdida real superó al VaR predicho. Para el VaR del 95%, el porcentaje de violaciones fue cercano al 5% esperado, mientras que para el 99% fue ligeramente superior al 1%, indicando que el modelo subestima ligeramente el riesgo en niveles de confianza muy altos.

Método de Volatilidad Móvil

Se implementó un método simplificado donde el VaR se calcula únicamente como el producto de un percentil normal por la desviación estándar móvil, sin incluir la media. Los percentiles utilizados fueron 1.6449 para el 95% y 2.3263 para el 99%. Este método funcionó muy bien para el nivel del 95%, con un porcentaje de violaciones del 4.88%, prácticamente el esperado. Sin embargo, para el nivel del 99%, el porcentaje de violaciones fue del 1.59%, superando el 1% esperado, lo que indica que subestima el riesgo en escenarios extremos.

Simulación de Monte Carlo

Se realizaron simulaciones de Monte Carlo con 10,000 iteraciones para estimar el VaR y el ES bajo distribuciones normal y t-Student. Los resultados obtenidos fueron consistentes con los métodos paramétricos, validando así los cálculos anteriores. Para el nivel del 95%, por ejemplo, el VaR normal simulado fue de aproximadamente 3.7% y el ES normal simulado de 4.7%, mientras que con t-Student fueron 3.4% y 5.1% respectivamente.

Aplicación en Streamlit

Finalmente, se desarrolló una aplicación web en Streamlit que integra todos los análisis anteriores. La aplicación cuenta con seis pestañas principales: una para visualizar los rendimientos y estadísticas descriptivas, otra para calcular el VaR y ES estáticos con diferentes métodos, una tercera para el análisis con rolling window, una cuarta para evaluar las violaciones, una quinta para la simulación de Monte Carlo y una última para el método de volatilidad móvil. La aplicación permite al usuario ajustar parámetros como la ventana de tiempo y el número de simulaciones, y visualizar los resultados de forma interactiva.

Conclusiones Principales

En conclusión, el Paladio es un activo volátil con una distribución de rendimientos de colas pesadas, lo que hace que el método t-Student sea el más apropiado para medir su riesgo. El método de volatilidad móvil resultó ser muy preciso para el nivel de confianza del 95%, pero deficiente para el 99%. La simulación de Monte Carlo validó los resultados obtenidos con métodos paramétricos. Finalmente, la aplicación en Streamlit logró encapsular todo el análisis en una herramienta interactiva, amigable y de fácil uso para cualquier usuario interesado en evaluar el riesgo del Paladio.
