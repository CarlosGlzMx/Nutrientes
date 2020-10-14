#Importa las librerías a utilizar
import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

#Linea por gustos personales, elimina notación científica al imprimir valores de arrays de Numpy
np.set_printoptions(suppress = True)

#Lee los datos y selecciona los valores "x" y "y" del modelo de regresión
df = pd.read_csv("Datos_Alimentos.csv")
X = df[["Grasas","Proteínas","Carbohidratos","Sodio"]]
y = df["Calorías"]
#X = sm.add_constant(X)

#Genera el modelo de regresión y lo muestra
model = sm.OLS(y,X).fit()
print(model.summary())

#Guarda los coefficientes obtenidos para cada variable
BetaGrasas = model.params[0]
BetaProteinas = model.params[1]
BetaCarbohidratos = model.params[2]
BetaSodio = model.params[3]

#Genera la lista de residuos en base a la resta de los valores predecidos a los obtenidos
residuos = []
predicciones = []
for i in range(len(y)):
    predicciones.append(BetaGrasas*X.iloc[i][0] + BetaProteinas*X.iloc[i][1] + BetaCarbohidratos*X.iloc[i][2]\
        + BetaSodio*X.iloc[i][3])
    residuos.append(y[i]-predicciones[i])
residuos = np.array(residuos)

#Crea el tablero de gráficas y prepara sus 4 subplots
tablero = plt.figure(figsize=(10,7))
normalidad = tablero.add_subplot(2,2,1)
escopetazo = tablero.add_subplot(2,2,2)
histograma = tablero.add_subplot(2,2,3)
independencia = tablero.add_subplot(2,2,4)

#Genera la primera gráfica - Prueba de normalidad de los residuos
residuosOrdenados = np.sort(residuos)
estadisticoZ = []
for i in range(len(residuos)):
    estadisticoZ.append(stats.norm.ppf((i + 0.5)/len(residuos)))
normalidad.scatter(residuosOrdenados,estadisticoZ)
previoTendencia = np.polyfit(residuosOrdenados,estadisticoZ,1)
tendencia = np.poly1d(previoTendencia)
normalidad.plot(residuosOrdenados,tendencia(residuosOrdenados),"r--")
normalidad.grid(True)
normalidad.axhline(y = 0, color = "k", linewidth = 0.75)
normalidad.axvline(x = 0, color = "k", linewidth = 0.75)
normalidad.set_title("Normalidad de los residuos")
normalidad.set_xlabel("Residuos")
normalidad.set_ylabel("Estadístico z")

#Genera la segunda gráfica - Prueba de desviación de los residuo
escopetazo.scatter(predicciones,residuos)
escopetazo.grid(True)
escopetazo.set_title("Variación de los residuos")
escopetazo.set_xlabel("Valores predecidos")
escopetazo.set_ylabel("Residuos")

#Genera la tercera gráfica - Histrograma de los residuos
histograma.hist(residuos)
histograma.grid(True)
histograma.set_title("Histograma de los residuos")
histograma.set_xlabel("Residuos")
histograma.set_ylabel("Frecuencia")

#Genera la cuarta gráfica - Prueba de independencia de los residuos
contadores = []
for i in range(len(residuos)):
    contadores.append(i + 1)
independencia.plot(contadores,residuos)
independencia.grid(True)
independencia.set_title("Independencia de los residuos")
independencia.set_xlabel("Números de registro")
independencia.set_ylabel("Residuos")

#Detalles finales para ajustar y mostrar el conjunto de datos
plt.tight_layout(pad = 4)
plt.suptitle("Análisis de residuos", size = 20)
plt.show()
