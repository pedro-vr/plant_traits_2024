#Creado por: Pedro Alan Velàzquez Romero
#2024-04-17

#Funciones generales para leer data

#Función que devuelve la ruta hacia la data que se selecciona (csv o imagenes de entrenamiento/prueba)
#INPUT
#type_of_data - Tipo de data (archivo) del que quieres obtener la ruta, solo acepta valores "csv", "train_images" y "test_images"
#OUTPUT
#data_path - Ruta final (en string) del archivo elejido a leer
def get_data_path(type_of_data):

	#Variable que regresará la ruta para la data deseada
	data_path = ''

	#Si se desea la data de csv entonces regresamos la ruta pertinenete
	if type_of_data == 'csv':
		data_path = '/Users/pedrovela/Documents/Git_repos/plant_traits_2024/Data/'
	#Si se desea la data de imágenes de entrenamiento entonces regresamos la ruta pertinenete
	elif type_of_data == 'train_images':
		data_path = '/Users/pedrovela/Docs/Datasets - ML/planttraits2024/train_images/'
	#Si se desea la data de imágenes de prueba entonces regresamos la ruta pertinenete
	elif type_of_data == 'test_images':
		data_path = '/Users/pedrovela/Docs/Datasets - ML/planttraits2024/test_images/'
	#Si el tipo de data ingresada no es una de las disponibles se manda mensaje
	else:
		data_path = 'Tipo de data no existente'

	return data_path

#Función para crear los histogramas de cada columna del conjunto de datos de prueba (el de entrenamiento contiene 
#las mismas columnas + las variables a predecir que no nos interesa incluir en este análisis)
#INPUT
#df - Data Frame del cual otendremos las columnas a graficar
#from_column - Número entero que denota el número de inicio del rango de columnas sobre los cuales haremos las gráficas
#to_column - Número entero que denota el número de fin del rango de columnas sobre los cuales haremos las gráficas
#num_rows_ax - Número entero que denota el número de renglones que deseas tener en la gráfica final
#num_cols_ax - Número entero que denota el número de columnas que deseas tener en la gráfica final
#OUTPUT
#Gráficas de distribución de todas las columnas dentro del rango elegido
def plot_hist_columns(df,from_column,to_column,num_rows_ax,num_cols_ax):

    #Librerias a utilizar
    import numpy as np
    import matplotlib.pyplot as plt

    #Creamos la lista con los nombres de columnas a plotear
    columnas_rec = list(df.columns)[from_column:to_column]

    #Creamos la plantilla para una gráfica del tamaño requerido
    fig, axes = plt.subplots(nrows=num_rows_ax, ncols=num_cols_ax, figsize=(10, 10))

    #Aplanamos los ejes para poder iterar sobre ellos más fácil
    axes = axes.flatten()

    #Lista para definir el número de columna que estamos graficando
    list_num_col = list(range(from_column,to_column))

    #Hacemos el loop sobre cada columna para hacer su gráfica
    for i,column in enumerate(columnas_rec):
        
        #Creamos el histograma de la columna correspondiente
        df[column].hist(ax=axes[i], #Definimos el eje sobre el cual estamos trabajando
                        edgecolor='white', #Color del borde
                        color='#69b3a2' #Color de los intervalos
                       )

        #Obtenemos el nombre de la columna correspondiente
        col_name = list_num_col[i]
    
        #Agregamos los títulos a las gráficas
        axes[i].set_title(f'COL_{col_name} dist.') 
        axes[i].set_xlabel(f'COL_{col_name}') 
        axes[i].set_ylabel('Frecuencia') 

    #Ajustamos la gráfica
    plt.tight_layout()

    #Mostramos la gráfica final
    plt.show()

#Función para mostrar información acerca de los coeficientes de sesgo de las columnas del conj. de entrenamiento
#INPUT
#df - Data Frame del cual obtendremos las columnas sobre las cuales calcular el coeficiente
#from_column - Número entero que denota el número de inicio del rango de columnas sobre los cuales haremos los cálculos
#to_column - Número entero que denota el número de fin del rango de columnas sobre los cuales haremos los cálculos
#OUTPUT
#Enunciados con resultados importantes de estos coeficientes
#Gráfica de histograma de los coeficientes de sesgo
def show_skew_coeff(df,from_column,to_column):
    #Significado coef. de sesgo
    # = 0: se parece más a la normal
    # > 0: sesgada a la izq
    # < 0: sesgada a la der

    #Importamos las librerías a utilizar
    from scipy.stats import skew 
    import matplotlib.pyplot as plt

    #Creamos una lista para ir guardando todos los coeficientes
    lista_skew = []
    #Iniciamos el loop donde vamos a ir guardando todos los coeficientes de sesgo para cada columna
    for column in list(df.columns)[from_column:to_column]:
        #Calculamos el coeficiente de sesgo para la columna
        skew_col = skew(df[column], axis = 0, bias = True)
        #Agregamos el coeficiente calculado a la lista iniciañ
        lista_skew.append(skew_col)

    #Imprimimos algunos resultados importantes
    num_col_sesgo1 = len((list(filter(lambda x: x >= 1, lista_skew))))
    print('En total, hay ' + str(num_col_sesgo1) + ' columnas con coeficiente de sesgo >= 1')
    num_col_sesgo_1 = len((list(filter(lambda x: x <= -1, lista_skew))))
    print('En total, hay ' + str(num_col_sesgo_1) + ' columnas con coeficiente de sesgo <= -1')
    num_col_sesgo_abs_1 = len((list(filter(lambda x: abs(x) < 1, lista_skew))))
    print('En total, hay ' + str(num_col_sesgo_abs_1) + ' columnas con coeficiente de sesgo cercano a cero')

    #Hacemos la gráfica de distribución de los coeficientes de sesgo
    plt.hist(lista_skew)
    plt.title("Coeficientes de sesgo para conj. de entrenamiento")
    plt.xlabel('Coeficiente de sesgo')
    plt.ylabel('Frecuencia')

