#Creado por: Pedro Alan Velàzquez Romero
#2024-04-17

#Funciones generales para leer data

#Función que devuelve la ruta hacia la data que se selecciona (csv o imagenes de entrenamiento/prueba)
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
        axes[i].set_ylabel('Frequency') 

    #Ajustamos la gráfica
    plt.tight_layout()

    #Mostramos la gráfica final
    plt.show()
