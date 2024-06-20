#Creado por: Pedro Alan Velàzquez Romero
#2024-04-17

#Funciones que nos ayudarán con el manejo de la data y a arrojar resultados en la exploración de los datos

#Función para obtener el número de registros de las tablas y obtener algunos resultados diferentes 
#INPUT
#file_type - Tipo de archivo que vamos a leer para contar sus registros
#OUTPUT
#df_train - Data Frame del conjunto de entranamiento para futuros análisis
#df_test - Data Frame del conjunto de prueba para futuros análisis
def get_records_count(file_type):

    #Librerias a importar
    import pandas as pd
    import functions.general_functions as gf

    #Obtenemos la ruta de los archivos CSV
    ruta_archivos = gf.get_data_path(file_type)
    nombre_archivo = 'train.csv'
    nombre_archivo_t = 'test.csv'

    #Leemos el archivo csv
    df_train = pd.read_csv(ruta_archivos + nombre_archivo)
    df_test = pd.read_csv(ruta_archivos + nombre_archivo_t)

    #Obtenemos el número de registros para cada tabla
    num_reg_train = str(df_train.shape[0])
    num_reg_test = str(df_test.shape[0])
    num_reg_total = str(df_train.shape[0] + df_test.shape[0])

    #Imprimimos los totales
    print('El conjunto/tabla de entrenamiento contiene un total de ' + num_reg_train + ' registros')
    print('El conjunto/tabla de prueba contiene un total de ' + num_reg_test + ' registros')
    print('En total, se cuenta con ' + num_reg_total + ' registros')

    #Regresamos los df de entrenamiento y prueba para poder utilizarlos en las demás funciones
    return df_train, df_test

#Función para obtener aquellas columnas que presentan al menos un valor nulo, esto para ambos conjuntos (entrenamiento y prueba)
#INPUT
#df1 - Data Frame 1 del que obtendrán las columnas con al menos un valor nulo
#df2 - Data Frame 2 del que obtendrán las columnas con al menos un valor nulo
#OUTPUT
#df_cols - Data Frame que presenta las columnas con al menos un registro nulo
def get_columns_null_values(df1,df2):

    #Librerias a utilizar
    import pandas as pd
    
    #Obtenemos los nombres de las columnas en una lista para cada conjunto de datos
    col_train = list(df1.columns)
    col_test = list(df2.columns)
    
    #Proceso para obtener el número de registros null y el tipo de dato para cada columna
    #Creamos listas nulas para ir guardando los números
    num_nulls_train = []
    num_nulls_test = []
    type_train = []
    type_test = []
    #Hacemos el loop para ir guardando los números del conj de entrenamiento
    for x in col_train:
        #Obtenemos el número de registros null
        num_nulls = df1[x].isnull().sum()
        #Guardamos el número en la lista
        num_nulls_train.append(num_nulls)
        #Obtenemos el tipo de dato en la columna
        type_data = str(df1[x].dtypes)
        #Guardamos el tipo de dato en la lista
        type_train.append(type_data)
    #Hacemos el loop para ir guardando los números del conj de prueba
    for y in col_test:
        #Obtenemos el número de registros null
        num_nulls = df2[y].isnull().sum()
        #Guardamos el número en la lista
        num_nulls_test.append(num_nulls)
        #Obtenemos el tipo de dato en la columna
        type_data = str(df2[y].dtypes)
        #Guardamos el tipo de dato en la lista
        type_test.append(type_data)
    
    #Creamos lista que denota el tipo de conjunto de datos
    tipo_train = ['train'] * len(col_train)
    tipo_test = ['test'] * len(col_test)
    
    #Creamos el diccionario para el conj de entrenamiento para después convertirlo en df
    dic_train = {'type':tipo_train,'columns':col_train,'data_type':type_train,'num_nulls':num_nulls_train}
    #Creamos el diccionario para el conj de prueba para después convertirlo en df
    dic_test = {'type':tipo_test,'columns':col_test,'data_type':type_test,'num_nulls':num_nulls_test}
    
    #Creamos el df para el conj de entrenamiento
    df_col_train = pd.DataFrame(dic_train)
    #Creamos el df para el conj de prueba
    df_col_test = pd.DataFrame(dic_test)
    
    #Unimos ambos dfs
    df_cols = pd.concat([df_col_train,df_col_test], ignore_index = True)
    
    #Obtenemos aquellos registros con num_nulls > 0
    df_cols = df_cols[df_cols['num_nulls'] > 0]

    #Regresamos el df con aquellas columnas que presentan al menos un registro nulo
    return df_cols

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

#Función para mostrar información acerca de los coeficientes de sesgo de las columnas del conj. de entrenamiento
#INPUT
#df - Data Frame del cual otendremos las columnas a graficar
#from_column - Número entero que denota el número de inicio del rango de columnas sobre los cuales haremos las gráficas
#to_column - Número entero que denota el número de fin del rango de columnas sobre los cuales haremos las gráficas
#num_rows_ax - Número entero que denota el número de renglones que deseas tener en la gráfica final
#num_cols_ax - Número entero que denota el número de columnas que deseas tener en la gráfica final
#OUTPUT
#Gráfica de boxplot con los registros de cada columna
def show_boxplots(df,from_column,to_column,num_rows_ax,num_cols_ax):

    #Importamos las librerias a utilizar
    import matplotlib.pyplot as plt

    #Creamos el subconjunto de las columnas que queremos graficar en cada paso
    columnas_rec = list(df.columns)[from_column:to_column]

    #Creamos la plantilla para una gráfica del tamaño requerido
    fig, axes = plt.subplots(nrows=num_rows_ax, ncols=num_cols_ax, figsize=(27, 15))

    #Aplanamos los ejes para poder iterar sobre ellos más fácil
    axes = axes.flatten()

    #Hacemos el loop sobre el cual graficaremos
    for i,column in enumerate(columnas_rec):
        
        #Creamos el histograma de la columna correspondiente
        df.boxplot(column,ax=axes[i], color = 'Blue')

        #Obtenemos el nombre de la columna correspondiente
        col_name = column[0:15]
    
        #Agregamos los títulos a las gráficas
        axes[i].set_title(f'{col_name} boxplot')

#Función para obtener los registros outliers de cada columna del df (se muestra resultado para todas las columnas numéricas del df)
#INPUT
#df - Data Frame del cual calcularemos el IQR de sus columnas
#from_column - parámetro que indica el inicio de las columnas sobre las cuales haremos los cálculos
#to_column - parámetro que indica el fin de las columnas sobre las cuales haremos los cálculos
#OUTPUT
#df_outliers - Data Frame con los valores outliers de la respectiva columna 
def get_outliers(df,from_column,to_column):

    #Importamos las librerias necesarias
    import pandas as pd

    #Creamos un df vacío para ahí ir metiendo cada registro
    df_outliers_all = pd.DataFrame()
    #Lista de los nombres de las columnas del df
    list_columns = list(df.columns)

    #Creamos el loop con el cual iremos obteniendo los resultados de cada columna
    for column in list_columns[from_column:to_column]:

        #Obtenemos el percentile 25 de la columna
        perc_25 = df[column].quantile(.25)
        #Obtenemos el percentile 75 de la columna
        perc_75 = df[column].quantile(.75)
        #Calculamos el IQR de la columna
        iqr = perc_75 - perc_25
    
        #Calculamos la cota inferior para el argumento del filtro (siempre es 1.5*iqr)
        lower_bound = perc_25 - 1.5*iqr
        #Calculamos la cota superior para el argumento del filtro (siempre es 1.5*iqr)
        upper_bound = perc_75 + 1.5*iqr
    
        #Del df solo nos quedamos con aquellos registros outliers de la respectiva columna y lo convertimos en df
        df_outliers = pd.DataFrame(df[column][(df[column] < lower_bound) | (df[column] > upper_bound)])
    
        #Hacemos reset del index del nuevo df 
        df_outliers.reset_index(inplace=True)
        #Eliminamos la columna que nos generó el reset del index
        df_outliers.drop(columns=['index'],inplace = True)
        #Renombramos la columna con los valores outliers
        df_outliers.rename(columns = {f'{column}':'value'},inplace = True)
        #Creamos una nueva columna con el nombre de la columna
        df_outliers['column_name'] = column
        #Reordenamos las columnas
        df_outliers = df_outliers[['column_name','value']]
        #Creamos una nueva columna con el lower bound del criterio
        df_outliers['lower_bound'] = lower_bound
        #Creamos una nueva columna con el upper bound del criterio
        df_outliers['upper_bound'] = upper_bound

        #Agregamos este df al df general de todas las columnas
        df_outliers_all = pd.concat([df_outliers_all,df_outliers])

    #Regresamos el df con los outliers
    return df_outliers_all

#Función para obtener el resumen del conteo de outiers para las columnas de un df
#INPUT
#df - Data Frame del cual obtendremos todas las métricas
#from_column - parámetro que indica el inicio de las columnas sobre las cuales haremos los cálculos
#to_column - parámetro que indica el fin de las columnas sobre las cuales haremos los cálculos
#OUTPUT
#df_outliers_count = Date Frame que contiene ya el conteo de registros outliers por columna
def summarize_outliers(df,from_column,to_column):

    #Librerias a importar
    import functions.data_exploratory_functions as dtef
    import pandas as pd

    #De la pregunta 2 del documento descubrimos que ninguna de las columnas que queremos analizar tiene valores nulos
    #por lo que podemos tomar el número total de registros del df para hacer los cálculos
    num_registros = df.shape[0]

    #Aplicamos la función en donde obtenemos los valores de los outliers para cada columna
    df_outliers = dtef.get_outliers(df,from_column,to_column)

    #Creamos el nuevo df que contendrán solo el número total de outliers por columna
    df_outliers_count = pd.DataFrame(df_outliers.groupby('column_name')['value'].count())
    #Hacemos reset del index
    df_outliers_count.reset_index(inplace = True)
    #Renombramos la columna 
    df_outliers_count.rename(columns = {'value':'num_outliers'},inplace = True)

    #Creamos la nuevac columna con el % de outliers para cada columna
    df_outliers_count['perc_outliers (%)'] = df_outliers_count['num_outliers'].apply(lambda x: round(x/num_registros*100,2))

    #Reordenamos los reegistros por el porcentaje de outliers (descendiente)
    df_outliers_count = df_outliers_count.sort_values(by = 'perc_outliers (%)',ascending = False)

    #Obtenemos números importantes a mostrar
    #Número total de columnas con al menos 1 outlier
    num_cols_outliers = df_outliers_count.shape[0]
    #Número total de registros outliers
    num_outliers = df_outliers_count['num_outliers'].sum()

    #Imprimimos algunos resultados importantes
    print('En total, ' + str(num_cols_outliers) + ' columnas (' + str(round(num_cols_outliers/164*100,2)) + '%) presentan al menos 1 registro outlier (de acuerdo al criterio IQR)')

    print('En total, se tienen ' + str(num_outliers) + ' registros outliers (' + str(round(num_outliers/(num_registros*164)*100,2)) + '%) dentro de todo el conjunto de datos')

    #Regresamos el df del resumen de la cuenta de outliers
    return df_outliers_count

#Función para obtener el primer resultado acerca de la agrupación de datos por variables: CONTEO DE REGISTROS
#INPUT
#df - Data Frame de donde analizaremos el nombre de sus columnas
#show_final_results - Flag que indica si queremos mostrar los enunciados/resultados finales del análisis
#OUTPUT
#list_climate - lista que contiene el nombre de las columnas relacionadas a variables climáticas
#list_soil - lista que contiene el nombre de las columnas relacionadas a variables del suelo
#list_sat - lista que contiene el nombre de las columnas relacionadas a variables satelitales
#list_other - lista que contiene el nombre de las columnas relacionadas a otro tipo de variables
def get_records_groups_count(df,show_final_results = True):

    #Lista que contiene el nombre de todas las columnas del df 
    df_columns = list(df.columns)

    #Listas en donde guardaremos los nombres de las columnas de cada grupo
    list_climate = list()
    list_soil = list()
    list_sat = list()
    list_other = list()

    #Iniciamos el loop en donde guardaremos cada nombre de columna en su lista correspondiente
    for column in df_columns:

        #Condición para las columnas de tipo climática
        if column.upper().startswith('WORLDCLIM_BIO'):
            #Agregamos el nombre de la columna a la lista correspondiente
            list_climate.append(column)
        #Condición para las columnas de tipo de suelo
        elif column.upper().startswith('SOIL'):
            #Agregamos el nombre de la columna a la lista correspondiente
            list_soil.append(column)
        #Condición para las columnas de tipo satelital
        elif column.upper().startswith('MODIS') or column.upper().startswith('VOD'):
            #Agregamos el nombre de la columna a la lista correspondiente
            list_sat.append(column)
        #Si no entra en ninguna de las 3 categorias entonces lo mandamos a la lista de otros
        else:
            list_other.append(column)

    #Ya que tenemos las listas, imprimimos los resultados deseados dependiendo si el flag inicial es True o False
    if show_final_results:
        print('En total, se tienen ' + str(len(list_climate)) + ' (' + str(round(len(list_climate)/len(df_columns)*100,2)) +'%) columnas relacionadas a variables CLIMÁTICAS')
        print('En total, se tienen ' + str(len(list_soil)) + ' (' + str(round(len(list_soil)/len(df_columns)*100,2)) +'%) columnas relacionadas a variables DEL SUELO')
        print('En total, se tienen ' + str(len(list_sat)) + ' (' + str(round(len(list_sat)/len(df_columns)*100,2)) +'%) columnas relacionadas a variables SATELITALES')
        print('En total, se tienen ' + str(len(list_other)) + ' (' + str(round(len(list_other)/len(df_columns)*100,2)) +'%) columnas relacionadas a variables OTRAS')
    
    return list_climate, list_soil, list_sat, list_other

#Función para graficar las tablas de correlación para las diferentes columnas del df con respecto a un grupo en específico
#INPUT
#df - Data Frame del cual haremos las gráficas de correlación de cada columna por cada grupo elegido
#group_name - Nombre del grupo de columnas sobre el cual se desea obtener las correlaciones, valores permitidos: climáticas, suelo, satelitales
#OUTPUT
#gráfica de correlación entre las diferentes columnas del grupo elegido
#df_corr - Gráfica de correlación de las columnas elejidas

#IDEAS DE PLOTS : 
# GRAFICAR SUS DESCRIPCIONES (FUNCIÓN DESCRIBE)
# GRAFICAR LAS VARIABLES VS UNA VARIABLE OBJETIVO (PENSAR BIEN COMO SEPARAR ESTO PORQUE HAY VARIAS VARIABLES OBJETIVO)
def plot_corr_columns(df,group_name,show_summary_results = False):

    #Librerias a importar
    import functions.data_exploratory_functions as dtef
    import seaborn as sns
    import matplotlib.pyplot as plt

    #Llamamos a la función para obtener las listas de cada grupo
    list_climate, list_soil, list_sat, list_other = dtef.get_records_groups_count(df,show_summary_results)

    #Creamos una nueva variable con base en el nombre del grupo que se elija para poder hacer su gráfica
    if group_name == 'climáticas':
        #Creamos el df de correlación sobre las columnas del grupo elegido
        df_corr = df[list_climate].corr()
        #Creamos la gráfica de correlación con sus respectivos parámetros
        plt.figure(figsize=(13, 6))
        sns.heatmap(df_corr, vmax=1, annot=True, linewidths=.5)
        plt.xticks(rotation=30, horizontalalignment='right')
        plt.title('Tabla de correlación para columnas climáticas',fontsize = 20)
        plt.show()
    elif group_name == 'suelo':
        #Creamos el df de correlación sobre las columnas del grupo elegido
        df_corr = df[list_soil].corr()
        #Creamos la gráfica de correlación con sus respectivos parámetros
        plt.figure(figsize=(40, 30))
        sns.heatmap(df_corr, vmax=1, annot=True, linewidths=.5)
        plt.xticks(rotation=30, horizontalalignment='right')
        plt.title('Tabla de correlación para columnas del suelo',fontsize = 40)
        plt.show()
    elif group_name == 'satelitales':
        #Creamos el df de correlación sobre las columnas del grupo elegido
        df_corr = df[list_sat].corr()
        #Creamos la gráfica de correlación con sus respectivos parámetros
        plt.figure(figsize=(80, 40))
        sns.heatmap(df_corr, vmax=1, annot=True, linewidths=.5)
        plt.xticks(rotation=30, horizontalalignment='right')
        plt.title('Tabla de correlación para columnas satelitales',fontsize = 80)
        plt.show()
    else:
        #Si no proporcionan un nombre de grupo válido entonces se lo informamos
        print('Grupo no encontrado')

    #Regresamos el df de correlación final sobre el grupo elegido
    return df_corr

#Función para obtener aquellas columnas con correlación casi perfecta (cercana 1 en valor absoluto) o casi nula (independientes, cercano a 0)
#INPUT
#
#OUTPUT
#
#AQUI LO QUE SE PRETENDE ES BUSCAR AQUELLOS VALORES <0.1 EN VALOR ABSOLUTO Y >0.9 EN VALOR ABS PARA SABER QUE ESTAN CASI PERFECTAMENTE CORRELACIONADOS O SON INDEPENDIENTES
def get_corr_columns(df_corr):

    #Librerias a importar

    import numpy as np

    lista_values = list()
    lista_ind = list()
    lista_ind_col = list()
    for column in list(df_corr.columns):
        for x in list(df_corr[column]):
            lista_values.append(x)
            if abs(x) < 0.1:
                lista_ind.append(x)
                lista_ind_col.append(column)

    df_corr.where(df_corr.eq(-0.06002508098663719)).stack().index.tolist()[0]
    
    np.quantile(lista_values,.25)