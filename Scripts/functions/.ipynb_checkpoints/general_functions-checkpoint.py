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