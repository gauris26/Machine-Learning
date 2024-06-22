import pickle
import os
import signal
import numpy as np
import pandas as pd
from telegram import Update, BotCommand, ParseMode, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
import geopandas as gpd
from shapely.geometry import Point

class ChileSchoolClustering:
    def __init__(self, model_path, shapefile_path):
        self.latitude = None
        self.longitude = None
        self.model_path = model_path
        self.shapefile_path = shapefile_path
        self.chilean_school_kmeans = None
        self.mapping_dict = {0: 'Norte Chico', 1: 'Zona Sur', 2: 'Norte Grande', 3: 'Zona Central', 4: 'Zona Austral'}
        self.cluster_centers = None
        self.cluster_labels = None
        self.chile_geo_pandas = None

    def load_model(self):
        with open(self.model_path, 'rb') as file:
            self.chilean_school_kmeans = pickle.load(file)
        self.cluster_centers = self.chilean_school_kmeans.cluster_centers_
        self.cluster_labels = self.chilean_school_kmeans.labels_

    def predict_cluster(self):
        input_data = pd.DataFrame([[self.latitude, self.longitude]], columns=self.chilean_school_kmeans.feature_names_in_)
        cluster_prediction = self.chilean_school_kmeans.predict(input_data)
        return cluster_prediction[0]

    def get_mapped_prediction(self, cluster_prediction):
        return self.mapping_dict[cluster_prediction]

    def process_chile_boundaries(self):
        self.chile_geo_pandas = gpd.read_file(self.shapefile_path, encoding='utf-8')
        # Convert the coordinate reference system (CRS) to WGS84 (EPSG:4326)
        self.chile_geo_pandas = self.chile_geo_pandas.to_crs(epsg=4326)

    def process(self,latitude, longitude):
        self.latitude = latitude
        self.longitude = longitude
        cluster_prediction = self.predict_cluster()
        mapped_prediction = self.get_mapped_prediction(cluster_prediction)

        result = {
            'Unique Clusters': np.unique(self.cluster_labels),
            'Cluster Prediction': cluster_prediction,
            'Mapped Prediction': mapped_prediction,
            'Is Anomalous': not self.is_point_in_chile(self.latitude, self.longitude)
        }
        return result

    def is_point_in_chile(self, latitude, longitude):
        point = Point(longitude, latitude)
        # Check if the point is within any of the geometries in the shapefile
        return self.chile_geo_pandas.contains(point).any()
    
    def get_chile_region(self, latitude, longitude):
        point = Point(longitude, latitude)
        return self.chile_geo_pandas.loc[self.chile_geo_pandas.contains(point)]['Region']

TELEGRAM_TOKEN = '7373795648:AAHTqveLbEkXgesaul0MPm07hr3X8emi4vs'    
# Create the Updater and pass it your bot's token.
updater = Updater(TELEGRAM_TOKEN)
resource_folder = '/home/gauris26/Machine-Learning/chilean-school-zone-clustering-telegram-bot/resources'
chile_shapefile_folder = f'{resource_folder}/chile_shapefile'
images_folder = f'{resource_folder}/images'
model_path = f'{resource_folder}/chilean_schools_clustering.pickle'
shapefile_path = f'{chile_shapefile_folder}/regiones_chile_2020_bnc_qgiswriteout_epsg32719Polygon.shp'

isSessionActive =  False
#Load models
clustering = ChileSchoolClustering(model_path, shapefile_path)
print("Cargando modelo de clusterizacion")
clustering.load_model()
print("Cargando modelo geografico")
clustering.process_chile_boundaries()
print("Cargado de modelos completado")

# Define command handlers
def start(update: Update, context: CallbackContext) -> None:
    """Send a message when the command /start is issued."""
    global clustering
    global isSessionActive
    if isSessionActive:
        update.message.reply_text("Ya hay una sesión activa")
        return
    
    update.message.reply_html("<b>Servicio de Zonificación Escolar Chileno (ZECH)</b> 🇨🇱")
    caption = "Bienvenido al Servicio <b>ZECH</b>"
    image_path = f'{images_folder}/Mapa_zonas_naturales_de_Chile.jpg'
    update.message.reply_photo(
        photo=open(image_path, 'rb'), 
        caption=caption,
        parse_mode=ParseMode.HTML
    )
    
    update.message.reply_text('Para iniciar la clusterización, por favor suministrar una geolocalización desde la opción de adjuntar ↘️')
    isSessionActive = True

def echo(update: Update, context: CallbackContext) -> None:
    """Echo the user message."""
    global clustering
    global isSessionActive
    if isSessionActive == False:
        update.message.reply_text("Tiene que iniciar una nueva sesión")
        return
    
    update.message.reply_text('No es una opción válida ❤️‍🩹, favor enviar una geolocalización📍')

def stop(update: Update, context: CallbackContext) -> None:
    """Stop the bot."""
    update.message.reply_text('Deteniendo el bot...')
    global clustering
    global isSessionActive
    isSessionActive = False
    #os.kill(os.getpid(), signal.SIGINT)

def dashboard(update: Update, context: CallbackContext) -> None:
    """Show the dashboard."""
    dashboard_link = "https://app.powerbi.com/groups/me/reports/7d43e829-90cb-40ec-88d1-d45851805858"
    button = InlineKeyboardButton("Mostrar Dashboard", url=dashboard_link)
    keyboard = InlineKeyboardMarkup([[button]])

    dashboard_message = 'Para abrir el dashboard, presionar el siguiente botón.'
    # Enviar mensaje con el botón
    update.message.reply_text(dashboard_message, reply_markup=keyboard)

def handle_location(update: Update, context: CallbackContext) -> None:
    """Handle the user location."""
    try:
        global clustering
        global isSessionActive
        if isSessionActive == False:
            update.message.reply_text("Tiene que iniciar una nueva sesión")
            return
    
        user_location = update.message.location
        latitude = user_location.latitude
        longitude = user_location.longitude
        response_message = f"Hemos recibido su geolocalización📍:\n<b>Latitud</b>: {latitude}\n<b>Longitud</b>: {longitude}"
        update.message.reply_html(response_message)

        update.message.reply_text("Clusterizando ⚙️ ...")

        is_out_of_chile_boundaries = not clustering.is_point_in_chile(latitude, longitude)

        if is_out_of_chile_boundaries:
            update.message.reply_html("Se ha detectado una <b>Ubicación Fuera de Rango</b>⚠️")
        else:
            result = clustering.process(latitude, longitude)
            cluster_prediction = result['Cluster Prediction']
            mapped_prediction = result['Mapped Prediction']

            images_chile_zones = [
                f'{images_folder}/Mapa_zonas_naturales_de_Chile_Norte_Chico.jpg',
                f'{images_folder}/Mapa_zonas_naturales_de_Chile_Zona_Sur.jpg',
                f'{images_folder}/Mapa_zonas_naturales_de_Chile_Norte_Grande.jpg',
                f'{images_folder}/Mapa_zonas_naturales_de_Chile_Zona_Central.jpg',
                f'{images_folder}/Mapa_zonas_naturales_de_Chile_Zona_Austral.jpg',
            ]

            caption = f"<b>Zona Encontrada</b> ✅: {mapped_prediction} 🇨🇱"
            image_path = images_chile_zones[cluster_prediction]
            update.message.reply_photo(
                photo=open(image_path, 'rb'), 
                caption=caption, 
                parse_mode=ParseMode.HTML
            )
            chile_region = clustering.get_chile_region(latitude, longitude)

            if(not chile_region.empty):
                update.message.reply_text(chile_region.values[0])
    except:
        update.message.reply_text('Se ha encontrado un error inesperado, por favor intente de nuevo.')

def main() -> None:
    """Start the bot."""

    # Get the dispatcher to register handlers
    dispatcher = updater.dispatcher

    # Register handlers
    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(CommandHandler("stop", stop))
    dispatcher.add_handler(CommandHandler("dashboard", dashboard))
    dispatcher.add_handler(MessageHandler(Filters.location, handle_location))
    dispatcher.add_handler(MessageHandler(Filters.all & ~Filters.command, echo))

    commands = [
        BotCommand("start", "Iniciar bot"),
        BotCommand("stop", "Detener bot"),
        BotCommand("dashboard", "Mostrar dashboard"),
    ]

    #Set custom commands for the bot
    updater.bot.set_my_commands(commands)

    updater.start_polling()

    print("Servicio escuchando peticiones")
    updater.idle()

# Run the main function
if __name__ == '__main__':
    main()