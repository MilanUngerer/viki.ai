#import Adafruit_DHT
import numpy as np
import math
import time
import requests
import datetime
import paho.mqtt.client as paho                     #mqtt library
import os
import json
import random


# importing  all the functions defined by me
from meteo import *
from sunpos import *
#from DHT11 import *

##### DATOS THINGSBOARD ###########
ACCESS_TOKEN='Aether0255'          #Token of your device
broker="thingsboard.cloud"           #host name
port=1883                            #data listening port

def on_publish(client,userdata,result):             #create function for callback
    print("data published to thingsboard \n")
    pass
client1= paho.Client("control1")                    #create client object
client1.on_publish = on_publish                     #assign function to callback
client1.username_pw_set(ACCESS_TOKEN)               #access token from thingsboard device
client1.connect(broker,port,keepalive=60)           #establish connection


##### DATOS GEOGRAFICOS ##########
location = (-33.05178823955611, -71.55734620441065)

##### SENSOR DHT11 ############### 
#sensor = Adafruit_DHT.DHT11 #Cambia por DHT22 y si usas dicho sensor
pin = 4 #Pin de GPIO en la raspberry donde conectamos el sensor


while True:

	######### SENSING DATA ##########################
#	humedad, temperatura = Adafruit_DHT.read_retry(sensor, pin)
	temperatura = random.randint(0,50) #provisorio para pruebas
	humedad = random.randint(0,50) #provisorio para pruebas

	###### METEOROLOGICAL DATA #######################	
	date, temp, cloud, epai = meteo() # FORMAT: NOW AND IN 15 MINUTES
#	print(epai)
	day = list(map(int, (date[0].split('T')[0]).split("-")))
	hour = list(map(int, (date[0].split('T')[1]).split(":")))


        ########## SUN AZIMUT AND ELEVATION #########################
        # year, month, day, hour, minute, second, timezone
	when = (day[0], day[1], day[2], hour[0], hour[1], hour[2], -3)
        # Get the Sun's apparent location in the sky
	az = sunpos(when, location, True)[0]
	el = sunpos(when, location, True)[1]
#	print(az, el)

	######### PLATFORM PUBLICATION ###########################
	payload="{"
	payload+="\"Temp. interior\":"+str(temperatura)+",";
	payload+="\"humedad interior\":"+str(humedad)+",";
	payload+="\"Temp. exterior\":"+str(temp[0])+",";
	payload+="\"Nubosidad\":"+str(cloud[0])+",";
	payload+="\"EPA index (calidad del aire)\":"+str(epai[0])+",";
	payload+="\"Azimuth\":"+str(az)+",";
	payload+="\"Elevacion\":"+str(el);
	payload+="}"
	ret= client1.publish("v1/devices/me/telemetry", payload) #topic-v1/devices/me/telemetry
	print("Please check LATEST TELEMETRY field of your device")

	time.sleep(120) #900 seg = 15 minutos
