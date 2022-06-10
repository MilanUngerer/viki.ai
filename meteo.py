import requests
import datetime
import numpy as np

def meteo():
	startt = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") #now
	endt = (datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=15)).strftime("%Y-%m-%dT%H:%M:%SZ") #+15 minutes

	url = "https://api.tomorrow.io/v4/timelines"

	querystring = {
	"location":"-33.05178823955611, -71.55734620441065", #RHONA
	"fields":["temperature", "cloudCover", "epaIndex"],
	"units":"metric",
	"timesteps":"15m",
	"timezone":"America/Santiago",
	"startTime":startt,
	"endTime":endt,
	"apikey":"luwuHX2FBlIkFLZhfXWKHzu241dGc2RU"}

	response = requests.request("GET", url, params=querystring)
#print(response.text)

	t = response.json()['data']['timelines'][0]['intervals'][0]['values']['temperature']
	cc = response.json()['data']['timelines'][0]['intervals'][0]['values']['cloudCover']
	airql = response.json()['data']['timelines'][0]['intervals'][0]['values']['epaIndex']


#print("Weather Forecast")
#print("================")

	results = response.json()['data']['timelines'][0]['intervals']
#print(results)
	date = [startt, endt]
	temp = np.zeros(2)
	cloud = np.zeros(2)
	epai = np.zeros(2)
	i = 0
	for daily_result in results:
		date[i] = daily_result['startTime'][0:19]
		temp[i] = round(daily_result['values']['temperature'])
		cloud[i] = round(daily_result['values']['cloudCover'])
		epai[i] = round(daily_result['values']['epaIndex'])
		i += 1
#		print("On",date,"it will be", temp, "°C and", cloud, "% of cloud cover")	
	return date, temp, cloud, epai
