import requests

def get_coordinates(location):
    url = "https://nominatim.openstreetmap.org/search?"
    params = {"q": location, "format": "json"}
    response = requests.get(url, params=params).json()[0]
    print(response)
    latitude = response["lat"]
    longitude = response["lon"]

    return latitude, longitude
