import requests

url = 'http://localhost:3000/predict'

r = requests.post(url,json={'demo-age' : 38,
                            'demo-year': 1999,
                            'demo-axi' : 3
                            }
                 )
# print(r.json())
# print(r.json()[0]["name"])