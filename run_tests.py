import requests, time
time.sleep(1)
r = requests.get('http://localhost:8000/recommend?user_id=U001&k=3')
assert r.status_code==200 and len(r.json())==3
print('OK')