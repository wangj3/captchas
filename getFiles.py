import requests
import time
start = time.time()

BASE_URL = "https://cs7ns1.scss.tcd.ie/2122/wangj3/pi-project2/wangj3-challenge-filenames.csv"
auth_params = { "shortname": "wangj3" }

r = requests.get(BASE_URL, params=auth_params)
if r.status_code != 200:
    print("error getting file manifest url")
    print(r)
    
contents = r.content.decode('utf-8')
f =open("filename.txt","w")
f.write(contents)
f.close()
end = time.time()
print( '{0} seconds'.format((end - start)))
