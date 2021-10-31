import requests
import shutil
import time
start = time.time()

BASE_URL = "https://cs7ns1.scss.tcd.ie/2122/wangj3/pi-project2/"

# Using readline(
file1 = open('filename.txt', 'r')
count = 0
 
while True:
    count += 1
 
    # Get next line from file
    line = file1.readline()
    filename= line.replace(',','').strip()
    # if line is empty
    # end of file is reached
    if not line:
        break
    print("filename {}: {}".format( filename,count))
    auth_params = { "shortname": "wangj3" }
    r = requests.get(BASE_URL+filename, params=auth_params,stream=True)
    if r.status_code == 200:
        r.raw.decode_content = True
        
        with open('img/'+filename,'wb') as f:
            shutil.copyfileobj(r.raw, f)

        print('Image sucessfully Downloaded: ',filename)
    else:
        print('Image Couldn\'t be retreived')
file1.close()
end = time.time()
print( '{0} seconds'.format((end - start)))

def downloadImage(filename):
    auth_params = { "shortname": "wangj3", "myfilename": filename }
    r = requests.get(BASE_URL, params=auth_params,stream=True)
    if r.status_code == 200:
        # Set decode_content value to True, otherwise the downloaded image file's size will be zero.
        r.raw.decode_content = True
    
        # Open a local file with wb ( write binary ) permission.
        with open('img/'+filename,'wb') as f:
            shutil.copyfileobj(r.raw, f)
        
        print('Image sucessfully Downloaded: ',filename)
    else:
        print('Image Couldn\'t be retreived')
