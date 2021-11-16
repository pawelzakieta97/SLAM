import datetime
import os
start = datetime.datetime.now()
with open('test_file', 'wb') as f:
    f.truncate(1024*1024*1024*10)
print(datetime.datetime.now()-start)
os.remove('test_file')