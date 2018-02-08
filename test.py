import os
try:
    with open('./path_tmp', 'r') as text_file:
        path = text_file.read().replace('\n', '')
except:
    with open('./path_tmp', 'w') as text_file:
        print("~/", file=text_file)