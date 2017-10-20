from time import gmtime, strftime

string = strftime("%m-%d %H:%M:%S", gmtime())


print(string)