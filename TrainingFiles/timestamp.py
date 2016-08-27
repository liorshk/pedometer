
import csv
import time
import datetime

def timestemp(filenameToRead,filenameToWrite):
    i = 0
    rows = []
    a=[]
    with open(filenameToRead, 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            row.insert(0,0)
            rows.append(row)
            i=i+1


    for i in range(1,len(rows)-1):
        #date = rows[i][2].split(' ')[0]
        #arr = date.split('-')
        #new_date = arr[1] + '/' + arr[2] + '/' + arr[0]
        #print new_date
        #time = rows[i][2].split(' ')[1]
        #int(datetime.datetime.strptime('01/12/2011', '%d/%m/%Y').strftime("%s"))
        #datetime.datetime.strptime(rows[i][2], "%Y-%m-%d %H:%M:%S.%f").timetuple()
        #time.struct_time(tm_year=2014, tm_mon=8, tm_mday=1, tm_hour=4, tm_min=41, tm_sec=52, tm_wday=4, tm_yday=213, tm_isdst=-1)
        dt = datetime.datetime.strptime(rows[i][2], "%Y-%m-%d %H:%M:%S.%f")
        a.append(time.mktime(dt.timetuple()) + (dt.microsecond / 1000))

    return a   

def main_function():
    list_of_paths_origin = []
    list_of_paths_target = []
    b = []
    for i in range(1,31):
        tmp_str = 'Train_' + str(i) + '.csv'
        tmp2_str = 'Train_' + str(i) + '_filtered.csv'
        list_of_paths_origin.append(tmp_str)
        list_of_paths_target.append(tmp2_str)

    for i in range (0,30):
        a = timestemp(list_of_paths_target[i],list_of_paths_target[i])
        b.append(a)
    return b

