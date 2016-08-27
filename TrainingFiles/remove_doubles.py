import csv

FIRST = 14
LAST = 26



def delete_doubles(filenameToRead,filenameToWrite):
    
    i = 0
    rows = []
    with open(filenameToRead, 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            row.insert(0,0)
            rows.append(row)
            i=i+1

    for i in range(0,len(rows)-2):
        if rows[i][3] == rows[i+1][3] and rows[i][4] == rows[i+1][4] and rows[i][5] == rows[i+1][5]:
            rows[i+1][0]=1

    
    with open(filenameToWrite, 'wb') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        data = []
        for i in range(0,len(rows)-1):
            if rows[i][0] != 1:
                tmp_list = [rows[i][1],rows[i][2],rows[i][3],rows[i][4],rows[i][5]]
                data.append(tmp_list)
        spamwriter.writerows(data)

            


def main_func():
    list_of_paths_origin = []
    list_of_paths_target = []
    for i in range(FIRST,LAST+1):
        tmp_str = 'Train_' + str(i) + '.csv'
        tmp2_str = 'Train_' + str(i) + '_doubled.csv'
        list_of_paths_origin.append(tmp_str)
        list_of_paths_target.append(tmp2_str)

    for i in range (0,LAST-FIRST+1):
        delete_doubles(list_of_paths_origin[i],list_of_paths_target[i])


main_func()
