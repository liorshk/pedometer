import csv

NUM_OF_FILES = 30
NUM_OF_SEC_BEFORE = 11.2
NUM_OF_SEC_AFTER = 0.2

def calculate_num_of_sumpling_per_sec(rows,col_of_time):
    mid = len(rows)/2
    counter =0
    sample = rows[mid][col_of_time].split('.')[0]
    for i in range (1,len(rows)):
        if rows[i][col_of_time].split('.')[0]==sample:
            counter = counter +1
    return counter
        
    

def mean_function(rows, column,limit):
    summ = 0
    numm = 0
    for i in range(1,len(rows)):
        if float(rows[i][column])>0 and float(rows[i][column]) < limit:
            summ = summ + float(rows[i][column])
            numm = numm + 1
    if numm != 0:
        mean = summ / numm
    else:
        mean = 0

    return mean

def standard_deviation_function(rows,colums,mean,devide_param):
    summ = 0
    tmp = 0
    var = 0
    counter = 0
    for i in range(1,len(rows)):
        tmp = float(rows[i][colums])-mean
        tmp = tmp*tmp
        summ = summ + tmp
        counter = counter + 1
    var = summ / counter
    s_t =  mean +  (var**(0.5)/devide_param)
    return s_t

def define_peaks(rows,range_to_check,column,s_t):
    for i in range(range_to_check+1,len(rows)-range_to_check):
        counter = 0
        if float(rows[i][column])>s_t:
            for k in range (i-range_to_check,i+range_to_check):
                if float(rows[k][column])<float(rows[i][column]):
                    counter = counter + 1
            if counter >=range_to_check*2-2:
                rows[i][0] = 1
    
def return_end_of_walking_index(rows,len_of_peace,max_space_between_peaks,starting_index,cur_max,curr_end):
    start_index = starting_index
    end_index = starting_index + len_of_peace
    counter = 0
    last_peak = starting_index

    if len(rows)< starting_index + len_of_peace:
        #print "heeerreee" + str(curr_end)
        #print 'max - herer - ' + str(cur_max)
        return curr_end
    
    for i in range (starting_index,starting_index + len_of_peace+1):
        counter = counter + rows[i][0]
        if rows[i][0]==1:
            #print i
            if (i - last_peak) > max_space_between_peaks:
                #print str(i) + ' split'
                return return_end_of_walking_index(rows,len_of_peace,max_space_between_peaks,i-1,counter,last_peak)  #was instead of cur_end - last peak
            last_peak=i

    if counter > cur_max:
        cur_max = counter

    #print cur_max
    #print last_peak

    for i in range(starting_index+len_of_peace+2,len(rows)):
        if rows[i][0]==1:
            if (i - last_peak) > max_space_between_peaks:
                #print str(i) + 'split2, curr_end = ' + str(end_index)
                #print 'max = ' + str(cur_max)
                return return_end_of_walking_index(rows,len_of_peace,max_space_between_peaks,i-1,cur_max,end_index)
            last_peak = i
        
        counter = counter - rows[i-len_of_peace][0] + rows[i][0]
        if counter > cur_max:
            #print i
            cur_max = counter
            end_index = i
            start_index = i-len_of_peace

    return end_index



def filter_scv(filenameToRead,filenameToWrite):

    i = 0
    rows = []
    with open(filenameToRead, 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            row.insert(0,0)
            rows.append(row)
            i=i+1


    j=4         # colums to check - I choose Y
    #ideal rangeC=3
    rangeC = 3  # number of left and right min values to find a peak

    samples_per_sec = calculate_num_of_sumpling_per_sec(rows,2)    
    mean = mean_function(rows,j,100)
    #mean = mean_function(rows,j,mean*3)
    # ideal = 4
    s_t = standard_deviation_function(rows,j,mean,1)

    #print "s_t = " + str(s_t)
    
    if mean + 3 <= s_t:
        s_t = standard_deviation_function(rows,j,mean,4)
    define_peaks(rows,rangeC,j,s_t)

    

    #for i in range (1,len(rows)):
    #    if rows[i][0]==1:
    #        print i

    #print mean
    #print s_t


    #between peacks ideal = 90
    index = return_end_of_walking_index(rows,samples_per_sec*10,130,0,0,501)
    #print filenameToRead + ': start at: ' + str(index - (samples_per_sec*NUM_OF_SEC_BEFORE)) + ' finish at: ' + str(index + samples_per_sec*NUM_OF_SEC_AFTER)
    #print 'mean: ' + str(mean) + ' stiat_teken: ' + str(s_t)


    with open(filenameToWrite, 'wb') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        data = []
        header = ['','epoch','load.txt.data.x','load.txt.data.y','load.txt.data.z']
        data.append(header)
        for i in range(int(index - (samples_per_sec*NUM_OF_SEC_BEFORE)), int(index + samples_per_sec* NUM_OF_SEC_AFTER)):
            tmp_list = [rows[i][1],rows[i][2],rows[i][3],rows[i][4],rows[i][5]]
            data.append(tmp_list)
        spamwriter.writerows(data)




def main_function():
    list_of_paths_origin = []
    list_of_paths_target = []
    
    for i in range(1,NUM_OF_FILES+1):
        tmp_str = 'Train_' + str(i) + '.csv'
        tmp2_str = 'Train_' + str(i) + '_filtered.csv'
        list_of_paths_origin.append(tmp_str)
        list_of_paths_target.append(tmp2_str)

    for i in range (0,NUM_OF_FILES):
        filter_scv(list_of_paths_origin[i],list_of_paths_target[i])
    


#filter_scv('Train_2.csv')
