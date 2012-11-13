import os
import numpy


def pssRecorder(filename):
    """
    at3 = oneFile(os.path.join(directory, 'pssguiRecorder_lat3fsm_'+date+'.dat'))
    at4 = oneFile(os.path.join(directory, 'pssguiRecorder_lat4fsm_'+date+'.dat'))
    time = numpy.linspace(max(at3['time'].min(),at4['time'].min()),
                          min(at3['time'].max(),at4['time'].max()),
                          len(at3['time']))
    """
    f = open(filename)
    lines = f.read().split('\n')
    f.close()
    var_names = ['index', 'time', 'date', 'time_stamp']
    k = 4
    for l in lines:
        if '# '+str(k)+' ' in l:
            tmp = l.split()[4:]
            tmp = reduce(lambda x, y: x+y, tmp)
            var_names.append(tmp)
            k+=1
        if 'start time:' in l:
            break        
    # set up        
    data = filter(lambda x: x[0]!='#', lines)
    rec = [[] for k in range(len(var_names))]
    for i in range(len(data)):
        for k in range(len(var_names)):
            try:
                rec[k].append(float(data[i].split()[k]))
            except:
                rec[k].append(data[i].split()[k])
    res = {}
    for k in range(len(var_names)):
        res[var_names[k]] = numpy.array(rec[k])
    return res


    

