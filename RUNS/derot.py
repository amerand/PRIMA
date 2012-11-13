import numpy

def tag2mjd(tag):
    """ take time tag ('YYYY-MM-DD hh:mm:ss.sss') and
        convert it to MJD
    """
    y = int(tag.split('-')[0])
    m = int(tag.split('-')[1])
    d = int(tag.split('-')[2].split(' ')[0])
    hh = float(tag.split('-')[2].split(' ')[1].split(':')[0])
    mm = float(tag.split('-')[2].split(' ')[1].split(':')[1])
    ss = float(tag.split('-')[2].split(' ')[1].split(':')[2])
    a = (14-m)/12
    y += 4800 - a
    m += 12*a - 3
    mjd = d + (153*m+2)/5 + 365*y +y/4 -y/100 +y/400 - 32045
    mjd += hh/24. + mm/(24*60.) + ss/(24*3600.0)
    mjd = mjd-2400001.0
    return mjd

def readFile(filename):
    """
    read report files from ATs
    """
    f = open(filename, 'rU')
    l = f.readline()
    var_names = []
    while l[0] == '#':
        l = f.readline()
        # works only for 6 or less variables
        if l[1]==' ' and l[2].isdigit():
            if int(int(l.split()[1])>3):
                var_names.append(l.split()[2])
    n_vars = len(var_names)
    data = []
    time = []

    while not l[0]=='#':
        time.append(tag2mjd(l.split()[2]+' '+l.split()[3]))
        data.append(l.split()[4:])
        l = f.readline()
    data = numpy.array(data)
    res = {}
    res['mjd'] = time
    print data.size
    for k,var in enumerate(var_names):
        print k, var
        res[var]=numpy.float_(data[:,k])
    data=[]
    f.close()
    return res
