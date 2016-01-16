import conf_pb2
import sys
from google.protobuf import text_format

def convert(oldlist, newfilename, flat):
    appconfig = conf_pb2.AppConfig()
    file_list = appconfig.filelist
    #file_list = conf_pb2.FileList()
    file_list.root = ""

    with open(oldlist, 'r') as oldf:
        lines = oldf.readlines()

    for line in lines:
        sp = line.split(' ')
        relpath = sp[0]
        value = sp[1]

        file = file_list.file.add()
        file.name = relpath
        if flag == 'label':
            file.gold_label = float(value)
        if flag == 'score':
            file.gold_score = float(value)

    f = open(newfilename, "w")
    f.write(text_format.MessageToString(appconfig))
    f.close()
    

if __name__ == '__main__':
    print "USAGE:"
    print "DEFAULT: fill the value to score"
    print "python oldlist2proto.py path/to/oldlist.txt newname"
    print "OR"
    print "python oldlist2proto.py path/to/oldlist.txt newname label"
    nargs = len(sys.argv)
    if  nargs < 3:
        exit(0)
    if nargs == 3:
        flag = 'score'
    else:
        flag = sys.argv[3]

    convert(sys.argv[1], sys.argv[2], flag)