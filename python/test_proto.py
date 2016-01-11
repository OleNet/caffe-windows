import caffe
import conf_pb2
from google.protobuf import text_format

def readProtoFile(filepath, parser_object):
    file = open(filepath, "r")
    text_format.Merge(str(file.read()), parser_object)
    file.close()
    return parser_object

if __name__ == '__main__':
    app_config = conf_pb2.AppConfig()
    readProtoFile('D:/Project/caffe-windows-master/data/Blur1000/test.proto', app_config)
    name = []
    score = []
    for file in app_config.filelist.file:
        name.append(file.name)
        score.append(file.gold_score)

