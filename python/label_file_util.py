from google.protobuf import text_format
import conf_pb2
import os

def read_proto_file(file_path, parser_object):
    file = open(file_path, "r")
    text_format.Merge(str(file.read()), parser_object)
    file.close()
    return parser_object


def get_name_score_from_proto(file_path):
    app_config = conf_pb2.AppConfig()
    read_proto_file(file_path, app_config)
    name = []
    score = []
    for f in app_config.filelist.file:
        name.append(f.name)
        score.append(f.gold_score)
    return name, score


def load_old_format_labelscore(file_path):
    with open(file_path, 'r') as T:
        lines = T.readlines()
    img_path = []
    score = []
    for line in lines:
        sp = line.split(' ')
        img_path.append(sp[0])
        score.append(float(sp[1]))
    return img_path, score


def load_gt_sep(LABEL_FILE):
    if os.path.splitext(LABEL_FILE)[1] == '.txt':
        file_list, score_list = load_old_format_labelscore(LABEL_FILE)
    if os.path.splitext(LABEL_FILE)[1] == '.proto':
        file_list, score_list = get_name_score_from_proto(LABEL_FILE)
    return file_list, score_list


def load_gt_dict(LABEL_FILE, split_name=False):
    if os.path.splitext(LABEL_FILE)[1] == '.txt':
        file_list, score_list = load_old_format_labelscore(LABEL_FILE)
    if os.path.splitext(LABEL_FILE)[1] == '.proto':
        file_list, score_list = get_name_score_from_proto(LABEL_FILE)

    if split_name:
        for i, name in enumerate(file_list):
            file_list[i] = os.path.split(name)[-1]
    gt_dict = {key: value for (key, value) in zip(file_list, score_list)}
    return gt_dict


'''def load_groundtruth(filepath):

    with open(filepath, 'r') as f:
        lines = f.readlines()

    score_record = {}
    for i, line in enumerate(lines):
        sp = str.split(line, ' ')
        name = os.path.split(sp[0])[-1]
        score = float(sp[1])
        score_record[name] = score
    return score_record'''