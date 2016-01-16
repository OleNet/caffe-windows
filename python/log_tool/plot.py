# -*- coding: utf-8 -*- 
#
# Author : fcbruce <fcbruce8964@gmail.com>
#
# Time : Mon 07 Dec 2015 02:16:27 PM CST
#
#

import inspect
import random
import sys
import matplotlib.cm as cmx
import matplotlib.colors as colors
import matplotlib.pyplot as plot
import matplotlib.legend as lgd
import matplotlib.markers as mks

index = {'train':['Iters', 'Seconds', 'LearningRate', 'accuracy', 'loss'], 
        'test':['Iters', 'Seconds', 'LearningRate', 'accuracy', 'loss']}

def load_data(data_file):
    data = []
    first = True
    with open(data_file, 'r') as f:
        for line in f:
            line = line.strip()
            fields = line.split(',')
            i = 0
            for ele in fields:
                if first:
                    data.append([])
                else:
                    data[i].append(float(ele.strip()))
                    i += 1
            first = False

    return data



def plot_chart(log_path, log_type):
    data = load_data(log_path + '.' + log_type)
    linewidth = 0.75
    for i in range(2, len(data)):
        label = index[log_type][i]
        color = [random.random(), random.random(), random.random()]
        plot.plot(data[0], data[i], label = label, color = color,
                linewidth = linewidth)
        plot.legend(loc = 'upper right', ncol = 1)

    plot.title(log_type)
    plot.xlabel(index[log_type][0])
    plot.savefig(log_path + '_' + log_type + '.png')



if __name__ == '__main__':
    if len(sys.argv) != 4:
        exit
    plot_chart(sys.argv[1], sys.argv[2])
