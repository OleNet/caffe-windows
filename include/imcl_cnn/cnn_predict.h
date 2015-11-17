#ifndef __CNN_PREDICT_H__
#define __CNN_PREDICT_H__

#include <glog/logging.h>

#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "caffe/caffe.hpp"
#include "caffe/util/signal_handler.h"
#include "IMCL_cnn.h"

// typedef caffe::Net<float> IMCL_CNNMODEL;
// 
// DL_SDK(void*) IMCL_CnnLoadModel(const char* modelPath);
// 
// DL_SDK(int) IMCL_CnnPredict(const IplImage* img, int* label, double* detail);
// 
// DL_SDK(int) IMCL_CnnDestropyModel(IMCL_CNNMODEL* cnn_model);


#endif //__CNN_PREDICT_H__
