#ifndef __IMCL_CNN_H__
#define __IMCL_CNN_H__


#ifndef DL_EXTERN_C
#ifdef __cplusplus
#define DL_EXTERN_C extern "C"
#else
#define DL_EXTERN_C
#endif
#endif


#ifdef IMCL_CNN_EXPORTS
#define DL_SDK(rettype)  DL_EXTERN_C  __declspec(dllexport) rettype 
#else
#define DL_SDK(rettype)  DL_EXTERN_C  __declspec(dllimport) rettype 
#endif


#ifndef MAX_FILEPATH
#define MAX_FILEPATH			(256)
#endif


struct _IplImage;
typedef _IplImage IplImage;
typedef caffe::Net<float> IMCL_CNNMODEL;

DL_SDK(int) IMCL_CnnLoadModel(const char* modelPath, IMCL_CNNMODEL** model);
DL_SDK(int) IMCL_CnnPredict(const IplImage* img, int* label, double* detail);
DL_SDK(int) IMCL_CnnDestropyModel(IMCL_CNNMODEL* cnn_model);


#endif // __IMCL_CNN_H__
