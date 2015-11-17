
#include "caffe/caffe.hpp"
#include "imcl_cnn/imcl_cnn.h"
#include "opencv/cv.h"
#include "opencv/highgui.h"


int main()
{
	double detail[20];
	int label;

	const char* modelPath = "D:\\STUDY\\[0] MachineLearning\\caffe-windows\\examples\\mnist\\lenet_solver.prototxt";
	IMCL_CNNMODEL* model;
	//(modelPath);
	IMCL_CnnLoadModel(modelPath, &model);
	IplImage* img = cvLoadImage("cat.jpg");
	IMCL_CnnPredict(img, &label, detail);
	IMCL_CnnDestropyModel(model);

	return 0;
}