#include <vector>
#include "caffe/caffe.hpp"
#include "IMCL_cnn/cnn_predict.h"

using namespace std;
using namespace caffe;

void* IMCL_CnnLoadModel(const char* modelPath, IMCL_CNNMODEL** cnn_model)
{
	//Setting CPU or GPU
	//if (argc >= 5 && strcmp(argv[4], "GPU") == 0) {
		//Caffe::set_mode(Caffe::GPU);
		int device_id = 0;
		//if (argc == 6) {
		//	device_id = atoi(argv[5]);
		//}
		//Caffe::SetDevice(device_id);
		//LOG(ERROR) << "Using GPU #" << device_id;
	//}
	//else {
		LOG(ERROR) << "Using CPU";
		Caffe::set_mode(Caffe::CPU);
	//}

	//get the net
	*cnn_model = new IMCL_CNNMODEL(modelPath);
	//get trained net
	const char* net_proto = "";
	(*cnn_model)->CopyTrainedLayersFrom(net_proto);
	return cnn_model;

}

int IMCL_CnnPredict(const IMCL_CNNMODEL* model, const IplImage* img, int* label, double* detail)
{
	//get datum
	Datum datum;
	if (!ReadImageToDatum("./cat.png", 1, 227, 227, &datum)) {
	//	LOG(ERROR) << "Error during file reading";
	//}
	//get the blob
	Blob<float>* blob = new Blob<float>(1, datum.channels(), datum.height(), datum.width());

	//get the blobproto
	BlobProto blob_proto;
	blob_proto.set_num(1);
	blob_proto.set_channels(datum.channels());
	blob_proto.set_height(datum.height());
	blob_proto.set_width(datum.width());
	const int data_size = datum.channels() * datum.height() * datum.width();
	int size_in_datum = std::max<int>(datum.data().size(),
		datum.float_data_size());
	for (int i = 0; i < size_in_datum; ++i) {
		blob_proto.add_data(0.);
	}
	const string& data = datum.data();
	if (data.size() != 0) {
		for (int i = 0; i < size_in_datum; ++i) {
			blob_proto.set_data(i, blob_proto.data(i) + (uint8_t)data[i]);
		}
	}

	//set data into blob
	blob->FromProto(blob_proto);

	//fill the vector
	vector<Blob<float>*> bottom;
	bottom.push_back(blob);
	float type = 0.0;

	const vector<Blob<float>*>& result = model->Forward(bottom, &type);

	//Here I can use the argmax layer, but for now I do a simple for :)
	float max = 0;
	float max_i = 0;
	for (int i = 0; i < 1000; ++i) {
		float value = result[0]->cpu_data()[i];
		if (max < value){
			max = value;
			max_i = i;
		}
	}
	LOG(ERROR) << "max: " << max << " i " << max_i;
}

int IMCL_CnnDestropyModel(IMCL_CNNMODEL* cnn_model)
{

}