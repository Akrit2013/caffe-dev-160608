#include <vector>

#include "caffe/layers/superpixel_predict_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SuperpixelPredictLayer<Dtype>::LayerSetUp(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
	// This layer has no params for now
	// Check the blobs
	CHECK_EQ(bottom[0]->num(), bottom[1]->num());
	// Currently, the channels of the bottom[1] must be 1
	CHECK_EQ(bottom[1]->channels(), 1);

	// Check the max superpixel label, which is the superpixel number - 1
	// and the size of the bottom[0]
	// TODO: Since the output is the abs max value, there might be a problem
	// when the value is negative
	sp_num_ = bottom[0]->count(2);
}

template <typename Dtype>
void SuperpixelPredictLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	// Reshape the top
	top[0]->Reshape(bottom[0]->num(), bottom[0]->channels(), bottom[1]->height(), bottom[1]->width());
}

template <typename Dtype>
void SuperpixelPredictLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	const Dtype* sp_data = bottom[1]->cpu_data();
	const Dtype* pred_data = bottom[0]->cpu_data();
	Dtype* out_data = top[0]->mutable_cpu_data();

	// Clear the memory to zero
	caffe_set(top[0]->count(), Dtype(0), out_data);

	// Iter all pixels in the minibatch
	for (int n = 0; n < top[0]->num(); n++){
		const Dtype* sp_data_n = sp_data + bottom[1]->offset(n);
		for (int c = 0; c < top[0]->channels(); c++){
			const Dtype* pred_data_nc = pred_data + bottom[0]->offset(n, c);
			Dtype* out_data_nc = out_data + top[0]->offset(n, c);
			for (int i = 0; i < top[0]->count(2); i++){
				out_data_nc[i] = pred_data_nc[static_cast<int>(sp_data_n[i])];
			}
		}
	}
}


INSTANTIATE_CLASS(SuperpixelPredictLayer);
REGISTER_LAYER_CLASS(SuperpixelPredict);

}  // namespace caffe
