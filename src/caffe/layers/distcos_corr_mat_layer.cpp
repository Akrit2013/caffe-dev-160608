#include <vector>
#include <cfloat>

#include "caffe/layers/distcos_corr_mat_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void DistCosCorrMatLayer<Dtype>::LayerSetUp(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
	// The superpixel pooled layer should have 1 width
	CHECK_EQ(bottom[0]->width(), 1);
	
	// Parse the params
	DistCosCorrMatParameter dist_corr_mat_param = this->layer_param_.distcos_corr_mat_param();
	// NOTE: The num_output_ must be equal with the number of superpixels
	if (dist_corr_mat_param.has_normalize()){
		normalize_ = dist_corr_mat_param.normalize();
	}else{
		normalize_ = false;
	}
}

template <typename Dtype>
void DistCosCorrMatLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	// Reshape the top
	top[0]->Reshape(bottom[0]->num(), 1, bottom[0]->height(), bottom[0]->height());
}

template <typename Dtype>
void DistCosCorrMatLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	const Dtype* bottom_data = bottom[0]->cpu_data();
	Dtype* top_data = top[0]->mutable_cpu_data();

	caffe_set(top[0]->count(), Dtype(0), top_data);

	const int stride = bottom[0]->count(2);
	const int channels = bottom[0]->channels();

	// Iter the superpixels
	for (int n = 0; n < bottom[0]->num(); n++){
		for (int h = 1; h < bottom[0]->height(); h++){
			for (int h2 = 0; h2 < h; h2++){
				const Dtype* data1 = bottom_data + bottom[0]->offset(n, 0, h);
				const Dtype* data2 = bottom_data + bottom[0]->offset(n, 0, h2);

				Dtype dst = CalcCosDistance(data1, data2, stride, channels);
				top_data[top[0]->offset(n, 0, h, h2)] = dst;
				top_data[top[0]->offset(n, 0, h2, h)] = dst;
			}
		}
	}

}


template <typename Dtype>
Dtype DistCosCorrMatLayer<Dtype>::L2Norm(const int count, const int stride, const Dtype* data){
	return sqrt(caffe_cpu_strided_dot(count, data, stride, data, stride));
}

template <typename Dtype>
Dtype DistCosCorrMatLayer<Dtype>::CalcCosDistance(const Dtype* data1, const Dtype* data2, const int stride, const int len){
	// Calc the dot between two data vectors
	const Dtype dot =caffe_cpu_strided_dot(len, data1, stride, data2, stride);

	const Dtype data1_norm = L2Norm(len, stride, data1);
	const Dtype data2_norm = L2Norm(len, stride, data2);

	Dtype ang;
	// When the data1_norm or data2_norm is 0
	if (data1_norm == 0 || data2_norm == 0){
		ang = 3.14 / 2;
	}else{
		// Calc the radian angle between two feature vector
		Dtype val = min(max(dot / data1_norm / data2_norm, Dtype(-1)), Dtype(1.0));
		ang = acos(val);
		if (isnan(ang)){
			LOG(INFO) << "VAL: " << val <<"angle: "<<ang;
		}
	}

	// Normalize the h and w first if needed
	if (normalize_){
		ang = min(ang / Dtype(3.1416), Dtype(1.0));
	}

	return ang;
}

INSTANTIATE_CLASS(DistCosCorrMatLayer);
REGISTER_LAYER_CLASS(DistCosCorrMat);

}  // namespace caffe
