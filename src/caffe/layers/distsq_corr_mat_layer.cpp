#include <vector>
#include <cfloat>

#include "caffe/layers/distsq_corr_mat_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void DistSqCorrMatLayer<Dtype>::LayerSetUp(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
	// The bottom[0] is a superpixel map, which channels is 1
	CHECK_EQ(bottom[0]->channels(), 1);
	// Check the format of the input layer
	CHECK_EQ(bottom[0]->width(), 2);
	
	// Parse the params
	DistCorrMatParameter dist_corr_mat_param = this->layer_param_.dist_corr_mat_param();
	// NOTE: The num_output_ must be equal with the number of superpixels
	if (dist_corr_mat_param.has_normalize()){
		normalize_ = dist_corr_mat_param.normalize();
	}else{
		normalize_ = false;
	}

	if (dist_corr_mat_param.has_width()){
		width_ = dist_corr_mat_param.width();
	}else{
		if (normalize_ == true){
			LOG(FATAL) << "When normalize set to true, the width must be set to normalize the distance between superpixels";
		}
		width_ = 1;
	}

	if (dist_corr_mat_param.has_height()){
		height_ = dist_corr_mat_param.height();
	}else{
		if (normalize_ == true){
			LOG(FATAL) << "When normalize set to true, the width must be set to normalize the distance between superpixels";
		}
		height_ = 1;
	}

}

template <typename Dtype>
void DistSqCorrMatLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	// Reshape the top
	top[0]->Reshape(bottom[0]->num(), 1, bottom[0]->height(), bottom[0]->height());
}

template <typename Dtype>
void DistSqCorrMatLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	const Dtype* bottom_data = bottom[0]->cpu_data();
	Dtype* top_data = top[0]->mutable_cpu_data();

	caffe_set(top[0]->count(), Dtype(0), top_data);
	// Iter the superpixels
	for (int n = 0; n < bottom[0]->num(); n++){
		for (int c = 0; c < bottom[0]->channels(); c++){
			for (int h = 1; h < bottom[0]->height(); h++){
				for (int h2 = 0; h2 < h; h2++){
					const Dtype* pCentroid1 = bottom_data + bottom[0]->offset(n, c, h);
					const Dtype* pCentroid2 = bottom_data + bottom[0]->offset(n, c, h2);
					Dtype dst = CalcDistanceSquare(pCentroid1, pCentroid2);
					top_data[top[0]->offset(n, c, h, h2)] = dst;
					top_data[top[0]->offset(n, c, h2, h)] = dst;
				}
			}
		}
	}

}

template <typename Dtype>
Dtype DistSqCorrMatLayer<Dtype>::CalcDistanceSquare(const Dtype* pCentroid1, const Dtype* pCentroid2){
	const Dtype h1 = pCentroid1[0];
	const Dtype w1 = pCentroid1[1];
	const Dtype h2 = pCentroid2[0];
	const Dtype w2 = pCentroid2[1];

	Dtype diff_h = h1 - h2;
	Dtype diff_w = w1 - w2;

	// Normalize the h and w first if needed
	if (normalize_){
		diff_h = diff_h / Dtype(height_);
		diff_w = diff_w / Dtype(width_);
	}

	Dtype dist = diff_h*diff_h + diff_w*diff_w;
	return dist;
}

INSTANTIATE_CLASS(DistSqCorrMatLayer);
REGISTER_LAYER_CLASS(DistSqCorrMat);

}  // namespace caffe
