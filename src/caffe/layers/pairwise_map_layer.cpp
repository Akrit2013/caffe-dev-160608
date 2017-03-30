#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/pairwise_map_layer.hpp"
#include "caffe/util/math_functions.hpp"

#include <math.h>

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void PairwiseMapLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Init the param
  PairwiseMapParameter pairwise_param = this->layer_param_.pairwise_map_param();

  if (pairwise_param.has_theta1()){
	  theta1_ = pairwise_param.theta1();
  }else{
	  theta1_ = 1;
  }

  if (pairwise_param.has_theta2()){
	  theta2_ = pairwise_param.theta2();
  }else{
	  theta2_ = 1;
  }

  if (pairwise_param.has_theta3()){
	  theta3_ = pairwise_param.theta3();
  }else{
	  theta3_ = 1;
  }

  if (pairwise_param.has_w1()){
	  w1_ = pairwise_param.w1();
  }else{
	  w1_ = 1;
  }

  if (pairwise_param.has_w2()){
	  w2_ = pairwise_param.w2();
  }else{
	  w2_ = 1;
  }

  if (pairwise_param.has_height()){
	  height_ = pairwise_param.height();
  }else{
	  LOG(FATAL) << "The height of the superpixel map must be set before calc the pairwise map";
  }

  if (pairwise_param.has_width()){
	  width_ = pairwise_param.width();
  }else{
	  LOG(FATAL) << "The width of the superpixel map must be set before calc the pairwise map";
  }
}

template <typename Dtype>
void PairwiseMapLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	top[0]->Reshape(bottom[0]->num(), 1, bottom[0]->count(2), bottom[0]->count(2));
}

template <typename Dtype>
void PairwiseMapLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	const Dtype* bottom_data = bottom[0]->cpu_data();
	Dtype* top_data = top[0]->mutable_cpu_data();

	const int num = top[0]->num();
	const int channels = bottom[0]->channels();
	const int sp_num = bottom[0]->count(2);

	// Clear the top data
	caffe_set(top[0]->count(), Dtype(0), top_data);

	// Iter the top data
	for (int n = 0; n < num; n++){
		for (int h = 0; h < top[0]->height(); h++){
			for (int w = 0; w < h; w++){
				const Dtype* data1 = bottom_data + bottom[0]->offset(n, 0, h);
				const Dtype* data2 = bottom_data + bottom[0]->offset(n, 0, w);
				int idx = top[0]->offset(n, 0, h, w);
				int idx2 = top[0]->offset(n, 0, w, h);

				// Calc the norm of the feature vectors
				Dtype data1_norm = L2Norm(channels, sp_num, data1);
				Dtype data2_norm = L2Norm(channels, sp_num, data2);
				// Calc the dot of the data1 and data2
				Dtype val = caffe_cpu_strided_dot(channels, data1, sp_num, data2, sp_num);
				Dtype ang;
				// When the data1_norm or data2_norm is 0
				if (data1_norm == 0 || data2_norm == 0){
					ang = 3.14 / 2;
				}else{
					// Calc the radian angle between two feature vector
					val = min(max(val / data1_norm / data2_norm, Dtype(-1)), Dtype(1.0));
					ang = acos(val);
					if (isnan(ang)){
						LOG(INFO) << "VAL: " << val <<"angle: "<<ang;
					}
				}

				// Calc the L2 distance of between these two pixels
				Dtype dist = CalcDistance2(h, w);

				top_data[idx] = -ang/theta1_ - dist/theta2_;
				top_data[idx2] = -ang/theta1_ - dist/theta2_;
			}
		}
	}
	// Calc the log
	caffe_exp(top[0]->count(), top_data, top_data);
	// Rescale the data
	caffe_scal(top[0]->count(), Dtype(w1_), top_data);
}

template <typename Dtype>
Dtype PairwiseMapLayer<Dtype>::CalcDistance2(const int index1, const int index2){
	// Calc the coordinate according to the height and width
	int h1 = index1 / width_;
	int h2 = index2 / width_;
	int w1 = index1 % width_;
	int w2 = index2 % width_;

	// Normalize the h and w between 0 and 1
	Dtype h1n = Dtype(h1) / Dtype(height_);
	Dtype h2n = Dtype(h2) / Dtype(height_);
	Dtype w1n = Dtype(w1) / Dtype(width_);
	Dtype w2n = Dtype(w2) / Dtype(width_);

	// Calc the distance
	Dtype dh = h1n - h2n;
	Dtype dw = w1n - w2n;

	return dh*dh + dw*dw;
}

template <typename Dtype>
Dtype PairwiseMapLayer<Dtype>::L2Norm(const int count, const int stride, const Dtype* data){
	return sqrt(caffe_cpu_strided_dot(count, data, stride, data, stride));
}


#ifdef CPU_ONLY
STUB_GPU(PairwiseMapLayer);
#endif

INSTANTIATE_CLASS(PairwiseMapLayer);
REGISTER_LAYER_CLASS(PairwiseMap);


}  // namespace caffe
