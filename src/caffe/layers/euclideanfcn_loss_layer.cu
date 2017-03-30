#include <vector>

#include "caffe/util/math_functions.hpp"
#include "caffe/layers/euclideanfcn_loss_layer.hpp"

namespace caffe {

template <typename Dtype>
void EuclideanfcnLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  caffe_gpu_sub(
      count,
      bottom[0]->gpu_data(),
      bottom[1]->gpu_data(),
      diff_.mutable_gpu_data());
  Dtype* data_diff = diff_.mutable_cpu_data();
  // const Dtype* data_pred = bottom[0]->cpu_data();
  const Dtype* data_label = bottom[1]->cpu_data();
  int bad_pixel_count = 0;
  for(int n = 0; n < diff_.num(); ++n){
	 for(int h = 0; h < diff_.height(); ++h){
		  for(int w = 0; w < diff_.width(); ++w){
			  bool valid_pixel = false;
			  for(int c = 0; c < diff_.channels(); ++c){
				  int index = ((n*diff_.channels()+c)*diff_.height()+h)*diff_.width()+w;
				  Dtype dataval = data_label[index];
				  if(dataval >= DEF_PIXEL_MIN){
					  valid_pixel = true;
				  }
			  }
			  if(!valid_pixel){
				  bad_pixel_count++;
				  // This pixel is not ok
				  for(int c = 0; c < diff_.channels(); ++c){
					  int index = ((n*diff_.channels()+c)*diff_.height()+h)*diff_.width()+w;
					  data_diff[index] = 0;
				  }
			  }
		  }
	 }
  }

  Dtype dot;
  caffe_gpu_dot(count, diff_.gpu_data(), diff_.gpu_data(), &dot);
  Dtype loss = dot / Dtype(2) / (count-bad_pixel_count);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void EuclideanfcnLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
      caffe_gpu_axpby(
          bottom[i]->count(),              // count
          alpha,                              // alpha
          diff_.gpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_gpu_diff());  // b
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(EuclideanfcnLossLayer);

}  // namespace caffe
