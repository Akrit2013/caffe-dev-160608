// This new layer is used in fcn model, it will ignore the 'bad pixel'
// in the loss calculation.
// Define the BAD Pixel:
// If the pixel is smaller than 0 or close to 0, it will be considered as 'bad pixel'

#include <vector>

#include "caffe/util/math_functions.hpp"
#include "caffe/layers/euclideanfcn_loss_layer.hpp"

namespace caffe {

template <typename Dtype>
void EuclideanfcnLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  diff_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void EuclideanfcnLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  caffe_sub(
      count,
      bottom[0]->cpu_data(),
      bottom[1]->cpu_data(),
      diff_.mutable_cpu_data());
  // Set the diff to zero if label pixel is zero (all channel is zero)
  // Iter the diff map
  Dtype* data_diff = diff_.mutable_cpu_data();
  // const Dtype* data_pred = bottom[0]->cpu_data();
  int bad_pixel_count = 0;
  const Dtype* data_label = bottom[1]->cpu_data();
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
				  // This pixel is not ok
				  bad_pixel_count += diff_.channels();
				  for(int c = 0; c < diff_.channels(); ++c){
					  int index = ((n*diff_.channels()+c)*diff_.height()+h)*diff_.width()+w;
					  data_diff[index] = 0;
				  }
			  }
		  }
	 }
  }

  Dtype dot = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
  Dtype loss = dot / Dtype(2) / (count - bad_pixel_count);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void EuclideanfcnLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
      caffe_cpu_axpby(
          bottom[i]->count(),              // count
          alpha,                              // alpha
          diff_.cpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_cpu_diff());  // b
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(EuclideanfcnLossLayer);
#endif

INSTANTIATE_CLASS(EuclideanfcnLossLayer);
REGISTER_LAYER_CLASS(EuclideanfcnLoss);

}  // namespace caffe
