// TODO (sergeyk): effect should not be dependent on phase. wasted memcpy.

#include <vector>

#include "caffe/layers/conv_dropout_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ConvDropoutLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
  threshold_ = this->layer_param_.dropout_param().dropout_ratio();
  DCHECK(threshold_ > 0.);
  DCHECK(threshold_ < 1.);
  scale_ = 1. / (1. - threshold_);
  uint_thres_ = static_cast<unsigned int>(UINT_MAX * threshold_);
}

template <typename Dtype>
void ConvDropoutLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::Reshape(bottom, top);
  // Set up the cache for random number generation
  // ReshapeLike does not work because rand_vec_ is of Dtype uint
  rand_vec_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
}

template <typename Dtype>
void ConvDropoutLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  unsigned int* mask = rand_vec_.mutable_cpu_data();
  // const int count = bottom[0]->count();
  const int mask_count = rand_vec_.count();
  const int feature_map_size = bottom[0]->height() * bottom[0]->width();
  if (this->phase_ == TRAIN) {
    // Create random numbers
    caffe_rng_bernoulli(mask_count, 1. - threshold_, mask);
    for (int i = 0; i < mask_count; ++i) {
	  Dtype* top_feature_data = top_data + i*feature_map_size;
	  const Dtype* bottom_feature_data = bottom_data + i*feature_map_size;

	  // Set the feature map data
	  // caffe_scal(feature_map_size, mask[i]*scale_, top_feature_data);
	  caffe_cpu_scale(feature_map_size, mask[i]*scale_, bottom_feature_data, top_feature_data);
	  // caffe_cpu_axpby(feature_map_size, mask[i]*scale_, bottom_feature_data, Dtype(0), top_feature_data);
    }
  } else {
    caffe_copy(bottom[0]->count(), bottom_data, top_data);
  }
}

template <typename Dtype>
void ConvDropoutLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
	const int feature_map_size = bottom[0]->height() * bottom[0]->width();
    if (this->phase_ == TRAIN) {
      const unsigned int* mask = rand_vec_.cpu_data();
	  const int mask_count = rand_vec_.count();
      for (int i = 0; i < mask_count; ++i) {
		Dtype* bottom_feature_diff = bottom_diff + i*feature_map_size;
		const Dtype* top_feature_diff = top_diff + i*feature_map_size;
		// caffe_scal(feature_map_size, mask[i] * scale_, bottom_feature_diff);
		caffe_cpu_scale(feature_map_size, mask[i]*scale_, top_feature_diff, bottom_feature_diff);
		// caffe_cpu_axpby(feature_map_size, mask[i]*scale_, top_feature_diff, Dtype(0), bottom_feature_diff);
      }
    } else {
      caffe_copy(top[0]->count(), top_diff, bottom_diff);
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(ConvDropoutLayer);
#endif

INSTANTIATE_CLASS(ConvDropoutLayer);
REGISTER_LAYER_CLASS(ConvDropout);

}  // namespace caffe
