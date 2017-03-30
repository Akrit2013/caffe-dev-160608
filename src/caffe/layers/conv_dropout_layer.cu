#include <vector>

#include "caffe/layers/conv_dropout_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void ConvDropoutForward(const int n, const int feature_map_size, const Dtype* in,
    const unsigned int* mask, const unsigned int threshold, const float scale,
    Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
	const Dtype* in_data = in + index*feature_map_size;
	Dtype* out_data = out + index*feature_map_size;
	const bool judge = (mask[index] > threshold);
	for (int i = 0; i < feature_map_size; i++){
		out_data[i] = in_data[i] * judge * scale;
	}
	// caffe_gpu_scale(feature_map_size, Dtype((mask[index] > threshold) * scale), in_data, out_data);
    // out[index] = in[index] * (mask[index] > threshold) * scale;

  }
}

template <typename Dtype>
void ConvDropoutLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  const int mask_count = rand_vec_.count();
  const int feature_map_size = bottom[0]->height() * bottom[0]->width();
  
  if (this->phase_ == TRAIN) {
    unsigned int* mask =
        static_cast<unsigned int*>(rand_vec_.mutable_gpu_data());
    caffe_gpu_rng_uniform(mask_count, mask);
    // set thresholds
    // NOLINT_NEXT_LINE(whitespace/operators)
    ConvDropoutForward<Dtype><<<CAFFE_GET_BLOCKS(mask_count), CAFFE_CUDA_NUM_THREADS>>>(
        mask_count, feature_map_size, bottom_data, mask, uint_thres_, scale_, top_data);
    CUDA_POST_KERNEL_CHECK;
  } else {
    caffe_copy(count, bottom_data, top_data);
  }
}

template <typename Dtype>
__global__ void ConvDropoutBackward(const int n, const int feature_map_size, const Dtype* in_diff,
    const unsigned int* mask, const unsigned int threshold, const float scale,
    Dtype* out_diff) {
  CUDA_KERNEL_LOOP(index, n) {
	const Dtype* in_data = in_diff + index*feature_map_size;
	Dtype* out_data = out_diff + index*feature_map_size;
	const bool judge = (mask[index] > threshold);
	for (int i = 0; i < feature_map_size; i++){
		out_data[i] = in_data[i] * judge * scale;
	}
	// caffe_gpu_scale(feature_map_size, Dtype((mask[index] > threshold) * scale), in_data, out_data);
    // out_diff[index] = in_diff[index] * scale * (mask[index] > threshold);
  }
}

template <typename Dtype>
void ConvDropoutLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    if (this->phase_ == TRAIN) {
      const unsigned int* mask =
          static_cast<const unsigned int*>(rand_vec_.gpu_data());
      const int count = bottom[0]->count();
  	  const int mask_count = rand_vec_.count();
	  const int feature_map_size = bottom[0]->height() * bottom[0]->width();
      // NOLINT_NEXT_LINE(whitespace/operators)
      ConvDropoutBackward<Dtype><<<CAFFE_GET_BLOCKS(mask_count),
        CAFFE_CUDA_NUM_THREADS>>>(
          mask_count, feature_map_size, top_diff, mask, uint_thres_, scale_, bottom_diff);
      CUDA_POST_KERNEL_CHECK;
    } else {
      caffe_copy(top[0]->count(), top_diff, bottom_diff);
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ConvDropoutLayer);

}  // namespace caffe
