#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/unpackbits_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

// Kernel used for compact unpact
// the thread number is n*c*h
template <typename Dtype>
__global__ void CompactUnpack(const int nthreads,
    const Dtype* const bottom_data,
	const int num, const int channels, const int height,
    const int width, const int target_channels,
	Dtype* const top_data) {
 CUDA_KERNEL_LOOP(index, nthreads) {
	const int h = index % height;
	// const int c = (index / height) % channels;
	const int n = index / height / channels;
	const int channel_offset = height * width;

	int s_iter = (target_channels+1)/2;
	// Loop the w
	// section 0, w < h
	const int base_idx = (n*channels*height + h)*width;
	const int base_top_idx = (n*target_channels*height+h)*width;
	for (int w = 0; w < h; w++){
		int idx = base_idx + w;
		int top_idx = base_top_idx + w;
		// TODO: If the channels is not 1, there will be a problem
		int top_idx_sym = (n*target_channels*height+w)*width+h;

		uint8_t val = static_cast<uint8_t>(bottom_data[idx]);
		for (int i = 0; i < s_iter; i++){
			int target_val = (val & (1u << i)) >> i;
			top_data[top_idx+channel_offset*i*2] = static_cast<Dtype>(target_val);
			top_data[top_idx_sym+channel_offset*i*2] = static_cast<Dtype>(target_val);
		}
	}
	// Section 1, w > h
	s_iter = target_channels/2;
	for (int w = h+1; w < width; w++){
		int idx = base_idx + w;
		int top_idx = base_top_idx + w;
		// TODO: If the channels is not 1, there will be a problem
		int top_idx_sym = (n*target_channels*height+w)*width+h;

		uint8_t val = static_cast<uint8_t>(bottom_data[idx]);
		for (int i = 0; i < s_iter; i++){
			int target_val = (val & (1u << i)) >> i;
			top_data[top_idx+channel_offset*(i*2+1)] = static_cast<Dtype>(target_val);
			top_data[top_idx_sym+channel_offset*(i*2+1)] = static_cast<Dtype>(target_val);
		}
	}
  }
}

// Kernel used for non-compact unpact
// the thread number is n*c*h
template <typename Dtype>
__global__ void NoCompactUnpack(const int nthreads,
    const Dtype* const bottom_data,
	const int num, const int channels, const int height,
    const int width, const int target_channels,
	Dtype* const top_data) {
 CUDA_KERNEL_LOOP(index, nthreads) {
	const int h = index % height;
	// const int c = (index / height) % channels;
	const int n = index / height / channels;
	const int channel_offset = height * width;

	// Loop the w
	// section 0, w < h
	const int base_idx = (n*channels*height + h)*width;
	const int base_top_idx = (n*target_channels*height+h)*width;
	for (int w = 0; w < h; w++){
		int idx = base_idx + w;
		int top_idx = base_top_idx + w;
		// TODO: If the channels is not 1, there will be a problem
		int top_idx_sym = (n*target_channels*height+w)*width+h;

		uint8_t val = static_cast<uint8_t>(bottom_data[idx]);
		for (int i = 0; i < target_channels; i++){
			int target_val = (val & (1u << i)) >> i;
			top_data[top_idx+channel_offset*i] = static_cast<Dtype>(target_val);
			top_data[top_idx_sym+channel_offset*i] = static_cast<Dtype>(target_val);
		}
	}
  }
}

template <typename Dtype>
void UnpackBitsLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  DLOG(INFO)<<"In UnpackBitsLayer Forward_gpu";
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = top[0]->count();
  // Init to 0
  caffe_gpu_set(count, Dtype(0), top_data);
  const int thread_num = bottom[0]->num() * bottom[0]->channels() * bottom[0]->height();

  if (compact_){
	  DLOG(INFO)<<"Compact: thread_num:"<<thread_num<<"channels:"<<channels_;
	  CompactUnpack<Dtype><<<CAFFE_GET_BLOCKS(thread_num), CAFFE_CUDA_NUM_THREADS>>>(thread_num, bottom_data, bottom[0]->num(), bottom[0]->channels(), bottom[0]->height(), bottom[0]->width(), channels_, top_data);
  }else{
	  NoCompactUnpack<Dtype><<<CAFFE_GET_BLOCKS(thread_num), CAFFE_CUDA_NUM_THREADS>>>(thread_num, bottom_data, bottom[0]->num(), bottom[0]->channels(), bottom[0]->height(), bottom[0]->width(), channels_, top_data);
  }
  CUDA_POST_KERNEL_CHECK;
  DLOG(INFO)<<"Out UnpackBitsLayer Forward_gpu";
}


INSTANTIATE_LAYER_GPU_FUNCS(UnpackBitsLayer);


}  // namespace caffe
