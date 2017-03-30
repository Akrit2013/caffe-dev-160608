#include <algorithm>
#include <vector>

#include "caffe/layers/l2norm_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void forward_gpu_kernel(const int num_kernels,
		const Dtype* const bottom_data,
		Dtype* top_data, 
		Dtype* len_data, 
		const int num, 
		const int channels, 
		const int height, 
		const int width, 
		const Dtype norm_val_){
	CUDA_KERNEL_LOOP(index, num_kernels){
		const int h = index / width;
		const int w = index % width;

		const int plan_size = height * width;

		// Iter the n and the c
		for (int n = 0; n < num; n++){
			int len_idx = n * plan_size + h * width + w;
			int bottom_idx = ((n*channels+0)*height+h)*width+w;
			for (int c = 0; c < channels; c++){
				len_data[len_idx] += bottom_data[bottom_idx] * bottom_data[bottom_idx] / norm_val_;
				bottom_idx += plan_size;
			}

			if (len_data[len_idx] == 0){
				len_data[len_idx] = 1;
			}else{
				len_data[len_idx] = sqrt(len_data[len_idx]);
			}

			bottom_idx = ((n*channels+0)*height+h)*width+w;
			for (int c = 0; c < channels; c++){
				top_data[bottom_idx] = bottom_data[bottom_idx] / len_data[len_idx];
				bottom_idx += plan_size;
			}
		}
	}
}


template <typename Dtype>
__global__ void backward_gpu_kernel(
        const int num_kernels,
		const Dtype* const top_diff,
		const Dtype* len_data,
	    Dtype* bottom_diff,
	   	const int num, 
		const int channels, 
		const int height, 
		const int width){
  CUDA_KERNEL_LOOP(index, num_kernels) {
	  const int n = index / channels / height;
	  const int c = (index / height) % channels;
	  const int h = index % height;

	  int offset = ((n*channels+c)*height+h)*width;
	  int len_offset = (n*height+h)*width;

	  for (int w = 0; w < width; w++){
		  bottom_diff[offset+w] = top_diff[offset+w] / len_data[len_offset+w];
	  }
  }
}

template <typename Dtype>
void L2NormLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  Dtype* len_data = length_map_.mutable_gpu_data();

  const int num = bottom[0]->num();
  const int channels = bottom[0]->channels();
  const int height = bottom[0]->height();
  const int width = bottom[0]->width();

  const int num_kernels = height*width;

  caffe_gpu_set(length_map_.count(), Dtype(0), len_data);

  forward_gpu_kernel<<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>(
		  num_kernels, bottom_data, top_data, len_data, num, channels, height, width, norm_val_);

  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void L2NormLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* len_data = length_map_.gpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

  const int num = bottom[0]->num();
  const int channels = bottom[0]->channels();
  const int height = bottom[0]->height();
  const int width = bottom[0]->width();

  const int num_kernels = num * channels * height;

  // Propagate to bottom
  if (propagate_down[0]) {
    backward_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
        CAFFE_CUDA_NUM_THREADS>>>(
        num_kernels, top_diff, len_data, bottom_diff, num, channels, height, width);
    CUDA_POST_KERNEL_CHECK;
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(L2NormLayer);


}  // namespace caffe
