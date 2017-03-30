#include <cfloat>
#include <vector>

#include "caffe/layers/fusion_l2max_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void compare_l2max_kernel(
				const int num_kernels, 
				const Dtype* const bottom_data, 
				int* mask_data, 
				Dtype* top_data, 
				const int num, 
				const int channels, 
				const int height, 
				const int width,
				const int bottom_index){
	CUDA_KERNEL_LOOP(index, num_kernels){
		const int h = index / width;
		const int w = index % width;

		const int plan_size = height * width;
		// Loop the num and channels
		for (int n = 0; n < num; n++){
			int data_offset = ((n*channels+0)*height+h)*width+w;
			int mask_offset = (n*height+h)*width+w;
			Dtype top_len = 0;
			Dtype bottom_len = 0;
			for (int c = 0; c < channels; c++){
				int data_idx = data_offset + c*plan_size;
				Dtype bottom_val = bottom_data[data_idx];
				Dtype top_val = top_data[data_idx];
				top_len += top_val * top_val;
				bottom_len += bottom_val * bottom_val;
			}
			if (bottom_len > top_len){
				for (int c = 0; c < channels; c++){
					int data_idx = data_offset + c*plan_size;
					top_data[data_idx] = bottom_data[data_idx];
				}
				mask_data[mask_offset] = bottom_index;
			}
		}
	}
}

template <typename Dtype>
void FusionL2MaxLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	int* mask_data =  max_idx_.mutable_gpu_data();
	Dtype* top_data = top[0]->mutable_gpu_data();
	const int num = top[0]->num();
	const int channels = top[0]->channels();
	const int height = top[0]->height();
	const int width = top[0]->width();

	caffe_gpu_set(max_idx_.count(), 0, mask_data);
	caffe_gpu_set(top[0]->count(), Dtype(0), top_data);

	// Compare the bottom with top
	const int num_kernels = height * width;
	for(int i = 0; i < bottom.size(); i++){
		const Dtype* bottom_data = bottom[i]->gpu_data();
		compare_l2max_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>(
				num_kernels, bottom_data, mask_data, top_data, num, channels, height, width, i);
		CUDA_POST_KERNEL_CHECK;
	}
}

template <typename Dtype>
__global__ void backward_gpu_kernel(
		const int num_kernels, 
		const Dtype* const top_diff, 
		const int* const mask_data, 
		Dtype* bottom_diff, 
		const int num, 
		const int channels, 
		const int height, 
		const int width, 
		const int bottom_index){
	CUDA_KERNEL_LOOP(index, num_kernels){
		const int h = index / width;
		const int w = index % width;

		const int plan_size = height * width;
		// Iter the num and channels
		for (int n = 0; n < num; n++){
			const int data_offset = ((n*channels+0)*height+h)*width+w;
			const int mask_offset = (n*height+h)*width+w;
			if (mask_data[mask_offset] == bottom_index){
				for (int c = 0; c < channels; c++){
					int data_idx = data_offset + c * plan_size;
					bottom_diff[data_idx] = top_diff[data_idx];
				}
			}
		}
	}
}


template <typename Dtype>
void FusionL2MaxLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	const Dtype* top_diff = top[0]->gpu_diff();
	const int* mask_data = max_idx_.gpu_data();
	const int num = top[0]->num();
	const int channels = top[0]->channels();
	const int height = top[0]->height();
	const int width = top[0]->width();

	for (int i = 0; i < bottom.size(); ++i) {
		if (propagate_down[i]) {
			Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
			// Clear the diff data
			caffe_gpu_set(bottom[i]->count(), Dtype(0), bottom_diff);
			const int num_kernels = height * width;

			backward_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>(
					num_kernels, top_diff, mask_data, bottom_diff, num, channels, height, width, i);
			CUDA_POST_KERNEL_CHECK;
		}
	}
}

INSTANTIATE_LAYER_GPU_FUNCS(FusionL2MaxLayer);

}  // namespace caffe
