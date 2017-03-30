#include <vector>

#include "caffe/layers/superpixel_pooling_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
__device__ double atomicAddD(double* address, double val)
{
	unsigned long long int* address_as_ull =
		(unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed,
				__double_as_longlong(val +
					__longlong_as_double(assumed)));
	} while (assumed != old);
	return __longlong_as_double(old);
}

template <typename Dtype>
__global__ void forward_gpu_kernel_ave(const int n, const Dtype* const sp_data,
	const Dtype* const bottom_data, Dtype* mask_data, Dtype* accum_data,
	Dtype* top_data, const int num_, const int in_channels_, const int in_height_,
	const int in_width_, const int sp_height_, const int sp_width_, const Dtype h_rate_,
	const Dtype w_rate_, const int num_output_);

/*
__device__ double atomicCAS(double* address, int compare, double val)
{
	unsigned long long int* address_as_ull =
		(unsigned long long int*)address;
	unsigned long long int compare_ull = compare;
	unsigned long long int old;
	old = atomicCAS(address_as_ull, compare_ull,
			__double_as_longlong(val));
	return __longlong_as_double(old);
}

__device__ float atomicCAS(float* address, int compare, float val)
{
	unsigned long int* address_as_ul = (unsigned long int*)address;
	unsigned long int compare_ul = compare;
	unsigned long int old;
	old = atomicCAS(address_as_ul, compare_ul, __float_as_long(val));
	return __long_as_float(old);
}
*/

/*
template <typename Dtype>
__global__ void forward_gpu_kernel_ave(const int n, const Dtype* const sp_data,
	const Dtype* const bottom_data, Dtype* mask_data, Dtype* accum_data,
	Dtype* top_data, const int num_, const int in_channels_, const int in_height_,
	const int in_width_, const int sp_height_, const int sp_width_, const Dtype h_rate_,
	const Dtype w_rate_, const int num_output_){
	CUDA_KERNEL_LOOP(index, n){
		const int bottom_n = index / in_height_ / in_channels_;
		const int bottom_c = (index / in_height_) % in_channels_;
		const int bottom_h = index % in_height_;

		// Calc the position
		const int sp_h = round(h_rate_*bottom_h);
		int sp_w = 0;
		int sp_idx = 0;
		const int sp_offset = (bottom_n*sp_height_+sp_h)*sp_width_;
		int bottom_idx = 0;
		const int top_offset = bottom_n * num_output_;
		int top_idx = 0;

		// The start position of the row
		const int offset = ((bottom_n*in_channels_+bottom_c)*in_height_+bottom_h)*in_width_;

		// Iter the width
		for (int w = 0; w < in_width_; w++){
			bottom_idx = offset + w;
			sp_w = round(w_rate_ * w);
			sp_idx = sp_offset + sp_w;
			top_idx = top_offset + (int)sp_data[sp_idx];
			// Accumulate
			atomicAdd((Dtype*)(accum_data+top_idx), Dtype(1));
			// Add the value
			atomicAdd(top_data+top_idx, bottom_data[bottom_idx]);
			// Record the label
			mask_data[bottom_idx] = (int)sp_data[sp_idx];
		}
	}
}
*/


template <>
__global__ void forward_gpu_kernel_ave<float>(const int n, const float* const sp_data,
	const float* const bottom_data, float* mask_data, float* accum_data,
	float* top_data, const int num_, const int in_channels_, const int in_height_,
	const int in_width_, const int sp_height_, const int sp_width_, const float h_rate_,
	const float w_rate_, const int num_output_){
	CUDA_KERNEL_LOOP(index, n){
		const int bottom_n = index / in_height_ / in_channels_;
		const int bottom_c = (index / in_height_) % in_channels_;
		const int bottom_h = index % in_height_;

		// Calc the position
		const int sp_h = round(h_rate_*(bottom_h+0.5));
		int sp_w = 0;
		int sp_idx = 0;
		const int sp_offset = (bottom_n*sp_height_+sp_h)*sp_width_;
		int bottom_idx = 0;
		const int top_offset = (bottom_n * in_channels_ + bottom_c) * num_output_;
		int top_idx = 0;

		// The start position of the row
		const int offset = ((bottom_n*in_channels_+bottom_c)*in_height_+bottom_h)*in_width_;

		// Iter the width
		for (int w = 0; w < in_width_; w++){
			bottom_idx = offset + w;
			sp_w = round(w_rate_ * (w+0.5));
			sp_idx = sp_offset + sp_w;
			top_idx = top_offset + (int)sp_data[sp_idx];
			// Accumulate
			atomicAdd((float*)(accum_data+top_idx), float(1));
			// Add the value
			atomicAdd(top_data+top_idx, bottom_data[bottom_idx]);
			// Record the label
			mask_data[bottom_idx] = (int)sp_data[sp_idx];
		}
	}
}

template <>
__global__ void forward_gpu_kernel_ave<double>(const int n, const double* const sp_data,
	const double* const bottom_data, double* mask_data, double* accum_data,
	double* top_data, const int num_, const int in_channels_, const int in_height_,
	const int in_width_, const int sp_height_, const int sp_width_, const double h_rate_,
	const double w_rate_, const int num_output_){
	CUDA_KERNEL_LOOP(index, n){
		const int bottom_n = index / in_height_ / in_channels_;
		const int bottom_c = (index / in_height_) % in_channels_;
		const int bottom_h = index % in_height_;

		// Calc the position
		const int sp_h = round(h_rate_*(bottom_h+0.5));
		int sp_w = 0;
		int sp_idx = 0;
		const int sp_offset = (bottom_n*sp_height_+sp_h)*sp_width_;
		int bottom_idx = 0;
		const int top_offset = (bottom_n * in_channels_ + bottom_c) * num_output_;
		int top_idx = 0;

		// The start position of the row
		const int offset = ((bottom_n*in_channels_+bottom_c)*in_height_+bottom_h)*in_width_;

		// Iter the width
		for (int w = 0; w < in_width_; w++){
			bottom_idx = offset + w;
			sp_w = round(w_rate_ * (w+0.5));
			sp_idx = sp_offset + sp_w;
			top_idx = top_offset + (int)sp_data[sp_idx];
			// Accumulate
			atomicAddD((double*)(accum_data+top_idx), double(1));
			// Add the value
			atomicAddD(top_data+top_idx, bottom_data[bottom_idx]);
			// Record the label
			mask_data[bottom_idx] = (int)sp_data[sp_idx];
		}
	}
}
/*
template <typename Dtype>
__global__ void forward_gpu_kernel_max(const int n, const Dtype* const sp_data,
	const Dtype* const bottom_data, Dtype* accum_data,
	Dtype* top_data, const int num_, const int in_channels_, const int in_height_,
	const int in_width_, const int sp_height_, const int sp_width_, const Dtype h_rate_,
	const Dtype w_rate_, const int num_output_){
	CUDA_KERNEL_LOOP(index, n){
		const int bottom_n = index / in_height_ / in_channels_;
		const int bottom_c = (index / in_height_) % in_channels_;
		const int bottom_h = index % in_height_;

		// Calc the position
		const int sp_h = round(h_rate_*bottom_h);
		int sp_w = 0;
		const int sp_offset = (bottom_n*sp_height_+sp_h)*sp_width_;
		int bottom_idx = 0;
		const int top_offset = bottom_n * num_output_;
		int top_idx = 0;

		Dtype val = 0;
		Dtype old_val = 0;

		// The start position of the row
		const int offset = ((bottom_n*in_channels_+bottom_c)*in_height_+bottom_h)*in_width_;

		// Iter the width
		for (int w = 0; w < in_width_; w++){
			bottom_idx = offset + w;
			sp_w = round(w_rate_ * w);
			sp_idx = sp_offset + sp_w;
			top_idx = top_offset + (int)sp_data[sp_idx];
			// Compare and Switch the value
			atomicCAS(top_data+top_idx, bottom_data[bottom_idx]);
			// Accumulate
			atomicAdd(accum_data+top_idx, Dtype(1));
		}
	}
}
*/

template <typename Dtype>
__global__ void fill_invalid_sp_kernel(const int num_kernels2, Dtype* accum_data,
		Dtype* top_data,
		const int num_, const int in_channels_, const int num_output_){
	CUDA_KERNEL_LOOP(index, num_kernels2){
		// Loop the num and channel
		int idx = 0;
		if (index == 0){
			for (int i = 0; i < num_ * in_channels_; i++){
				idx = i * num_output_;
				if (accum_data[idx] == 0){
					top_data[idx] = top_data[idx+1];
					accum_data[idx] = accum_data[idx+1];
					if (accum_data[idx] == 0){
						accum_data[idx] = 1;
					}
				}
			}
		}else if(index == num_output_-1){
			for (int i = 0; i < num_ * in_channels_; i++){
				idx = index + i * num_output_;
				if (accum_data[idx] == 0){
					top_data[idx] = top_data[idx-1];
					accum_data[idx] = accum_data[idx-1];
					if (accum_data[idx] == 0){
						accum_data[idx] = 1;
					}
				}
			}
		}else{
			for (int i = 0; i < num_ * in_channels_; i++){
				idx = index + i * num_output_;
				if (accum_data[idx] == 0){
					top_data[idx] = top_data[idx-1] + top_data[idx+1];
					accum_data[idx] = accum_data[idx-1] + accum_data[idx+1];
					if(accum_data[idx] == 0){
						accum_data[idx] = 1;
					}
				}
			}
		}
	}
}


template <typename Dtype>
__global__ void	backward_gpu_kernel_ave(const int n, Dtype* bottom_diff,
		const Dtype* const mask_data, const Dtype* const top_diff,
	    const int num_, const int in_channels_, const int in_height_, const int in_width_,
		const int sp_height_, const int w_rate_, const int num_output_){
	CUDA_KERNEL_LOOP(index, n){
		const int bottom_n = index / in_height_ / in_channels_;
		const int bottom_c = (index / in_height_) % in_channels_;
		const int bottom_h = index % in_height_;

		const int top_offset = (bottom_n * in_channels_ + bottom_c) * num_output_;
		const int offset = ((bottom_n*in_channels_+bottom_c)*in_height_+bottom_h)*in_width_;
		int bottom_idx = 0;
		int top_idx = 0;
		// Iter the width of the mask and the bottom_diff
		for (int w = 0; w < in_width_; w++){
			bottom_idx = offset + w;
			top_idx = top_offset + (int)mask_data[bottom_idx];
			bottom_diff[bottom_idx] = top_diff[top_idx];
		}
	}
}


template <typename Dtype>
void SuperpixelPoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	const Dtype* sp_data = bottom[1]->gpu_data();
	const Dtype* bottom_data = bottom[0]->gpu_data();
	Dtype* top_data = top[0]->mutable_gpu_data();
	Dtype* mask_data = mask_.mutable_gpu_data();
	Dtype* accum_data = accum_.mutable_gpu_data();

	// Clear the memory to zero
	caffe_gpu_set(accum_.count(), Dtype(0), accum_data);
	// Clear the memory to zero
	caffe_gpu_set(top[0]->count(), Dtype(0), top_data);

	const int num_kernels_ = num_ * in_channels_ * in_height_;
	const int num_kernels2 = num_output_;

	switch (pool_method_){
		case AVE:
			forward_gpu_kernel_ave<Dtype><<<CAFFE_GET_BLOCKS(num_kernels_), CAFFE_CUDA_NUM_THREADS>>>(
					num_kernels_, sp_data, bottom_data, mask_data, accum_data, top_data, num_,
					in_channels_, in_height_, in_width_, sp_height_,
					sp_width_, h_rate_, w_rate_, num_output_);
			CUDA_POST_KERNEL_CHECK;

			// If the accum_data have 0 position, that means the superpixel have no
			// value, use the adjancent value to fix it
			// Average the pooling
			// NOTE: Here may cause a NaN since some accum_data could be 0
			// So we have to check the accum_data and set all 0 to 1
			fill_invalid_sp_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels2), CAFFE_CUDA_NUM_THREADS>>>(
					num_kernels2, accum_data, top_data, num_, in_channels_, num_output_);
			CUDA_POST_KERNEL_CHECK;
			caffe_gpu_div(top[0]->count(), top_data, accum_data, top_data);

			break;
		case MAX:
			/*
			forward_gpu_kernel_max<Dtype><<<CAFFE_GET_BLOCKS(num_kernels_), CAFFE_CUDA_NUM_THREADS>>>(
					num_kernels_, sp_data, bottom_data, accum_data, top_data, num_,
					in_channels_, in_height_, in_width_, sp_height_,
					sp_width_, h_rate_, w_rate_, num_output_);
			CUDA_POST_KERNEL_CHECK;
			*/
			NOT_IMPLEMENTED;
			break;
		case STOCHASTIC:
			NOT_IMPLEMENTED;
			break;
		default:
			NOT_IMPLEMENTED;
			break;
	}
}

template <typename Dtype>
void SuperpixelPoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
	if (!propagate_down[0]){return;}

	const Dtype* top_diff = top[0]->gpu_diff();
	Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
	const Dtype* mask_data = mask_.gpu_data();

	// Clear the diff
	caffe_gpu_set(bottom[0]->count(), Dtype(0), bottom_diff);
	// The num_kernels_
	const int num_kernels_ = num_ * in_channels_ * in_height_; 
	switch (pool_method_){
		case AVE:
			backward_gpu_kernel_ave<Dtype><<<CAFFE_GET_BLOCKS(num_kernels_), CAFFE_CUDA_NUM_THREADS>>>(
					num_kernels_, bottom_diff, mask_data, top_diff, num_,
					in_channels_, in_height_, in_width_, sp_height_, w_rate_, num_output_);
			CUDA_POST_KERNEL_CHECK;
			break;
		case MAX:
			NOT_IMPLEMENTED;
			break;
		case STOCHASTIC:
			NOT_IMPLEMENTED;
			break;
		default:
			NOT_IMPLEMENTED;
			break;
	}
}



INSTANTIATE_LAYER_GPU_FUNCS(SuperpixelPoolingLayer);

}  // namespace caffe
