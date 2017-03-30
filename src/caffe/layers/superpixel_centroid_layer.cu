#include <vector>

#include "caffe/layers/superpixel_centroid_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
__device__ double atomicAddD3(double* address, double val)
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
__global__ void forward_gpu_kernel_accum(
			const int num_kernels,
			const Dtype* const sp_data,
			Dtype* sp_accum_data,
			Dtype* sp_num_data,
			const int num,
			const int channels,
			const int height,
			const int width,
			const int num_output);

template <>
__global__ void forward_gpu_kernel_accum<float>(
			const int num_kernels,
			const float* const sp_data,
			float* sp_accum_data,
			float* sp_num_data,
			const int num,
			const int channels,
			const int height,
			const int width,
			const int num_output){
	CUDA_KERNEL_LOOP(index, num_kernels){
		const int n = index / height / channels;
		const int c = (index / height) % channels;
		const int h = index % height;

		const int sp_offset = ((((n * channels) + c) * height) + h) * width;
		const int accu_offset = (n * channels + c) * num_output * 2;
		const int num_offset = (n * channels + c) * num_output;
		for(int w = 0; w < width; w++){
			const int sp_id = sp_data[sp_offset + w];
			const int accu_idx = accu_offset + sp_id * 2;
			const int num_idx = num_offset + sp_id;

			atomicAdd((float*)(sp_accum_data+accu_idx), float(h));
			atomicAdd((float*)(sp_accum_data+accu_idx+1), float(w));
			atomicAdd((float*)(sp_num_data+num_idx), float(1));
		}
	}
}

template <>
__global__ void forward_gpu_kernel_accum<double>(
			const int num_kernels,
			const double* const sp_data,
			double* sp_accum_data,
			double* sp_num_data,
			const int num,
			const int channels,
			const int height,
			const int width,
			const int num_output){
	CUDA_KERNEL_LOOP(index, num_kernels){
		const int n = index / height / channels;
		const int c = (index / height) % channels;
		const int h = index % height;

		const int sp_offset = ((((n * channels) + c) * height) + h) * width;
		const int accu_offset = (n * channels + c) * num_output * 2;
		const int num_offset = (n * channels + c) * num_output;
		for(int w = 0; w < width; w++){
			const int sp_id = sp_data[sp_offset + w];
			const int accu_idx = accu_offset + sp_id * 2;
			const int num_idx = num_offset + sp_id;

			atomicAddD3((double*)(sp_accum_data+accu_idx), double(h));
			atomicAddD3((double*)(sp_accum_data+accu_idx+1), double(w));
			atomicAddD3((double*)(sp_num_data+num_idx), double(1));
		}
	}
}

template <typename Dtype>
__global__ void forward_gpu_kernel_average(
			const int num_kernels,
			Dtype* top_data,
			const Dtype* const sp_accum_data,
			const Dtype* const sp_num_data,
			const int num,
			const int channels,
			const int num_output,
			const bool normalize,
			const Dtype height,
			const Dtype width){
	CUDA_KERNEL_LOOP(index, num_kernels){
		const int n = index / channels / num_output;
		const int c = (index / num_output) % channels;
		const int h = index % num_output;

		const int top_offset = ((n * channels + c ) * num_output + h )* 2;
		const int num_offset = (n * channels + c ) * num_output + h;
		if(normalize){
			top_data[top_offset] = sp_accum_data[top_offset] / sp_num_data[num_offset] / height;
			top_data[top_offset+1] = sp_accum_data[top_offset+1] / sp_num_data[num_offset] / width;
		}else{
			top_data[top_offset] = sp_accum_data[top_offset] / sp_num_data[num_offset];
			top_data[top_offset+1] = sp_accum_data[top_offset+1] / sp_num_data[num_offset];
		}
	}
}

template <typename Dtype>
void SuperpixelCentroidLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	const Dtype* sp_data = bottom[0]->gpu_data();
	Dtype* top_data = top[0]->mutable_gpu_data();
	Dtype* sp_accum_data = sp_accum_.mutable_gpu_data();
	Dtype* sp_num_data = sp_num_.mutable_gpu_data();

	const int num = bottom[0]->num();
	const int channels = bottom[0]->channels();
	const int height = bottom[0]->height();
	const int width = bottom[0]->width();

	// Check the max id of superpixel map
	const int max_id = int(caffe_gpu_amax(bottom[0]->count(), sp_data));
	if(max_id + 1 != num_output_){
		if (check_){
			LOG(FATAL) << "The num_output and max superpixel+1 not match: "<<num_output_<<" vs "<<max_id+1;
		}else{
			LOG(WARNING) << "The num_output and max superpixel+1 not match: "<<num_output_<<" vs "<<max_id+1;
		}
	}
	// Clear the sp_accum_ and sp_num_
	caffe_gpu_set(sp_accum_.count(), Dtype(0), sp_accum_data);
	caffe_gpu_set(sp_num_.count(), Dtype(0), sp_num_data);
	// Accumulate the pixel coordinates
	const int num_kernels = num * channels * height;
	forward_gpu_kernel_accum<Dtype><<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>(
			num_kernels,
			sp_data,
			sp_accum_data,
			sp_num_data,
			num,
			channels,
			height,
			width,
			num_output_);
	// Average the accumulation to calc the central coordinates
	const int num_kernels2 = num * channels * num_output_;
	forward_gpu_kernel_average<<<CAFFE_GET_BLOCKS(num_kernels2), CAFFE_CUDA_NUM_THREADS>>>(
			num_kernels2,
			top_data,
			sp_accum_data,
			sp_num_data,
			num,
			channels,
			num_output_,
			normalize_,
			height_,
			width_);

}


INSTANTIATE_LAYER_GPU_FUNCS(SuperpixelCentroidLayer);

}  // namespace caffe
