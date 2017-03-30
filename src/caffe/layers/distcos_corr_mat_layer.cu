#include <vector>

#include "caffe/layers/distcos_corr_mat_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


template <typename Dtype>
__device__ Dtype dot_stride2(
		const int count,
		const Dtype* const data1,
		const int stride1,
		const Dtype* const data2,
		const int stride2){
	// Perform the dot with stride
	int offset1 = 0;
	int offset2 = 0;

	Dtype accum = 0;
	for (int i = 0; i < count; i++){
		offset1 = i * stride1;
		offset2 = i * stride2;

		accum += data1[offset1] * data2[offset2];
	}

	return accum;
}


template <typename Dtype>
__global__ void forward_gpu_kernel(
			const int num_kernels,
			const Dtype* bottom_data,
			Dtype* top_data,
			const int num,
			const int channels,
			const int height,
			const int width,
			const int stride,
			const int normalize){
	CUDA_KERNEL_LOOP(index, num_kernels){
		const int n = index / height / width;
		const int h = (index / width) % height;
		const int w = index % width;

		if (w >= h){
			return;
		}
		
		const int feature1_idx = (n*channels+0)*height+h;
		const int feature2_idx = (n*channels+0)*height+w;

		const int top_idx1 = ((n*height)+h)*width+w;
		const int top_idx2 = ((n*height)+w)*width+h;

		// Calc the angle between two feature vectors
		// Calc the L2 normal of the given two vectors
		Dtype feature1_norm = sqrt(dot_stride2(channels, bottom_data+feature1_idx, stride, bottom_data+feature1_idx, stride));
		Dtype feature2_norm = sqrt(dot_stride2(channels, bottom_data+feature2_idx, stride, bottom_data+feature2_idx, stride));
		
		// Calc the dot between the two feature vector
		Dtype dot = dot_stride2(channels, bottom_data+feature1_idx, stride, bottom_data+feature2_idx, stride);

		// Calc the angle between the vector
		Dtype ang;
		if (feature1_norm == 0 || feature2_norm == 0){
			ang = 3.14 / 2;
		}else{
			Dtype val = min(max(dot / feature1_norm / feature2_norm, Dtype(-1)), Dtype(1));
			ang = acos(val);
		}

		// Normalize the h and w first if needed
		if (normalize){
			ang = min(ang / Dtype(3.1416), Dtype(1.0));
		}

		top_data[top_idx1] = ang;
		top_data[top_idx2] = ang;
	}
}

template <typename Dtype>
void DistCosCorrMatLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	const Dtype* bottom_data = bottom[0]->gpu_data();
	Dtype* top_data = top[0]->mutable_gpu_data();

	const int num = bottom[0]->num();
	const int channels = bottom[0]->channels();
	const int height = top[0]->height();
	const int width = top[0]->width();
	const int stride = bottom[0]->count(2);

	CHECK_EQ(height, width);

	// Clear the top
	caffe_gpu_set(top[0]->count(), Dtype(0), top_data);

	// TODO: Here the Load of each kernel is unbanlanced, there must be a better
	// way to balance the kernel load
	const int num_kernels = num * height * width;
	forward_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>(
			num_kernels,
			bottom_data,
			top_data,
			num,
			channels,
			height,
			width,
			stride,
			normalize_);
}


INSTANTIATE_LAYER_GPU_FUNCS(DistCosCorrMatLayer);

}  // namespace caffe
