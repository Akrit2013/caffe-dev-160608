#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/pairwise_map_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__device__ Dtype dot_stride(
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
__device__ Dtype calc_dist2(
		const int index1,
		const int index2,
		const int height,
		const int width){
	// Calc the coordinate of the two points
	int h1 = index1 / width;
	int h2 = index2 / width;
	int w1 = index1 % width;
	int w2 = index2 % width;

	// Normalize the coordinate to 0 and 1
	Dtype h1n = Dtype(h1) / height;
	Dtype w1n = Dtype(w1) / width;
	Dtype h2n = Dtype(h2) / height;
	Dtype w2n = Dtype(w2) / width;

	// Calc the Square of the L2 distance
	Dtype dh = h1n - h2n;
	Dtype dw = w1n - w2n;

	return dh * dh + dw * dw;
}

template <typename Dtype>
__global__ void forward_gpu_kernel(
		const int num_kernels, 
		const Dtype* const bottom_data, 
		Dtype* top_data, 
		const int num, 
		const int channels, 
		const int height, 
		const int width,
		const int sp_height,
		const int sp_width,
		const Dtype theta1,
		const Dtype theta2){
	CUDA_KERNEL_LOOP(index, num_kernels){
		const int n = index / height / width;
		const int h = (index / width) % height;
		const int w = index % width;

		if (w >= h){
			return;
		}

		const int sp_plan = height;

		const int feature1_idx = (n*channels+0)*sp_plan+h;
		const int feature2_idx = (n*channels+0)*sp_plan+w;
		const int top_idx = ((n*height)+h)*width+w;
		const int top_idx2 = ((n*height)+w)*width+h;
		// Calc the angle between two feature vectors
		// Calc the L2 normal of the given two vectors
		Dtype feature1_norm = sqrt(dot_stride(channels, bottom_data+feature1_idx, sp_plan, bottom_data+feature1_idx, sp_plan));
		Dtype feature2_norm = sqrt(dot_stride(channels, bottom_data+feature2_idx, sp_plan, bottom_data+feature2_idx, sp_plan));

		// Calc the dot between the two feature vector
		Dtype dot = dot_stride(channels, bottom_data+feature1_idx, sp_plan, bottom_data+feature2_idx, sp_plan);

		// Calc the angle between the vector
		Dtype ang;
		if (feature1_norm == 0 || feature2_norm == 0){
			ang = 3.14 / 2;
		}else{
			Dtype val = min(max(dot / feature1_norm / feature2_norm, Dtype(-1)), Dtype(1));
			ang = acos(val);
		}

		// Calc the distance between two points
		Dtype dist = calc_dist2<Dtype>(w, h, sp_height, sp_width);

		top_data[top_idx] = -ang / theta1 - dist / theta2;
		top_data[top_idx2] = top_data[top_idx];
	}
}

template <typename Dtype>
void PairwiseMapLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	const Dtype* bottom_data = bottom[0]->gpu_data();
	Dtype* top_data = top[0]->mutable_gpu_data();
	const int num = top[0]->num();
	const int channels = bottom[0]->channels();
	const int height = top[0]->height();
	const int width = top[0]->width();

	CHECK_EQ(height, width);
	// Clear the top data
	caffe_gpu_set(top[0]->count(), Dtype(0), top_data);

	// The kernel number is the num * height * width
	// TODO: Since the top is a symmtic matrix, so half of the calculation is not
	// necessary. There might be a better method to assign the threads to the pixels
	const int num_kernels = num * height * width;
	forward_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>(
			num_kernels, bottom_data, top_data, num, channels,
		   	height, width, height_, width_, theta1_, theta2_);

	CUDA_POST_KERNEL_CHECK;

	// exp
	caffe_gpu_exp(top[0]->count(), top_data, top_data);
}



// INSTANTIATE_LAYER_GPU_FUNCS(PairwiseMapLayer);
template void PairwiseMapLayer<float>::Forward_gpu(const vector<Blob<float>*>& bottom,
      const vector<Blob<float>*>& top);
template void PairwiseMapLayer<double>::Forward_gpu(const vector<Blob<double>*>& bottom,
      const vector<Blob<double>*>& top);


}  // namespace caffe
