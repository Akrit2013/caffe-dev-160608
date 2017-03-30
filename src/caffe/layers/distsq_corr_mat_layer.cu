#include <vector>

#include "caffe/layers/distsq_corr_mat_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void forward_gpu_kernel(
			const int num_kernels,
			const Dtype* bottom_data,
			Dtype* top_data,
			const int num,
			const int channels,
			const int height,
			const int normalize,
			const int im_height,
			const int im_width){
	CUDA_KERNEL_LOOP(index, num_kernels){
		const int n = index / channels / height;
		const int c = (index / height) % channels;
		const int h = index % height;

		const int bottom_offset = (n * channels + c) * height * 2;
		const int top_offset = ((n * channels + c) * height + h) * height;

		const Dtype* sp1_data = bottom_data + bottom_offset + h * 2;
		const Dtype h1 = sp1_data[0];
		const Dtype w1 = sp1_data[1];

		// Iter the row
		for(int w = 0; w < height; w++){
			const Dtype* sp2_data = bottom_data + bottom_offset + w * 2;
			const Dtype h2 = sp2_data[0];
			const Dtype w2 = sp2_data[1];

			Dtype diff_h = h1 - h2;
			Dtype diff_w = w1 - w2;

			if(normalize){
				diff_h = diff_h / Dtype(im_height);
				diff_w = diff_w / Dtype(im_width);
			}

			Dtype dist = diff_h * diff_h + diff_w * diff_w;

			const int top_idx1 = top_offset + w;
			const int top_idx2 = ((n * channels + c) * height + w) * height + h;

			top_data[top_idx1] = dist;
			top_data[top_idx2] = dist;
		}
	}
}

template <typename Dtype>
void DistSqCorrMatLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	const Dtype* bottom_data = bottom[0]->gpu_data();
	Dtype* top_data = top[0]->mutable_gpu_data();

	const int num = bottom[0]->num();
	const int channels = bottom[0]->channels();
	const int height = bottom[0]->height();
	// Clear the top
	caffe_gpu_set(top[0]->count(), Dtype(0), top_data);

	// TODO: Here the Load of each kernel is unbanlanced, there must be a better
	// way to balance the kernel load
	const int num_kernels = num * channels * height;
	forward_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>(
			num_kernels,
			bottom_data,
			top_data,
			num,
			channels,
			height,
			normalize_,
			height_,
			width_);
}


INSTANTIATE_LAYER_GPU_FUNCS(DistSqCorrMatLayer);

}  // namespace caffe
