#include <vector>

#include "caffe/layers/superpixel_predict_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void forward_gpu_kernel(const int n, const Dtype* const sp_data,
		const Dtype* const pred_data, Dtype* out_data, const int num,
		const int channels, const int height, const int width, const int sp_num){
	CUDA_KERNEL_LOOP(index, n){
		const int n_idx = index / height / channels;
		const int c = (index / height) % channels;
		const int h = index % height;

		// Iter the width
		const int offset_out = ((n_idx*channels+c)*height+h)*width;
		const int offset_pred = (n_idx*channels+c)*sp_num;
		const int offset_sp = (n_idx*height+h)*width;
		const Dtype* pred_data_n = pred_data + offset_pred;
		const Dtype* sp_data_n = sp_data + offset_sp;
		Dtype* out_data_n = out_data + offset_out;

		for (int i = 0; i < width; i++){
			out_data_n[i] = pred_data_n[(int)(sp_data_n[i])];
		}
	}
}

template <typename Dtype>
void SuperpixelPredictLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	const Dtype* sp_data = bottom[1]->gpu_data();
	const Dtype* pred_data = bottom[0]->gpu_data();
	Dtype* out_data = top[0]->mutable_gpu_data();

	const int num = top[0]->num();
	const int channels = top[0]->channels();
	const int height = top[0]->height();
	const int width = top[0]->width();

	// Clear the memory to zero
	caffe_gpu_set(top[0]->count(), Dtype(0), out_data);

	// Iter all pixels in the minibatch
	// The num_kernels is n*c*h
	const int num_kernels = num * channels * height;
	forward_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
		CAFFE_CUDA_NUM_THREADS>>>(
				num_kernels, sp_data, pred_data, out_data, num, channels, height, width, sp_num_);
	CUDA_POST_KERNEL_CHECK;
}


template void SuperpixelPredictLayer<float>::Forward_gpu(const vector<Blob<float>*>& bottom, const vector<Blob<float>*>& top);
template void SuperpixelPredictLayer<double>::Forward_gpu(const vector<Blob<double>*>& bottom, const vector<Blob<double>*>& top);

//INSTANTIATE_LAYER_GPU_FUNCS(SuperpixelPredictLayer);

}  // namespace caffe
