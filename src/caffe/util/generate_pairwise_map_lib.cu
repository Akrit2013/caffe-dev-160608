#include "caffe/util/generate_pairwise_map_lib.hpp"

namespace caffe{
__device__ double atomicAddD2(double* address, double val)
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
__global__ void count_sup_label_kernel(const int num_kernels,
		const Dtype* const sup_data,
		Dtype* H_cord_data,
	    Dtype* W_cord_data,
		Dtype* label_counter_data, 
		const int num, 
		const int channels, 
		const int height_sup,
		const int width_sup, 
		const int label_num);

template <>
__global__ void count_sup_label_kernel<float>(const int num_kernels,
		const float* const sup_data,
		float* H_cord_data,
	    float* W_cord_data,
		float* label_counter_data, 
		const int num, 
		const int channels, 
		const int height_sup,
		const int width_sup, 
		const int label_num){
	CUDA_KERNEL_LOOP(index, num_kernels){
		const int n = index / channels / height_sup;
		const int c = (index / height_sup) % channels;
		const int h = index % height_sup;

		const int sup_offset = ((n * channels + c) * height_sup + h) * width_sup;
		const int offset = (n * channels + c) * label_num;

		// Iter the width
		for (int w = 0; w < width_sup; w++){
			int label = sup_data[sup_offset+w] + 0.5;

			atomicAdd((float*)(H_cord_data+offset+label), float(h));
			atomicAdd((float*)(W_cord_data+offset+label), float(w));
			atomicAdd((float*)(label_counter_data+offset+label), float(1));
		}
	}
}

template <>
__global__ void count_sup_label_kernel<double>(const int num_kernels,
		const double* const sup_data,
		double* H_cord_data,
	    double* W_cord_data,
		double* label_counter_data, 
		const int num, 
		const int channels, 
		const int height_sup,
		const int width_sup, 
		const int label_num){
	CUDA_KERNEL_LOOP(index, num_kernels){
		const int n = index / channels / height_sup;
		const int c = (index / height_sup) % channels;
		const int h = index % height_sup;

		const int sup_offset = ((n * channels + c) * height_sup + h) * width_sup;
		const int offset = (n * channels + c) * label_num;

		// Iter the width
		for (int w = 0; w < width_sup; w++){
			int label = sup_data[sup_offset+w] + 0.5;

			atomicAddD2((double*)(H_cord_data+offset+label), double(h));
			atomicAddD2((double*)(W_cord_data+offset+label), double(w));
			atomicAddD2((double*)(label_counter_data+offset+label), double(1));
		}
	}
}

template <typename Dtype>
__global__ void gen_pairwise_kernel(const int num_kernels,
		const Dtype* const sup_data, 
		const Dtype* const seg_data, 
		const Dtype* const H_cord_data, 
		const Dtype* const W_cord_data, 
		Dtype* pair_data, 
		const int num, 
		const int channels,
		const int height_sup, 
		const int width_sup, 
		const int height_seg, 
		const int width_seg, 
		const int label_num){
	CUDA_KERNEL_LOOP(index, num_kernels){
		// Calc the h and w, assume the h > w
		/*
		int h = 0;
		int w = 0;

		bool find = false;
		for (int ih = 1; ih < label_num; ih++){
			for (int iw = 0; iw < ih; iw++){
				int idx = ih + ih*(ih-1)/2 + iw + 1;
				if (idx == index){
					h = ih;
					w = iw;
					find = true;
					break;
				}
			}
			if (find){
				break;
			}
		}

		*/

		const int h = index / label_num;
		const int w = index % label_num;

		if (w >= h){
			return;
		}


		int label1 = h;
		int label2 = w;
		// Calc the zoom rate from the sup map to seg map
		const Dtype h_rate = Dtype(height_seg) / Dtype(height_sup);
		const Dtype w_rate = Dtype(width_seg) / Dtype(height_sup);

		// Iter the num and channel
		for (int n = 0; n < num; n++){
			for (int c = 0; c < channels; c++){
				const int offset_cord = (n * channels + c) * label_num;
				const int offset_seg = (n * channels + c) * height_seg * width_seg;
				const int offset_pair = (n * channels + c) * label_num * label_num; 

				// The cord in sup map
				const Dtype h1_sup = H_cord_data[offset_cord + label1];
				const Dtype h2_sup = H_cord_data[offset_cord + label2];
				const Dtype w1_sup = W_cord_data[offset_cord + label1];
				const Dtype w2_sup = W_cord_data[offset_cord + label2];

				// The cord in seg map
				const Dtype h1_seg = h1_sup * h_rate;
				const Dtype h2_seg = h2_sup * h_rate;
				const Dtype w1_seg = w1_sup * w_rate;
				const Dtype w2_seg = w2_sup * w_rate;

				// Calc a line between label1 and label2 in seg map
				// We use the smaller cord as the start and the larger cord as the end
				Dtype h_start, h_end;
				Dtype w_start, w_end;

				// Decide we should use the h or w as the base cord
				if (fabs(h1_seg-h2_seg) > fabs(w1_seg-w2_seg)){
					if (h1_seg > h2_seg){
						h_start = h2_seg;
						h_end = h1_seg;
						w_start = w2_seg;
						w_end = w1_seg;
					}else{
						h_start = h1_seg;
						h_end = h2_seg;
						w_start = w1_seg;
						w_end = w2_seg;
					}
					const Dtype grad = (w_end - w_start) / (h_end - h_start);

					// Iter the line points
					Dtype max_val = 0;

					for (int ih = h_start; ih < h_end; ih++){
						int iw = w_start + grad * (ih - h_start) + 0.5;
						const Dtype val = seg_data[offset_seg + ih * width_seg + iw];
						if (val > max_val){
							max_val = val;
						}
					}
					// Set the max_val to the pairwise map
					pair_data[offset_pair + label1 * label_num + label2] = max_val;
					pair_data[offset_pair + label2 * label_num + label1] = max_val;
				}else{
					if (w1_seg > w2_seg){
						w_start = w2_seg;
						w_end = w1_seg;
						h_start = h2_seg;
						h_end = h1_seg;
					}else{
						w_start = w1_seg;
						w_end = w2_seg;
						h_start = h1_seg;
						h_end = h2_seg;
					}
					const Dtype grad = (h_end - h_start) / (w_end - w_start);
					// Iter the line points
					Dtype max_val = 0;

					for (int iw = w_start; iw < w_end; iw++){
						int ih = h_start + grad * (iw - w_start) + 0.5;
						const Dtype val = seg_data[offset_seg + ih * width_seg + iw];
						if (val > max_val){
							max_val = val;
						}
					}
					// Set the max_val to the pairwise map
					pair_data[offset_pair + label1 * label_num + label2] = max_val;
					pair_data[offset_pair + label2 * label_num + label1] = max_val;
				}
			}
		}
	}
}

template <typename Dtype>
void GenPairwiseMap_gpu(const Blob<Dtype>& blob_sup, const Blob<Dtype>& blob_seg, Blob<Dtype>& blob_pair){
	const Dtype* sup_data = blob_sup.gpu_data();
	const Dtype* seg_data = blob_seg.gpu_data();
	Dtype* pair_data = blob_pair.mutable_gpu_data();

	CHECK_EQ(blob_seg.num(), blob_sup.num());
	CHECK_EQ(blob_seg.num(), blob_pair.num());

	CHECK_EQ(blob_seg.channels(), blob_sup.channels());
	CHECK_EQ(blob_seg.channels(), blob_pair.channels());

	const int num = blob_seg.num();
	const int channels = blob_seg.channels();
	const int height_seg = blob_seg.height();
	const int width_seg = blob_seg.width();
	const int height_sup = blob_sup.height();
	const int width_sup = blob_sup.width();
	const int height_pair = blob_pair.height();
	const int width_pair = blob_pair.width();

	CHECK_EQ(height_pair, width_pair);

	// Check the max superpixel and the size of the pairwise map
	// Normally, the size of pairwise map equals to the max_label + 1
	const int max_label = round(caffe_gpu_amax(blob_sup.count(2), sup_data));
	CHECK_EQ(height_pair, max_label + 1) << "The size of the pairwise blob not match with the max label";

	const int label_num = height_pair; 

	// Scan the whole superpixel image to record the loacation of  every pixels
	// The tmp blob to record the accumulation of H and W of each label
	Blob<Dtype> H_cord_accum, W_cord_accum;
	H_cord_accum.Reshape(num, channels, label_num, 1);
	W_cord_accum.Reshape(num, channels, label_num, 1);

	// The tmp blob for the number of the 
	Blob<Dtype> label_counter;
	label_counter.Reshape(num, channels, label_num, 1);

	// The number of the kernel
	int num_kernels = num * channels * height_sup;
	Dtype* H_cord_data = H_cord_accum.mutable_gpu_data();
	Dtype* W_cord_data = W_cord_accum.mutable_gpu_data();
	Dtype* label_counter_data = label_counter.mutable_gpu_data();

	caffe_gpu_set(H_cord_accum.count(), Dtype(0), H_cord_data);
	caffe_gpu_set(W_cord_accum.count(), Dtype(0), W_cord_data);
	caffe_gpu_set(label_counter.count(), Dtype(0), label_counter_data);

	count_sup_label_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>
		(num_kernels, sup_data, H_cord_data, W_cord_data, label_counter_data, num, channels,
		 height_sup, width_sup, label_num);
	CUDA_POST_KERNEL_CHECK;

	// Calc the central cord of the X and Y
	caffe_gpu_div(H_cord_accum.count(), H_cord_data, label_counter_data, H_cord_data);
	caffe_gpu_div(W_cord_accum.count(), W_cord_data, label_counter_data, W_cord_data);

	// Calc the pairwise map
	// TODO: Here will be a better way to save half of the threads
	// The thread number should be height_pair * width_pair / 2
	// num_kernels = label_num * (label_num - 1) / 2;
	num_kernels = label_num * label_num;
	gen_pairwise_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>
		(num_kernels, sup_data, seg_data, H_cord_data, W_cord_data, pair_data, num, channels,
		 height_sup, width_sup, height_seg, width_seg, label_num);
	CUDA_POST_KERNEL_CHECK;

}


template void GenPairwiseMap_gpu<float>(const Blob<float>& blob_sup, const Blob<float>& blob_seg, Blob<float>& blob_pair);
template void GenPairwiseMap_gpu<double>(const Blob<double>& blob_sup, const Blob<double>& blob_seg, Blob<double>& blob_pair);


}
