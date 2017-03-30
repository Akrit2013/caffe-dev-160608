#include <vector>

#include "caffe/layers/crf_unary_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void Forward_gpu_kernel(
		 const int nthreads,
		 const Dtype* const data_label,
		 Dtype* data_diff,
		 Dtype* bad_pixel_data,
		 const int num,
		 const int channels,
		 const int height,
		 const int width,
		 const bool has_max_label,
		 const bool has_min_label,
		 const bool has_invalid_label,
		 const Dtype max_label,
		 const Dtype min_label,
		 const Dtype invalid_label,
		 const Dtype C,
		 const bool has_h_rate,
		 const Dtype H){
	CUDA_KERNEL_LOOP(index, nthreads){
		const int n = index / height;
		const int h = index % height;

		const int data_offset = (n*channels*height+h)*width;
		const int bad_pixel_idx = index;
		const int interval = height * width;

		// Iter the width and channels
		for (int w = 0; w < width; w++){
			// Iter the channels
			int err_counter = 0;
			for (int c = 0; c < channels; c++){
				const int idx = data_offset + c * interval + w;
				Dtype dataval = data_label[idx];
				Dtype diffval = data_diff[idx];

				if (has_max_label && dataval > max_label){
					err_counter++;
				}else if(has_min_label && dataval < min_label){
					err_counter++;
				}else if(has_invalid_label && fabs(dataval - invalid_label) < 0.0001){
					err_counter++;
				}
				// alter the diff value
				if (diffval > 0 && diffval < C){
					// L1
					data_diff[idx] = C;
				}else if(diffval < 0 && -diffval < C){
					data_diff[idx] = -C;
				}
				if (has_h_rate && diffval > H){
					data_diff[idx] = H;
				}else if(has_h_rate && -diffval > H){
					data_diff[idx] = -H;
				}
			}

			// Only if all channels invalid, the pixel will be considered
			// as invalid
			if(err_counter == channels){
				bad_pixel_data[bad_pixel_idx] += channels;
				for (int c = 0; c < channels; c++){
					const int idx = data_offset + c * interval;
					data_diff[idx] = 0;
				}
			}
		}
	}
}

	
template <typename Dtype>
__global__ void calc_a_gpu_kernel(const int n, Dtype* a_data,
		const int num, const int channels, const int height, const int width,
		const Dtype* r_data){
	CUDA_KERNEL_LOOP(index, n) {
		const int n_idx = index / height / channels;
		const int c = (index / height) % channels;
		const int h = index % height;

		const int idx = ((n_idx*channels+c)*height+h)*width;
		// Calc the sum of the row in r_data
		Dtype sum = 0;
		for (int i = 0; i < width; i++){
			sum += r_data[idx+i];
		}
		a_data[idx+h] = sum + 1;
	}
}


template <typename Dtype>
void CrfUnaryLossLayer<Dtype>::Calc_A_gpu(void){
	// A = I + D -R
	// Set A to 0
	caffe_gpu_set(A_.count(), Dtype(0), A_.mutable_gpu_data());
	const int num = A_.num();
	const int channels = A_.channels();
	const int height = A_.height();
	const int width = A_.width();
	const Dtype* r_data = R_.gpu_data();
	Dtype* a_data = A_.mutable_gpu_data();
	// A = I + D
	// kernel num: n*c*h
	int num_kernels = num * channels * height;
	calc_a_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                             CAFFE_CUDA_NUM_THREADS>>>(
      num_kernels, a_data, num, channels, height, width, r_data);
	CUDA_POST_KERNEL_CHECK;
	// A = A - R
	caffe_gpu_axpy(A_.count(), Dtype(-1), r_data, a_data);

}

template <typename Dtype>
void CrfUnaryLossLayer<Dtype>::Calc_R_gpu(const Blob<Dtype>* bottom){
	caffe_gpu_scale(bottom->count(), Dtype(alpha_), bottom->gpu_data(), R_.mutable_gpu_data());
}

template <typename Dtype>
void CrfUnaryLossLayer<Dtype>::Euclidean_loss_gpu(const Blob<Dtype>* gt, Blob<Dtype>* top){
	const Dtype* gt_data = gt->gpu_data();
	const Dtype* pred_data = Pred_.gpu_data();
	const int count = Pred_.count();
	caffe_gpu_sub(
			count,
			pred_data,
			gt_data,
			Pred_.mutable_gpu_diff());
	Dtype dot;
	caffe_gpu_dot(count, Pred_.gpu_diff(), Pred_.gpu_diff(), &dot);
	Dtype loss = dot / count / Dtype(2);
	top->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void CrfUnaryLossLayer<Dtype>::Berhu_loss_gpu(const Blob<Dtype>* gt, Blob<Dtype>* top){
  const int count = Pred_.count();
  const Dtype* label_data = gt->gpu_data();
  const Dtype* pred_data = Pred_.gpu_data();
  caffe_gpu_sub(
      count,
	  pred_data,
	  label_data,
      Pred_.mutable_gpu_diff());
  Dtype max_diff = 0;

  switch(c_rate_mode_){
	  case MAX:
		  // Get the abs max diff to determine the C
		  max_diff = caffe_gpu_amax(count, Pred_.gpu_diff(), 1);
		  // Calc the Threshold C
		  break;
	  case AVE:
		  // Calc the mean of the abs diff
		  caffe_gpu_asum(count, Pred_.gpu_diff(), &max_diff);
		  max_diff = max_diff / count;
		  break;
	  default:
		  LOG(FATAL) << "False c_rate_mode";
		  break;
  }
  Dtype C = fabs(max_diff * c_rate_);
  Dtype H = fabs(max_diff * h_rate_);

  Dtype* data_diff = Pred_.mutable_gpu_diff();
  // const Dtype* data_pred = bottom[0]->cpu_data();
  const int num = Pred_.num();
  const int channels = Pred_.channels();
  const int height = Pred_.height();
  const int width = Pred_.width();
  // The number of kernel is num * height, process a row each time
  const int num_kernels = num * height;
  // Set the bad_pixel_ buffer to zero
  Dtype* bad_pixel_data = bad_pixel_.mutable_gpu_data();
  caffe_gpu_set(bad_pixel_.count(), Dtype(0), bad_pixel_data);
  // Find the bad pixel and alter the diff
  Forward_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>(
		 num_kernels,
		 label_data,
		 data_diff,
		 bad_pixel_data,
		 num,
		 channels,
		 height,
		 width,
		 has_max_label_,
		 has_min_label_,
		 has_invalid_label_,
		 max_label_,
		 min_label_,
		 invalid_label_,
		 C,
		 has_h_rate_,
		 H);

  Dtype bad_pixel_count;
  caffe_gpu_asum(bad_pixel_.count(), bad_pixel_data, &bad_pixel_count);
  Dtype dot;
  caffe_gpu_dot(count, Pred_.gpu_diff(), Pred_.gpu_diff(), &dot);
  Dtype loss = dot / Dtype(2) / (count-bad_pixel_count);
  top->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void CrfUnaryLossLayer<Dtype>::Inference_gpu(const Blob<Dtype>* Z){
	const int dim = A_inv_.height();
	const Dtype* a_data = A_inv_.gpu_data();
	const Dtype* z_data = Z->gpu_data();
	Dtype* pred_data = Pred_.mutable_gpu_data();

	for (int n = 0; n < A_inv_.num(); n++){
		const Dtype* a_data_n = a_data + A_inv_.offset(n);
		for (int c = 0; c < Z->channels(); c++){
			const Dtype* z_data_nc = z_data + Z->offset(n, c);
			Dtype* pred_data_nc = pred_data + Pred_.offset(n, c);
			caffe_gpu_gemv(CblasNoTrans, dim, dim, Dtype(1), a_data_n, z_data_nc, Dtype(0), pred_data_nc);
		}
	}
}


template <typename Dtype>
void CrfUnaryLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top){
  // Generate the R
  Calc_R_gpu(bottom[2]);
  // Calc the A matrix
  Calc_A_gpu();
  // Calc the A_inv matrix
  // ---- DEBUG -----
  // test the speed
  // time_t start, end;
  //runstart = clock();
  Calc_A_inv_gpu();
  //end = clock();
  // LOG(INFO)<<"TIME: "<<(Dtype)(end-start)/CLOCKS_PER_SEC;
// ---- DEBUG -----
  // ----DEBUG ----
  /*
  Blob<Dtype> tmp;
  int dim = A_.height();
  tmp.Reshape(1,1,A_.height(), A_.width());
  for (int n = 0; n < A_.num(); n++){
	  caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, dim, dim, dim, Dtype(1), A_.cpu_data()+A_.offset(n), A_inv_.cpu_data()+A_inv_.offset(n), Dtype(0), tmp.mutable_cpu_data());

	  LOG(INFO)<<"Det(S): "<<caffe_cpu_det(A_.height(), tmp.cpu_data());
  }
  */


  // Inference the crf
  Inference_gpu(bottom[0]);
  // Copy the result if needed
  if (top.size() == 2){
	  caffe_copy(Pred_.count(), Pred_.gpu_data(), top[1]->mutable_gpu_data());
  }
  // Calc the loss according to the Pred_ and the bottom[0]
  switch (unary_mode_){
	  case L2:
		  Euclidean_loss_gpu(bottom[1], top[0]);
		  break;
	  case Berhu:
		  Berhu_loss_gpu(bottom[1], top[0]);
		  break;
	  default:
		  LOG(FATAL)<<"Unknow unary_mode_ in CrfUnaryLossLayer";
		  break;
  }
}


template <typename Dtype>
void CrfUnaryLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	// The BP will performed on the bottom[0] and bottom[2]
	const Dtype loss_weight = top[0]->cpu_diff()[0];
	Dtype beta;
	if (normalize_){
		beta = loss_weight / bottom[0]->count();
	}else{
		beta = loss_weight / bottom[0]->num();
	}

	// BP for bottom[0]

	caffe_gpu_axpby(
			bottom[0]->count(),
			beta,
			Pred_.gpu_diff(),
			Dtype(0),
			bottom[0]->mutable_gpu_diff());

}


template <typename Dtype>
void CrfUnaryLossLayer<Dtype>::Calc_A_inv_gpu(void){
	const int num = A_.num();
	const Dtype* a_data = A_.gpu_data();
	Dtype* a_inv_data = A_inv_.mutable_gpu_data();
	const int height = A_.height();

	caffe_gpu_inv(height, num, a_data, a_inv_data);
}

template void CrfUnaryLossLayer<float>::Calc_A_gpu(void);
template void CrfUnaryLossLayer<double>::Calc_A_gpu(void);

template void CrfUnaryLossLayer<float>::Calc_R_gpu(const Blob<float>* bottom);
template void CrfUnaryLossLayer<double>::Calc_R_gpu(const Blob<double>* bottom);

template void CrfUnaryLossLayer<float>::Euclidean_loss_gpu(const Blob<float>* gt, Blob<float>* top);
template void CrfUnaryLossLayer<double>::Euclidean_loss_gpu(const Blob<double>* gt, Blob<double>* top);

template void CrfUnaryLossLayer<float>::Berhu_loss_gpu(const Blob<float>* gt, Blob<float>* top);
template void CrfUnaryLossLayer<double>::Berhu_loss_gpu(const Blob<double>* gt, Blob<double>* top);

template void CrfUnaryLossLayer<float>::Inference_gpu(const Blob<float>* Z);
template void CrfUnaryLossLayer<double>::Inference_gpu(const Blob<double>* Z);

template void CrfUnaryLossLayer<float>::Calc_A_inv_gpu(void);
template void CrfUnaryLossLayer<double>::Calc_A_inv_gpu(void);

INSTANTIATE_LAYER_GPU_FUNCS(CrfUnaryLossLayer);

}  // namespace caffe
