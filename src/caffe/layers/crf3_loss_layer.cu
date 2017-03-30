#include <vector>

#include "caffe/layers/crf3_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

	
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
__global__ void calc_j_gpu_kernel_part1(const int n,
	   const Dtype* const bottom_data, const Dtype* const r_data, const int num,
	   const int channels, const int height, const int width, 
	   const Dtype gamma, Dtype* j_data){
	CUDA_KERNEL_LOOP(index, n) {
		const int n_idx = index / height / channels;
		const int c = (index / height) % channels;
		const int h = index % height;
		const int idx = ((n_idx*channels+c)*height+h)*width;
		const int r_idx = (n_idx*height+h)*width;

		for (int w = 0; w < width; w++){
			j_data[idx+w] = gamma * bottom_data[idx+w] * r_data[r_idx+w];
		}
	}
}


template <typename Dtype>
__global__ void calc_j_gpu_kernel_part2(const int n,
	   const int num, const int channels, const int height,
	   const int width, Dtype* j_data){
	CUDA_KERNEL_LOOP(index, n) {
		const int n_idx = index / height / channels;
		const int c = (index / height) % channels;
		const int h = index % height;
		const int idx = ((n_idx*channels+c)*height+h)*width;
		Dtype sum = 0;

		for (int i = 0; i < width; i++){
			sum += j_data[idx+i];
		}
		j_data[idx+h] -= sum;
	}
}


template <typename Dtype>
void Crf3LossLayer<Dtype>::Calc_A_gpu(void){
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
void Crf3LossLayer<Dtype>::Calc_R_gpu(const Blob<Dtype>* bottom){
	const Dtype* alpha_data = this->blobs_[0]->cpu_data();
	const Dtype* beta_data = this->blobs_[1]->cpu_data();
	const Dtype* bottom_data = bottom->gpu_data();
	Dtype* r_data = R_.mutable_gpu_data();
	const int dim = bottom->height() * bottom->width();
	const int count = R_.count();
	caffe_gpu_set(count, Dtype(0), r_data);

	for (int n = 0; n < bottom->num(); n++){
		Dtype* r_data_n = r_data + R_.offset(n);
		const Dtype* bottom_data_n = bottom_data + bottom->offset(n);
		caffe_gpu_axpy(dim, -beta_data[0], bottom_data_n, r_data_n);
	}
	// Calc the exp
	caffe_gpu_exp(count, r_data, r_data);
	caffe_gpu_scal(count, alpha_data[0], r_data);
}

template <typename Dtype>
void Crf3LossLayer<Dtype>::Euclidean_loss_gpu(const Blob<Dtype>* gt, Blob<Dtype>* top){
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
void Crf3LossLayer<Dtype>::Inference_gpu(const Blob<Dtype>* Z){
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
void Crf3LossLayer<Dtype>::Calc_J_gpu(const Blob<Dtype>* bottom){
	const Dtype* bottom_data = bottom->gpu_data();
	Dtype* j_data = J_.mutable_gpu_data();
	const int num = J_.num();
	const int channels = J_.channels();
	const int height = J_.height();
	const int width = J_.width();

	caffe_gpu_axpby(J_.count(), Dtype(-1), bottom_data, Dtype(0), j_data);
	// num_kernels: n*c*h
	const int num_kernels = num * channels * height;
	// Calc the first part
	// calc_j_gpu_kernel_part1<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
	//	CAFFE_CUDA_NUM_THREADS>>>(num_kernels, bottom_data, r_data, num, channels, height, width, gamma_, j_data);
	// CUDA_POST_KERNEL_CHECK;
	// Calc the second part
	calc_j_gpu_kernel_part2<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
		CAFFE_CUDA_NUM_THREADS>>>(num_kernels, num, channels, height, width, j_data);
	CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void Crf3LossLayer<Dtype>::Pairwise_BP_gpu(const Blob<Dtype>* gt){
	const Dtype* a_data = A_inv_.gpu_data();
	const Dtype* y_data = gt->gpu_data();
	const Dtype* Pred_data = Pred_.gpu_data();
	const Dtype* j_data = J_.gpu_data();
	const int dim = A_inv_.height();
	const int num = A_inv_.num();
	const int dat_channels = gt->channels(); 
	Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
	// Set the diff to 0
	caffe_set(this->blobs_[0]->count(), Dtype(0), weight_diff);

	Blob<Dtype> tmp_blob;
	tmp_blob.Reshape(1, dim, 1, 1);
	Dtype* buf_data = tmp_blob.mutable_gpu_data();

	Blob<Dtype> tmp2_blob;
	tmp2_blob.Reshape(1, 1, dim, dim);
	Dtype* buf2_data = tmp2_blob.mutable_gpu_data();


	for (int n = 0; n < num; n++){
		const Dtype* a_data_n = a_data + A_inv_.offset(n);
		for (int ch = 0; ch < dat_channels; ch++){
		const Dtype* y_data_nc = y_data + gt->offset(n, ch);
		const Dtype* Pred_data_nc = Pred_data + Pred_.offset(n, ch);

			for (int c = 0; c < J_.channels(); c++){
				const Dtype* j_data_nc = j_data + J_.offset(n, c);
				Dtype val = 0;

				// yJy
				caffe_gpu_gemv(CblasNoTrans, dim, dim, Dtype(1), j_data_nc, y_data_nc, Dtype(0), buf_data);
				caffe_gpu_dot(dim, buf_data, y_data_nc, &val);
				// zAJAz
				// Az = Pred
				// caffe_gpu_gemv(CblasNoTrans, dim, dim, Dtype(1), a_data_n, z_data_n, Dtype(0), buf_data);
				// NOTE: Here can not use the in-place operation
				caffe_gpu_gemv(CblasNoTrans, dim, dim, Dtype(1), j_data_nc, Pred_data_nc, Dtype(0), buf_data);
				// caffe_gpu_gemv(CblasNoTrans, dim, dim, Dtype(1), a_data_n, buf_diff, Dtype(0), buf_data);
				Dtype val2 = 0;
				caffe_gpu_dot(dim, buf_data, Pred_data_nc, &val2);
				val -= val2;
				// Calc the trace
				if(partition_){
					// AJ
					caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, dim, dim, dim, Dtype(1), a_data_n, j_data_nc, Dtype(0), buf2_data);
					Dtype tr = 0;
					const Dtype* buf2_data_cpu = tmp2_blob.cpu_data();
					for (int i = 0; i < dim; i++){
						tr += buf2_data_cpu[i+i*dim];
					}
					weight_diff[c] += val - tr / 2;
				}else{
					weight_diff[c] += val;
				}
			}
		}
	}
	// Apply the beta
	this->blobs_[0]->scale_diff(pairwise_lr_ / Dtype(num) / Dtype(dat_channels));
}

template <typename Dtype>
void Crf3LossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top){
  // First, normalize the weight if needed
  if (this->phase_ == TRAIN){
	  Normalize_weight_cpu();
  }
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
  // Print the weight of the pairwise
  if (this->phase_ == TRAIN && disp_w_ != 0 && counter_++ % disp_w_ == 0){
	  const Dtype* weight = this->blobs_[0]->cpu_data();
	  LOG(INFO)<<"Alpha: "<<weight[0]<<" Beta: "<<this->blobs_[1]->cpu_data()[0];
  }

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
  Euclidean_loss_gpu(bottom[1], top[0]);
}


template <typename Dtype>
void Crf3LossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	// The BP will performed on the bottom[0] and bottom[2]
	const Dtype loss_weight = top[0]->cpu_diff()[0];
	const Dtype beta = loss_weight / bottom[0]->num();

	// BP for bottom[0]

	caffe_gpu_axpby(
			bottom[0]->count(),
			beta,
			Pred_.gpu_diff(),
			Dtype(0),
			bottom[0]->mutable_gpu_diff());

	// BP for bottom[2]
	// Calc the J matrix
	Calc_J_gpu(bottom[2]);

	// Calc the pairwise diff
	Pairwise_BP_gpu(bottom[1]);
}


template <typename Dtype>
void Crf3LossLayer<Dtype>::Calc_A_inv_gpu(void){
	const int num = A_.num();
	const Dtype* a_data = A_.gpu_data();
	Dtype* a_inv_data = A_inv_.mutable_gpu_data();
	const int height = A_.height();

	caffe_gpu_inv(height, num, a_data, a_inv_data);
}

template void Crf3LossLayer<float>::Calc_A_gpu(void);
template void Crf3LossLayer<double>::Calc_A_gpu(void);

template void Crf3LossLayer<float>::Calc_R_gpu(const Blob<float>* bottom);
template void Crf3LossLayer<double>::Calc_R_gpu(const Blob<double>* bottom);

template void Crf3LossLayer<float>::Euclidean_loss_gpu(const Blob<float>* gt, Blob<float>* top);
template void Crf3LossLayer<double>::Euclidean_loss_gpu(const Blob<double>* gt, Blob<double>* top);

template void Crf3LossLayer<float>::Inference_gpu(const Blob<float>* Z);
template void Crf3LossLayer<double>::Inference_gpu(const Blob<double>* Z);

template void Crf3LossLayer<float>::Calc_J_gpu(const Blob<float>* bottom);
template void Crf3LossLayer<double>::Calc_J_gpu(const Blob<double>* bottom);

template void Crf3LossLayer<float>::Pairwise_BP_gpu(const Blob<float>* gt);
template void Crf3LossLayer<double>::Pairwise_BP_gpu(const Blob<double>* gt);

template void Crf3LossLayer<float>::Calc_A_inv_gpu(void);
template void Crf3LossLayer<double>::Calc_A_inv_gpu(void);

INSTANTIATE_LAYER_GPU_FUNCS(Crf3LossLayer);

}  // namespace caffe
