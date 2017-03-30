#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/crf3_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void Crf3LossLayer<Dtype>::LayerSetUp(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
	LossLayer<Dtype>::LayerSetUp(bottom, top);
	// Parse the param
	Crf3LossParameter crf_param = this->layer_param_.crf3_loss_param();

	if (crf_param.has_disp_w()){
		disp_w_ = crf_param.disp_w();
	}else{
		disp_w_ = 0;
	}

	if (crf_param.has_unary()){
		switch (crf_param.unary()){
			case CrfLossParameter_UnaryDist_L2:
				unary_mode_ = L2;
				break;
			default:
				unary_mode_ = L2;
		}
	}else{
		unary_mode_ = L2;
	}

	if(crf_param.has_pairwise_lr()){
		pairwise_lr_ = crf_param.pairwise_lr();
	}else{
		pairwise_lr_ = 1.0;
	}
	
	// Init the param blobs
	// Only weight, no bias
	this->blobs_.resize(2);
	// Gen the weight shape for the weight
	vector<int> weight_shape(4);
	weight_shape[0] = 1;
	weight_shape[1] = 1;
	weight_shape[2] = 1;
	weight_shape[3] = 1;
	this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
	this->blobs_[1].reset(new Blob<Dtype>(weight_shape));

	if(crf_param.has_alpha()){
		this->blobs_[0]->mutable_cpu_data()[0] = crf_param.alpha();
	}else{
		this->blobs_[0]->mutable_cpu_data()[0] = 1;
	}

	if(crf_param.has_beta()){
		this->blobs_[1]->mutable_cpu_data()[0] = crf_param.beta();
	}else{
		this->blobs_[1]->mutable_cpu_data()[0] = 1;
	}

	this->param_propagate_down_.resize(this->blobs_.size(), true);
	// Check the input
	// The bottom[0] indicate the unary potential
	// The bottom[1] indicate the ground truth
	// The bottom[2] indicate the pairwise potential matrix
	CHECK_EQ(bottom[0]->count(), bottom[1]->count()) << "The size of the bottom[0](prediction) and the bottom[1](ground truth) must be the same";
	CHECK_EQ(bottom[0]->count(2), bottom[2]->height());
	CHECK_EQ(bottom[2]->height(), bottom[2]->width()) << "The height and width of the pairwise potential must be the same";
	CHECK_LE(top.size(), 2) << "Only support at most 2 top blobs";
	CHECK_EQ(bottom[2]->channels(), 1) << "Currently, only support the pairwise with 1 channel";
}

template <typename Dtype>
void Crf3LossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";

  // Reshape the top[1]
  if(top.size()==2){
	  top[1]->ReshapeLike(*bottom[0]);
  }
  // Reshape the internal param
  // The A_ is a positive defination matrix with shape [n, 1, h, w]
  A_.Reshape(bottom[2]->num(), 1, bottom[2]->height(), bottom[2]->width());
  A_inv_.ReshapeLike(A_);
  R_.ReshapeLike(A_);
  Pred_.ReshapeLike(*bottom[0]);
  J_.ReshapeLike(*bottom[2]);
}

template <typename Dtype>
void Crf3LossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // First, normalize the weight if needed
  if (this->phase_ == TRAIN){
	  Normalize_weight_cpu();
  }
  // Generate the R
  Calc_R_cpu(bottom[2]);
  // Calc the A matrix
  Calc_A_cpu();
  // Calc the A_inv matrix
  Calc_A_inv_cpu();
  /*
  // ---- DEBUG----
  // Check A and A_inv
  DLOG(INFO)<<"Det(A): "<<caffe_cpu_det(A_.height(), A_.cpu_data()+A_.offset(0));
  DLOG(INFO)<<"Det(A_inv): "<<caffe_cpu_det(A_inv_.height(), A_inv_.cpu_data()+A_.offset(0));
  Blob<Dtype> tmp;
  int dim = A_.height();
  tmp.Reshape(1,1,A_.height(), A_.width());
  caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, dim, dim, dim, Dtype(1), A_.cpu_data(), A_inv_.cpu_data(), Dtype(0), tmp.mutable_cpu_data());

  DLOG(INFO)<<"Det(S): "<<caffe_cpu_det(A_.height(), tmp.cpu_data()+tmp.offset(0));
  for (int i = 0; i < A_.height(); i++){
	  int idx = A_.offset(0, 0, i, i);
	  DLOG(INFO)<<"Tr(A): "<<A_.cpu_data()[idx]<<" Tr(A_inv): "<<A_inv_.cpu_data()[idx] <<"Tr(S): "<<tmp.cpu_data()[idx];
  }
  */
  // ---- DEBUG -----
  // Print the weight of the pairwise
  if (this->phase_ == TRAIN && disp_w_ != 0 && counter_++ % disp_w_ == 0){
	  const Dtype* weight = this->blobs_[0]->cpu_data();
	  LOG(INFO)<<"Alpha: "<<weight[0]<<" Beta: "<<this->blobs_[1]->cpu_data()[0];
  }

  // Inference the crf
  Inference_cpu(bottom[0]);
  // Copy the result if needed
  if (top.size() == 2){
	  caffe_copy(Pred_.count(), Pred_.cpu_data(), top[1]->mutable_cpu_data());
  }
  // Calc the loss according to the Pred_ and the bottom[0]
  // The diff will be stored in Pred_.diff
  Euclidean_loss_cpu(bottom[1], top[0]);
}

template <typename Dtype>
void Crf3LossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	// The BP will performed on the bottom[0] and bottom[2]
	const Dtype loss_weight = top[0]->cpu_diff()[0];
	const Dtype beta = loss_weight / bottom[0]->num();


	// BP for bottom[0]
	caffe_cpu_axpby(
			bottom[0]->count(),
			beta,
			Pred_.cpu_diff(),
			Dtype(0),
			bottom[0]->mutable_cpu_diff());

	// BP for bottom[2]
	// Calc the J matrix
	Calc_J_cpu(bottom[2]);

	// Calc the pairwise diff
	Pairwise_BP_cpu(bottom[1]);
}

template <typename Dtype>
void Crf3LossLayer<Dtype>::Calc_R_cpu(const Blob<Dtype>* bottom){
	// Calc the pairwise R matrix according to the bottom
	const Dtype* alpha_data = this->blobs_[0]->cpu_data();
	const Dtype* beta_data = this->blobs_[1]->cpu_data();
	const Dtype* bottom_data = bottom->cpu_data();

	Dtype* r_data = R_.mutable_cpu_data();
	const int dim = bottom->height() * bottom->width();
	const int count = R_.count();
	caffe_set(count, Dtype(0), r_data);

	for (int n = 0; n < bottom->num(); n++){
		Dtype* r_data_n = r_data + R_.offset(n);
		// Only support the pairwise map with single channel
		const Dtype* bottom_data_n = bottom_data + bottom->offset(n);
		caffe_axpy(dim, -beta_data[0], bottom_data_n, r_data_n);
	}
	// Calc the exp
	caffe_exp(count, r_data, r_data);
	// Mult with the beta
	caffe_scal(count, alpha_data[0], r_data);
}

template <typename Dtype>
void Crf3LossLayer<Dtype>::Calc_A_cpu(void){
	// A = I + D -R
	// Set A to 0
	caffe_set(A_.count(), Dtype(0), A_.mutable_cpu_data());
	const int num = A_.num();
	const int channels = A_.channels();
	const int height = A_.height();
	const int width = A_.width();
	const Dtype* r_data = R_.cpu_data();
	Dtype* a_data = A_.mutable_cpu_data();
	// A = I + D
	for (int n = 0; n < num; n++){
		for(int c = 0; c < channels; c++){
			// Here the channel should be 1
			// And the height should equal to width
			for (int h = 0; h < height; h++){
				const int w = h;
				const int index = ((n*channels+c)*height+h)*width;
				const Dtype d = caffe_cpu_asum(width, r_data+index);
				a_data[index+w] = d + 1;
			}
		}
	}
	// A = A - R
	caffe_axpy(A_.count(), Dtype(-1), r_data, a_data);
}

template <typename Dtype>
void Crf3LossLayer<Dtype>::Inference_cpu(const Blob<Dtype>* Z){
	const int dim = A_inv_.height();
	const Dtype* a_data = A_inv_.cpu_data();
	const Dtype* z_data = Z->cpu_data();
	Dtype* pred_data = Pred_.mutable_cpu_data();

	for (int n = 0; n < A_inv_.num(); n++){
		const Dtype* a_data_n = a_data + A_inv_.offset(n);
		for (int c = 0; c < Z->channels(); c++){
			const Dtype* z_data_nc = z_data + Z->offset(n, c);
			Dtype* pred_data_nc = pred_data + Pred_.offset(n, c);
			caffe_cpu_symv(dim, Dtype(1), a_data_n, z_data_nc, Dtype(0), pred_data_nc);
		}
	}
}

template <typename Dtype>
void Crf3LossLayer<Dtype>::Pairwise_BP_cpu(const Blob<Dtype>* gt){

	const Dtype* a_data = A_inv_.cpu_data();
	const Dtype* y_data = gt->cpu_data();
	const Dtype* Pred_data = Pred_.cpu_data();
	const Dtype* j_data = J_.cpu_data();
	const int dim = A_inv_.height();
	const int num = A_inv_.num();
	const int dat_channels = gt->channels();
	Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
	// Set the diff to 0
	caffe_set(this->blobs_[0]->count(), Dtype(0), weight_diff);

	Blob<Dtype> tmp_blob;
	tmp_blob.Reshape(1, dim, 1, 1);
	Dtype* buf_data = tmp_blob.mutable_cpu_data();

	Blob<Dtype> tmp2_blob;
	tmp2_blob.Reshape(1, 1, dim, dim);
	Dtype* buf2_data = tmp2_blob.mutable_cpu_data();

	for (int n = 0; n < num; n++){
		const Dtype* a_data_n = a_data + A_inv_.offset(n);
		// Iter the data channel first
		for (int ch = 0; ch < dat_channels; ch++){
			const Dtype* Pred_data_nc = Pred_data + Pred_.offset(n, ch);
			const Dtype* y_data_nc = y_data + gt->offset(n, ch);

			for (int c = 0; c < J_.channels(); c++){
				const Dtype* j_data_nc = j_data + J_.offset(n, c);
				Dtype val = 0;

				// yJy
				caffe_cpu_gemv(CblasNoTrans, dim, dim, Dtype(1), j_data_nc, y_data_nc, Dtype(0), buf_data);
				val = caffe_cpu_dot(dim, buf_data, y_data_nc);
				// zAJAz
				// Az = Pred
				// caffe_cpu_symv(dim, Dtype(1), a_data_n, z_data_n, Dtype(0), buf_data);
				// NOTE: Here can not use the in-place operation
				caffe_cpu_gemv(CblasNoTrans, dim, dim, Dtype(1), j_data_nc, Pred_data_nc, Dtype(0), buf_data);
				// Since A_int_ is symmetric, ztA = (Az)t
				// caffe_cpu_symv(dim, Dtype(1), a_data_n, buf_diff, Dtype(0), buf_data);
				val -= caffe_cpu_dot(dim, buf_data, Pred_data_nc);
				// Calc the trace
				if(partition_){
					// AJ
					caffe_cpu_symm(CblasLeft, CblasUpper, dim, dim, Dtype(1), a_data_n, j_data_nc, Dtype(0), buf2_data);
					Dtype tr = 0;
					for (int i = 0; i < dim; i++){
						tr += buf2_data[i+i*dim];
					}
					weight_diff[c] += val - tr / 2;
				}else{
					weight_diff[c] += val;
				}
			}
		}
	}
	// Since may have multi channels, average the diff
	// Apply the pairwose_lr_ and the channel number
	this->blobs_[0]->scale_diff(pairwise_lr_ / Dtype(num) / Dtype(dat_channels));
}

template <typename Dtype>
void Crf3LossLayer<Dtype>::Calc_J_cpu(const Blob<Dtype>* bottom){
	const Dtype* bottom_data = bottom->cpu_data();
	Dtype* j_data = J_.mutable_cpu_data();
	caffe_cpu_axpby(J_.count(), Dtype(-1), bottom_data, Dtype(0), j_data);

	for (int n = 0; n < J_.num(); n++){
		for (int c = 0; c < J_.channels(); c++){
			for (int h = 0; h < J_.height(); h++){
				const int index = J_.offset(n, c, h, h);
				const int b_index = J_.offset(n, c, h);
				// TODO: Here is a sum of ABS value, that may cause a problem
				Dtype sum = caffe_cpu_asum(J_.width(), j_data + b_index);
				j_data[index] += sum;
			}
		}
	}
}


template <typename Dtype>
void Crf3LossLayer<Dtype>::Normalize_weight_cpu(void){
	// Normalize the weight if needed
	/*
	if (normalize_ == false && positive_ == false){
		return;
	}
	Dtype* weight_data = this->blobs_[0]->mutable_cpu_data();
	const int count = this->blobs_[0]->count();
	if (positive_){
		// Set all negetive val to 0
		for (int i = 0; i < count; i++){
			if (weight_data[i] < 0){
				weight_data[i] = 0;
			}
		}
	}
	if (normalize_){
		// Calc the sum of the blob
		const Dtype sum = this->blobs_[0]->asum_data();
		this->blobs_[0]->scale_data(norm_val_ / sum);
	}
	*/

	// Keep the alpha >= 0
	if (this->blobs_[0]->cpu_data()[0] < 0){
		this->blobs_[0]->mutable_cpu_data()[0] = 0;
	}
}

template <typename Dtype>
void Crf3LossLayer<Dtype>::Calc_A_inv_cpu(void){
	// Calc the A_inv_ according to A_
	CHECK_EQ(A_.height(), A_.width());
	const Dtype* a_data = A_.cpu_data();
	Dtype* a_inv_data = A_inv_.mutable_cpu_data();
	const int dim = A_.height();

	for (int n = 0; n < A_.num(); n++){
		for (int c = 0; c < A_.channels(); c++){
			const Dtype* a_data_n = a_data + A_.offset(n, c);
			Dtype* a_inv_data_n = a_inv_data + A_inv_.offset(n, c);
			caffe_cpu_inv_sympd(dim, a_data_n, a_inv_data_n);
		}
	}
}

template <typename Dtype>
void Crf3LossLayer<Dtype>::Euclidean_loss_cpu(const Blob<Dtype>* gt, Blob<Dtype>* top){
	const Dtype* gt_data = gt->cpu_data();
	const Dtype* pred_data = Pred_.cpu_data();
	const int count = Pred_.count();
	caffe_sub(
			count,
			pred_data,
			gt_data,
			Pred_.mutable_cpu_diff());
	Dtype dot = caffe_cpu_dot(count, Pred_.cpu_diff(), Pred_.cpu_diff());
	Dtype loss = dot / count / Dtype(2);
	top->mutable_cpu_data()[0] = loss;
}

#ifdef CPU_ONLY
STUB_GPU(Crf3LossLayer);
#endif

INSTANTIATE_CLASS(Crf3LossLayer);
REGISTER_LAYER_CLASS(Crf3Loss);

}  // namespace caffe
