#include <vector>

#include "caffe/layers/crf_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void CrfLossLayer<Dtype>::LayerSetUp(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
	LossLayer<Dtype>::LayerSetUp(bottom, top);
	// Parse the param
	CrfLossParameter crf_param = this->layer_param_.crf_loss_param();
	if (crf_param.has_alpha()){
		alpha_ = crf_param.alpha();
	}else{
		alpha_ = 1.0;
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
	// Check the input
	// The bottom[0] indicate the unary potential
	// The bottom[1] indicate the ground truth
	// The bottom[2] indicate the pairwise potential matrix
	channels_ = bottom[0]->channels();
	CHECK_EQ(bottom[0]->count(), bottom[1]->count()) << "The size of the bottom[0](prediction) and the bottom[1](ground truth) must be the same";
	CHECK_EQ(bottom[0]->count(1), bottom[2]->height());
	CHECK_EQ(bottom[2]->channels(), 1) << "Currently, the channels of the bottom[2] (pairwise potential) must be 1";
	CHECK_EQ(bottom[2]->height(), bottom[2]->width()) << "The height and width of the pairwise potential must be the same";
	CHECK_LE(top.size(), 2) << "Only support at most 2 top blobs";
}

template <typename Dtype>
void CrfLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  diff_.ReshapeLike(*bottom[0]);

  // Reshape the top[1]
  if(top.size()==2){
	  top[1]->ReshapeLike(*bottom[0]);
  }
  // Reshape the internal param
  // The A_ is a positive defination matrix with shape [n, 1, h, w]
  A_.ReshapeLike(*bottom[2]);
  A_inv_.ReshapeLike(A_);
  // Reshape the energy
  E_.Reshape(bottom[0]->num(), 1, 1, 1);
  // Reshape the probabilty
  prob_.Reshape(bottom[0]->num(), 1, 1, 1);
}

template <typename Dtype>
void CrfLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // Calc the A matrix
  Calc_A_cpu(bottom[2]);
  // Calc the average prob as the loss
  // The A_inv_ will also be calculated in this function
  top[0]->mutable_cpu_data()[0] = Calc_prob_cpu(&A_, bottom[1], bottom[0], &A_inv_, &prob_);
  // Check if need to do the inference process
  if (top.size() > 1){
	  // Indicate the top[1] is the output of the inference
	  Inference_cpu(&A_inv_, bottom[0], top[1]);
  }
}

template <typename Dtype>
void CrfLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	// The BP will performed on the bottom[0] and bottom[2]
	const int dim = A_inv_.height();
	const Dtype* a_data = A_inv_.cpu_data();
	const Dtype* z_data = bottom[0]->cpu_data();
	const Dtype* y_data = bottom[1]->cpu_data();
	Dtype* z_diff = bottom[0]->mutable_cpu_diff();
	const Dtype loss_weight = top[0]->cpu_diff()[0];

	// BP for bottom[0]
	for (int n = 0; n < bottom[0]->num(); n++){
		const Dtype* a_data_n = a_data + A_inv_.offset(n);
		const Dtype* z_data_n = z_data + bottom[0]->offset(n);
		const Dtype* y_data_n = y_data + bottom[1]->offset(n);
		Dtype* z_diff_n = z_diff + bottom[0]->offset(n);
		caffe_cpu_symv(dim, Dtype(1), a_data_n, z_data_n, Dtype(0), z_diff_n);
		caffe_sub(dim, z_diff_n, y_data_n, z_diff_n);
		caffe_scal(dim, Dtype(2)*loss_weight, z_diff_n);
	}
	// BP for bottom[2]
	BP_pairwise_cpu(&A_inv_, bottom[1], bottom[0], bottom[2]);
	// Apply loss weight for bottom[2]
	caffe_scal(bottom[2]->count(), loss_weight, bottom[2]->mutable_cpu_diff());
}

template <typename Dtype>
void CrfLossLayer<Dtype>::Calc_A_cpu(const Blob<Dtype>* r){
	// A = I + D -R
	// Set A to 0
	caffe_set(A_.count(), Dtype(0), A_.mutable_cpu_data());
	const int num = A_.num();
	const int channels = A_.channels();
	const int height = A_.height();
	const int width = A_.width();
	const Dtype* r_data = r->cpu_data();
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
	caffe_cpu_axpby(A_.count(), Dtype(-1), r_data, Dtype(1), a_data);
}

template <typename Dtype>
Dtype CrfLossLayer<Dtype>::Calc_prob_cpu(const Blob<Dtype>* A, const Blob<Dtype>* Y, const Blob<Dtype>* Z, Blob<Dtype>* A_inv, Blob<Dtype>* prob){
	// This function calc the mean probabilty of the crf of the current minibatch
	const Dtype PI = 3.1415926;
	const int dim = A->height();
	const Dtype* y_data = Y->cpu_data();
	const Dtype* a_data = A->cpu_data();
	const Dtype* z_data = Z->cpu_data();
	Dtype* prob_data = prob->mutable_cpu_data();
	Dtype* a_inv_data = A_inv->mutable_cpu_data();

	Blob<Dtype> tmp_blob;
	tmp_blob.Reshape(1, dim, 1, 1);
	Dtype* tmp_data = tmp_blob.mutable_cpu_data();
	Dtype PI_n2 = std::pow(PI, dim);
	PI_n2 = std::sqrt(PI_n2);

	Dtype accu = 0;
	for (int n = 0; n < prob->num(); n++){
		Dtype rst = 0;
		const Dtype* a_data_n = a_data + A->offset(n);
		const Dtype* y_data_n = y_data + Y->offset(n);
		const Dtype* z_data_n = z_data + Z->offset(n);
		Dtype* a_inv_data_n = a_inv_data + A_inv->offset(n);
		Dtype* prob_data_n = prob_data + prob->offset(n);

		caffe_cpu_symv(dim, Dtype(1), a_data_n, y_data_n, Dtype(0), tmp_data);
		rst = -caffe_cpu_dot(dim, tmp_data, y_data_n);

		rst += 2.0*caffe_cpu_dot(dim, z_data_n, y_data_n);
		// Calc the inverse of the A
		caffe_cpu_inv_sympd(dim, a_data_n, a_inv_data_n);
		caffe_cpu_symv(dim, Dtype(1), a_inv_data_n, z_data_n, Dtype(0), tmp_data);
		rst -= caffe_cpu_dot(dim, tmp_data, z_data_n);
		// Calc the exp
		rst = std::exp(rst);
		// Calc the det of A
		rst *= std::sqrt(caffe_cpu_det(dim, a_data_n)) / PI_n2;
		prob_data_n[0] = rst;
		accu += rst;
	}
	return accu / prob_.num();
}

template <typename Dtype>
void CrfLossLayer<Dtype>::Inference_cpu(const Blob<Dtype>* A_inv, const Blob<Dtype>* Z, Blob<Dtype>* Pred){
	const int dim = A_inv->height();
	const Dtype* a_data = A_inv->cpu_data();
	const Dtype* z_data = Z->cpu_data();
	Dtype* pred_data = Pred->mutable_cpu_data();

	for (int n = 0; n < A_inv->num(); n++){
		const Dtype* a_data_n = a_data + A_inv->offset(n);
		const Dtype* z_data_n = z_data + Z->offset(n);
		Dtype* pred_data_n = pred_data + Pred->offset(n);
		caffe_cpu_symv(dim, Dtype(1), a_data_n, z_data_n, Dtype(0), pred_data_n);
	}
}

template <typename Dtype>
Dtype CrfLossLayer<Dtype>::BP_r_cpu(const int n, const Blob<Dtype>* A_inv, const Blob<Dtype>* Y, const Blob<Dtype>* Z, const Blob<Dtype>* J, Blob<Dtype>* buf_blob){
	const Dtype* a_data = A_inv->cpu_data() + A_inv->offset(n);
	const Dtype* y_data = Y->cpu_data() + Y->offset(n);
	const Dtype* z_data = Z->cpu_data() + Z->offset(n);
	const Dtype* j_data = J->cpu_data();
	const int dim = A_inv->height();

	Blob<Dtype> tmp_blob;
	if (buf_blob == NULL){
		tmp_blob.Reshape(1, dim, 1, 1);
		buf_blob = &tmp_blob;
	}
	Dtype* buf_data = buf_blob->mutable_cpu_data();

	Blob<Dtype> tmp2_blob;
	tmp2_blob.Reshape(1, 1, dim, dim);
	Dtype* buf2_data = tmp2_blob.mutable_cpu_data();

	Dtype val = 0;
	// yJy
	caffe_cpu_gemv(CblasNoTrans, dim, dim, Dtype(1), j_data, y_data, Dtype(0), buf_data);
	val = caffe_cpu_dot(dim, buf_data, y_data);
	// zAJAz
	caffe_cpu_symv(dim, Dtype(1), a_data, z_data, Dtype(0), buf_data);
	// TODO: in-place operation, might cause a problem
	caffe_cpu_gemv(CblasNoTrans, dim, dim, Dtype(1), j_data, buf_data, Dtype(0), buf_data);
	caffe_cpu_symv(dim, Dtype(1), a_data, buf_data, Dtype(0), buf_data);
	val -= caffe_cpu_dot(dim, buf_data, z_data);
	// AJ
	caffe_cpu_symm(CblasLeft, CblasUpper, dim, dim, Dtype(1), a_data, j_data, Dtype(0), buf2_data);
	// Calc the trace
	Dtype tr = 0;
	for (int i = 0; i < dim; i++){
		tr += buf2_data[i+i*dim];
	}
	return val - tr / 2;
}

template <typename Dtype>
void CrfLossLayer<Dtype>::get_J_cpu(const int h, const int w, Blob<Dtype>* J){
	const int height = J->height();
	const int width = J->width();
	CHECK_LT(w, width);
	CHECK_LT(h, height);
	CHECK(w!=h);

	Dtype* j_data = J->mutable_cpu_data();
	caffe_set(J->count(), Dtype(0), j_data);

	j_data[h*width + w] = Dtype(-1);
	j_data[h*width + h] = Dtype(1);
}

template <typename Dtype>
void CrfLossLayer<Dtype>::BP_pairwise_cpu(const Blob<Dtype>* A_inv, const Blob<Dtype>* Y, const Blob<Dtype>* Z, Blob<Dtype>* Out){
	CHECK_EQ(A_inv->channels(), 1);
	const int dim = A_inv->height();
	const int num = A_inv->num();
	Dtype* out_data = Out->mutable_cpu_data();

	Blob<Dtype> J;
	J.Reshape(1, 1, dim, dim);

	for (int h = 0; h < dim; h++){
		for (int w = 0; w < h; w++){
			// Generate the J for current position
			get_J_cpu(h, w, &J);
			for (int n = 0; n < num; n++){
				const int index = (n*dim+h)*dim + w;
				const int index2 = (n*dim+w)*dim + h;
				out_data[index] = BP_r_cpu(n, A_inv, Y, Z, &J);
				out_data[index2] = out_data[index];
			}
		}
	}
}

#ifdef CPU_ONLY
STUB_GPU(CrfLossLayer);
#endif

INSTANTIATE_CLASS(CrfLossLayer);
REGISTER_LAYER_CLASS(CrfLoss);

}  // namespace caffe
