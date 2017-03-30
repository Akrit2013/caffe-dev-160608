#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/crf_unary_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void CrfUnaryLossLayer<Dtype>::LayerSetUp(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
	LossLayer<Dtype>::LayerSetUp(bottom, top);
	// Parse the param
	CrfUnaryLossParameter crf_param = this->layer_param_.crf_unary_loss_param();

	if (crf_param.has_unary()){
		switch (crf_param.unary()){
			case CrfUnaryLossParameter_UnaryDist_L2:
				unary_mode_ = L2;
				break;
			case CrfUnaryLossParameter_UnaryDist_Berhu:
				unary_mode_ = Berhu;
				break;
			case CrfUnaryLossParameter_UnaryDist_Berhuber:
				unary_mode_ = Berhuber;
				break;
			default:
				unary_mode_ = L2;
		}
	}else{
		unary_mode_ = L2;
	}

	if (crf_param.has_alpha()){
		alpha_ = crf_param.alpha();
	}else{
		alpha_ = 1;
	}

	if (crf_param.has_normalize()){
		normalize_ = crf_param.normalize();
	}else{
		normalize_ = false;
	}

	// Init the params for berhu param
	if(crf_param.has_h_rate()){
		h_rate_ = crf_param.h_rate();
		has_h_rate_ = true;
	}else{
		h_rate_ = 0;
		has_h_rate_ = false;
	}
	if(crf_param.has_c_rate()){
		c_rate_ = crf_param.c_rate();
	}else{
		c_rate_ = 0.2;
	}
	if(crf_param.has_max_label()){
		has_max_label_ = true;
		max_label_ = crf_param.max_label();
	}else{
		has_max_label_ = false;
		max_label_ = 0;
	}
	if(crf_param.has_min_label()){
		has_min_label_ = true;
		min_label_ = crf_param.min_label();
	}else{
		has_min_label_ = false;
		min_label_ = 0;
	}
	if(crf_param.has_invalid_label()){
		has_invalid_label_ = true;
		invalid_label_ = crf_param.invalid_label();
	}else{
		has_invalid_label_ = false;
		invalid_label_ = 0;
	}	
	if(crf_param.has_c_rate_mode()){
		switch (crf_param.c_rate_mode()){
			case CrfUnaryLossParameter_CRateMode_MAX:
				c_rate_mode_ = MAX;
				break;
			case CrfUnaryLossParameter_CRateMode_AVE:
				c_rate_mode_ = AVE;
				break;
			default:
				LOG(FATAL) << "Unsupport c_rate_mode";
				break;
		}
	}else{
		c_rate_mode_ = MAX;
	}
	
	// Check the input
	// The bottom[0] indicate the unary potential
	// The bottom[1] indicate the ground truth
	// The bottom[2] indicate the R matrix
	CHECK_EQ(bottom[0]->count(), bottom[1]->count()) << "The size of the bottom[0](prediction) and the bottom[1](ground truth) must be the same";
	CHECK_EQ(bottom[0]->count(2), bottom[2]->height());
	CHECK_EQ(bottom[2]->height(), bottom[2]->width()) << "The height and width of the pairwise potential must be the same";
	CHECK_LE(top.size(), 2) << "Only support at most 2 top blobs";
	CHECK_EQ(bottom[2]->channels(), 1) << "Currently, only support the pairwise with 1 channel";
}

template <typename Dtype>
void CrfUnaryLossLayer<Dtype>::Reshape(
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
  // Reshape the bad_pixel_
  bad_pixel_.Reshape(bottom[0]->num(), 1, bottom[0]->height(), 1); 
}

template <typename Dtype>
void CrfUnaryLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // Generate the R
  Calc_R_cpu(bottom[2]);
  // Calc the A matrix
  Calc_A_cpu();
  // Calc the A_inv matrix
  Calc_A_inv_cpu();

  // Inference the crf
  Inference_cpu(bottom[0]);
  // Copy the result if needed
  if (top.size() == 2){
	  caffe_copy(Pred_.count(), Pred_.cpu_data(), top[1]->mutable_cpu_data());
  }
  // Calc the loss according to the Pred_ and the bottom[0]
  // The diff will be stored in Pred_.diff
  switch (unary_mode_){
	  case L2:
		  Euclidean_loss_cpu(bottom[1], top[0]);
		  break;
	  case Berhu:
		  Berhu_loss_cpu(bottom[1], top[0]);
		  break;
	  default:
		  LOG(FATAL)<<"Unknow unary_mode_ in CrfUnaryLossLayer";
		  break;
  }
}

template <typename Dtype>
void CrfUnaryLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	// The BP will performed on the bottom[0] and bottom[2]
	const Dtype loss_weight = top[0]->cpu_diff()[0];
	Dtype beta;
	if (normalize_) {
		beta = loss_weight / bottom[0]->count();
	}else{
		beta = loss_weight / bottom[0]->num();
	}

	// BP for bottom[0]
	caffe_cpu_axpby(
			bottom[0]->count(),
			beta,
			Pred_.cpu_diff(),
			Dtype(0),
			bottom[0]->mutable_cpu_diff());
}

template <typename Dtype>
void CrfUnaryLossLayer<Dtype>::Calc_R_cpu(const Blob<Dtype>* bottom){
	// The bottom[2] is the R
	caffe_cpu_scale(bottom->count(), alpha_, bottom->cpu_data(), R_.mutable_cpu_data());
}

template <typename Dtype>
void CrfUnaryLossLayer<Dtype>::Calc_A_cpu(void){
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
void CrfUnaryLossLayer<Dtype>::Inference_cpu(const Blob<Dtype>* Z){
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
void CrfUnaryLossLayer<Dtype>::Calc_A_inv_cpu(void){
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
void CrfUnaryLossLayer<Dtype>::Euclidean_loss_cpu(const Blob<Dtype>* gt, Blob<Dtype>* top){
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

template <typename Dtype>
void CrfUnaryLossLayer<Dtype>::Berhu_loss_cpu(const Blob<Dtype>* gt, Blob<Dtype>* top){
  const int count = Pred_.count();
  const Dtype* label_data = gt->cpu_data();
  const Dtype* pred_data = Pred_.cpu_data();
  caffe_sub(
      count,
	  pred_data,
	  label_data,
	  Pred_.mutable_cpu_diff());

  Dtype max_diff = 0;

  switch(c_rate_mode_){
	  case MAX:
		  // Get the abs max diff to determine the C
		  max_diff = caffe_amax(count, Pred_.cpu_diff(), 1);
		  // Calc the Threshold C
		  break;
	  case AVE:
		  // Calc the mean of the abs diff
		  max_diff = caffe_cpu_asum(count, Pred_.cpu_diff()) / count;
		  break;
	  default:
		  LOG(FATAL) << "False c_rate_mode";
		  break;
  }
  Dtype C = fabs(max_diff * c_rate_);

  Dtype H = fabs(max_diff * h_rate_);

  // For debug
  // LOG(INFO)<<"Max_diff:"<<max_diff<<" C:"<<C;
  // Iter the diff map
  Dtype* data_diff = Pred_.mutable_cpu_diff();
  // const Dtype* data_pred = bottom[0]->cpu_data();
  // Set the diff to zero if label pixel is zero (all channel is zero)
  int bad_pixel_count = 0;
  for(int n = 0; n < Pred_.num(); ++n){
	 for(int h = 0; h < Pred_.height(); ++h){
		  for(int w = 0; w < Pred_.width(); ++w){
			  int err_counter = 0;
			  for(int c = 0; c < Pred_.channels(); ++c){
				  int index = ((n*Pred_.channels()+c)*Pred_.height()+h)*Pred_.width()+w;
				  Dtype dataval = label_data[index];
				  if (has_max_label_ && dataval > max_label_){
					  err_counter++;
				  }else if(has_min_label_ && dataval < min_label_){
					  err_counter++;
				  }else if(has_invalid_label_ && fabs(dataval - invalid_label_) < 0.0001){
					  err_counter++;
				  }
				  // Set the diff to L1 or L2 according to the C
				  Dtype diff_val = data_diff[index];
				  if (fabs(diff_val) <= C){
					  // L1
					  if (diff_val > 0){
						  data_diff[index] = C;
					  }else if(diff_val < 0){
						  data_diff[index] = -C;
					  }
				  }else if(has_h_rate_ && fabs(diff_val) > H){
					  if (diff_val > 0){
						  data_diff[index] = H;
					  }else{
						  data_diff[index] = -H;
					  }
				  }
			  }
			  if(err_counter == Pred_.channels()){
				  // This pixel is not ok
				  bad_pixel_count += Pred_.channels();
				  for(int c = 0; c < Pred_.channels(); ++c){
					  int index = ((n*Pred_.channels()+c)*Pred_.height()+h)*Pred_.width()+w;
					  data_diff[index] = 0;
				  }
			  }
		  }
	 }
  }

  Dtype dot = caffe_cpu_dot(count, Pred_.cpu_diff(), Pred_.cpu_diff());
  Dtype loss = dot / Dtype(2) / (count - bad_pixel_count);
  top->mutable_cpu_data()[0] = loss;
}

#ifdef CPU_ONLY
STUB_GPU(CrfUnaryLossLayer);
#endif

INSTANTIATE_CLASS(CrfUnaryLossLayer);
REGISTER_LAYER_CLASS(CrfUnaryLoss);

}  // namespace caffe
