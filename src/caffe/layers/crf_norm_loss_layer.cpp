#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/crf_norm_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

#include <math.h>

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void CrfNormLossLayer<Dtype>::LayerSetUp(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
	LossLayer<Dtype>::LayerSetUp(bottom, top);
	// Parse the param
	CrfNormLossParameter crf_param = this->layer_param_.crf_norm_loss_param();

	if (crf_param.has_unary()){
		switch (crf_param.unary()){
			case CrfNormLossParameter_UnaryDist_L2:
				unary_mode_ = L2;
				break;
			case CrfNormLossParameter_UnaryDist_Berhu:
				unary_mode_ = Berhu;
				break;
			case CrfNormLossParameter_UnaryDist_ScaleInvariant:
				unary_mode_ = ScaleInvariant;
				break;
			default:
				unary_mode_ = L2;
		}
	}else{
		unary_mode_ = L2;
	}

	if (crf_param.has_w1()){
		w1_ = crf_param.w1();
	}else{
		w1_ = 1;
	}

	if (crf_param.has_w2()){
		w2_ = crf_param.w2();
	}else{
		w2_ = 1;
	}

	if (crf_param.has_w3()){
		w3_ = crf_param.w3();
	}else{
		w3_ = 1;
	}

	if (crf_param.has_focal()){
		f_ = crf_param.focal();
	}else{
		LOG(FATAL)<<"The focal value must be set";
	}

	if (crf_param.has_alpha()){
		alpha_ = crf_param.alpha();
	}else{
		alpha_ = 1;
	}

	if (crf_param.has_theta()){
		theta_ = crf_param.theta();
	}else{
		theta_ = 0;
	}

	if (crf_param.has_normalize()){
		normalize_ = crf_param.normalize();
	}else{
		normalize_ = false;
	}

	if (crf_param.has_height()){
		height_ = crf_param.height();
	}else{
		LOG(FATAL) << "The height is not set in CrfNormLossLayer";
	}

	if (crf_param.has_width()){
		width_ = crf_param.width();
	}else{
		LOG(FATAL) << "The width is not set in CrfNormLossLayer";
	}

	if(crf_param.has_c_rate()){
		c_rate_ = crf_param.c_rate();
	}else{
		c_rate_ = 0.2;
	}

	if (crf_param.has_disable_normal()){
		disable_normal_guidance_ = crf_param.disable_normal();
	}else{
		disable_normal_guidance_ = false;
	}

	if (crf_param.has_delta()){
		delta_ = crf_param.delta();
	}else{
		delta_ = 0.5;
	}

	if (crf_param.has_adjust_pixel_num()){
		adjust_pixel_num_ = crf_param.adjust_pixel_num();
	}else{
		adjust_pixel_num_ = false;
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

	if (bottom.size() == 4){
		has_appearance_ = false;
	}else{
		has_appearance_ = true;
		LOG(FATAL) << "Currently, the appearance bottom is not supported";
	}
	
	// Check the input
	/*
	 * bottom[0]: [n, 1, Hp, 1]. The superpixel pooled depth prediction, Hp is the number of the superpixel
	 * bottom[1]: [n, 1, Hp, 1]. The superpixel pooled depth label
	 * bottom[2]: [n, 3, Hp, 1]. The superpixel pooled normal prediction
	 * bottom[3]: [n, 1, Hp, 2]. The centroid coordination provided by the SuperpixelCentroidLayer
	 *		The 2 elements on width is [height, width] coordinate.
	 * bottom[4]: [n, 1, Hp, Hp]. The superpixel appearance CorrMat provided by the DistCosCorrMatLayer [optional]
	 */
	// Check the number of input blobs
	superpixel_num_ = bottom[0]->height();
	if (bottom.size() < 4 || bottom.size() > 5){
		LOG(FATAL) << "The number of bottom blobs must be 4 or 5";
	}
	CHECK_EQ(bottom[0]->count(), bottom[1]->count()) << "The size of the bottom[0](prediction) and the bottom[1](ground truth) must be the same";
	CHECK_EQ(bottom[0]->count(2), bottom[1]->height());
	CHECK_EQ(bottom[0]->channels(), 1) << "The bottom[0] is the depth prediciton";
	CHECK_EQ(bottom[1]->channels(), 1) << "The bottom[0] is the superpixel";
	CHECK_EQ(bottom[1]->height(), superpixel_num_);

	if (bottom[2]->channels() == 2){
		use_gradient_ = true;
	}else if (bottom[2]->channels() == 3){
		use_gradient_ = false;
	}else{
		LOG(FATAL) << "CrfNormLossLayer: The channels of bottom[2] must be 2 or 3";
	}
	CHECK_EQ(bottom[2]->height(), superpixel_num_);

	CHECK_EQ(bottom[3]->height(), superpixel_num_);
	CHECK_EQ(bottom[3]->width(), 2) << "The bottom[3] is the centroid coordination provided by SuperpixelCentroidLayer";

	if (has_appearance_){
		CHECK_EQ(bottom[4]->height(), superpixel_num_);
		CHECK_EQ(bottom[4]->width(), superpixel_num_);
	}

	CHECK_LE(top.size(), 2) << "Only support at most 2 top blobs";
}

template <typename Dtype>
void CrfNormLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";

  // Reshape the top[1]
  if(top.size()==2){
	  top[1]->ReshapeLike(*bottom[0]);
  }
  // Reshape the internal param
  // The A_ is a positive defination matrix with shape
  // [n, 1, superpixel_num_, superpixel_num_]
  A_.Reshape(bottom[0]->num(), 1, superpixel_num_, superpixel_num_);
  A_inv_.ReshapeLike(A_);
  R_.ReshapeLike(A_);
  Pred_.ReshapeLike(*bottom[0]);
  // Reshape the bad_pixel_
  bad_pixel_.Reshape(bottom[0]->num(), 1, bottom[0]->height(), 1); 
  // Reshape the D_;
  D_.Reshape(bottom[0]->num(), 2, superpixel_num_, superpixel_num_);
  // Reshape the G_;
  G_.Reshape(bottom[0]->num(), 2, superpixel_num_, 1);
  // If use scale invariant mode, init the Q and P
  if (unary_mode_ == ScaleInvariant){
	  Q_.ReshapeLike(A_);
	  P_.ReshapeLike(A_);
	  vecSum_.Reshape(bottom[0]->num(), 1, 1, 1);
	  buf_.ReshapeLike(*bottom[0]);
  }
}

template <typename Dtype>
void CrfNormLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // D and G must be calc before the R
  // Since the projected depth diff need G and D matrix
  // Generate the D
  Calc_D_cpu(bottom[3]);
  // Generate the G
  Calc_G_cpu(bottom[2]);
  // Generate the R
  Calc_R_cpu(bottom);
  // Init the P and Q
  Init_QP_cpu();
  // Calc the A matrix
  Calc_A_cpu();
  // Calc the A_inv matrix
  Calc_A_inv_cpu();

  // Inference the crf
  switch (unary_mode_){
	  case ScaleInvariant:
		  Inference_scaleinvariant_cpu(bottom[0]);
		  break;
	  default:
		  Inference_cpu(bottom[0]);
		  break;
  }
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
	  case ScaleInvariant:
		  ScaleInvariant_loss_cpu(bottom[1], top[0]);
		  break;
	  default:
		  LOG(FATAL)<<"Unknow unary_mode_ in CrfNormLossLayer";
		  break;
  }
}

template <typename Dtype>
void CrfNormLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	// The BP will performed on the bottom[0] and bottom[2]
	const Dtype loss_weight = top[0]->cpu_diff()[0];
	Dtype beta;
	if (normalize_) {
		beta = loss_weight / bottom[0]->count();
	}else{
		beta = loss_weight / bottom[0]->num();
	}

	if (unary_mode_ == ScaleInvariant){
		caffe_cpu_axpby(
				bottom[0]->count(),
				beta,
				Pred_.cpu_diff(),
				Dtype(0),
				buf_.mutable_cpu_data());

		// In scale invariant mode, the BP should be P*A_inv_*P*Z - P*Y
		// diff = A_inv_*P*Z - Y, so the BP should be
		// P * diff
		const Dtype* p_data = P_.cpu_data();
		const Dtype* buf_data = buf_.cpu_data();
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

		for (int n = 0; n < bottom[0]->num(); n++){
			const Dtype* p_data_n = p_data + P_.offset(n);
			for (int c = 0; c < bottom[0]->channels(); c++){
				const Dtype* buf_data_nc = buf_data + buf_.offset(n, c);
				Dtype* bottom_diff_nc = bottom_diff + bottom[0]->offset(n, c);
				caffe_cpu_symv(P_.height(), Dtype(1), p_data_n, buf_data_nc, Dtype(0), bottom_diff_nc);
			}
		}
	}else{
		// BP for bottom[0]
		caffe_cpu_axpby(
				bottom[0]->count(),
				beta,
				Pred_.cpu_diff(),
				Dtype(0),
				bottom[0]->mutable_cpu_diff());
	}

}

/* Formulate the R matrix, which can be calculated from
 * bottom[2] normal prediction
 * bottom[3] centroid coordination
 * bottom[4] superpixel appearance [TODO]
 *
 * The R is combination of three parts:
 * 1. The cosine distance of surface normal, which is 1 - cos(angle)
 * 2. The normalized distance between two superpixel
 * 3. The cosine distance between appearance vector [optional]
 *
 * The overall R is:
 * exp(-w1*M1 - w2*M2 - w3*M3)
 */
template <typename Dtype>
void CrfNormLossLayer<Dtype>::Calc_R_cpu(const vector<Blob<Dtype>*>& bottom){
	const Blob<Dtype>* norm_bottom = bottom[2];
	const Blob<Dtype>* centroid_bottom = bottom[3];
//    const Blob<Dtype>* appearance_bottom = bottom[4];

	const Dtype* norm_data = norm_bottom->cpu_data();
	const Dtype* centroid_data = centroid_bottom->cpu_data();
//    const Dtype* appearance_data = appearance_bottom->cpu_data();

	Dtype* R_data = R_.mutable_cpu_data();

	// Set R to zeros
	caffe_set(R_.count(), Dtype(0), R_data);

	// Iter the superpixel, and calc the cos dist of the surface normal
	// and calc the projected depth diff between two superpixels
	for (int n = 0; n < norm_bottom->num(); n++){
		for (int h = 0; h < norm_bottom->height(); h++){
			for (int w = 0; w < h; w++){
				Dtype val = 0;
				Dtype data1_norm = 1;
				Dtype data2_norm = 1;

				if (use_gradient_){
					// Calc the x,y,z from the dx and dy
					const Dtype data1_dx = norm_data[norm_bottom->offset(n, 0, h)];
					const Dtype data1_dy = norm_data[norm_bottom->offset(n, 1, h)];
					const Dtype data2_dx = norm_data[norm_bottom->offset(n, 0, w)];
					const Dtype data2_dy = norm_data[norm_bottom->offset(n, 1, w)];
					const Dtype data1_z = 1;
					const Dtype data1_x = - data1_z * data1_dx;
					const Dtype data1_y = - data1_z * data1_dy;

					const Dtype data2_z = 1;
					const Dtype data2_x = - data2_z * data2_dx;
					const Dtype data2_y = - data2_z * data2_dy;

					val = data1_x * data2_x + data1_y * data2_y + data1_z + data2_z;
					data1_norm = sqrt(data1_x * data1_x + data1_y * data1_y + data1_z + data1_z);
					data2_norm = sqrt(data2_x * data2_x + data2_y * data2_y + data2_z + data2_z);
				}else{
					const Dtype* data1 = norm_data + norm_bottom->offset(n, 0, h);
					const Dtype* data2 = norm_data + norm_bottom->offset(n, 0, w);
					// Calc the dot of data1 and data2
					val = caffe_cpu_strided_dot(norm_bottom->channels(), data1, superpixel_num_, data2, superpixel_num_);
					// Calc the L2 norm of the normal vectors
					data1_norm = L2Norm(norm_bottom->channels(), superpixel_num_, data1);
					data2_norm = L2Norm(norm_bottom->channels(), superpixel_num_, data2);
				}

				const Dtype* coord_data1 = centroid_data + centroid_bottom->offset(n, 0, h);
				const Dtype* coord_data2 = centroid_data + centroid_bottom->offset(n, 0, w);

				int idx1 = R_.offset(n, 0, h, w);
				int idx2 = R_.offset(n, 0, w, h);

				// Calc the cosine of the normal angle
				Dtype cos_val = 0;
				if (data1_norm == 0 || data2_norm == 0){
					cos_val = 0;
				}else{
					cos_val = min(max(val / data1_norm / data2_norm, Dtype(-1)), Dtype(1.0));
				}
				// Apply the theta Threshold
				if (cos_val < theta_){
					cos_val = 0;
				}
				// The larger means the less regulation
				cos_val = Dtype(1) - cos_val;
				if (isnan(cos_val)){
					LOG(INFO) << "VAL: " << val <<"cosine: "<<cos_val;
				}
				// Calc the distance square between two superpixels
				Dtype distsq = L2DistSq(coord_data1, coord_data2);

				// Calc the project depth diff between superpixel h and superpixel w
				Dtype proj_diff = 0;
			    if (w3_ != 0){
					proj_diff = Calc_project_depth_diff_cpu(n, h, w, bottom[0]);
				}
				
				// Set the R matrix
				R_data[idx1] = -cos_val*w1_-distsq*w2_-proj_diff*w3_;
//                LOG(INFO)<<"a1:"<<cos_val<<"a2:"<<distsq<<"a3:"<<proj_diff;
				R_data[idx2] = R_data[idx1];
			}
		}
	}
	// Turn R_ into the exp space
	caffe_exp(R_.count(), R_data, R_data);
	// Re-scale the R data
	caffe_scal(R_.count(), Dtype(alpha_), R_data);
}

/*
 * Calc the D matrix according to the centroid superpixel
 * The bottom shape is [n, 1, superpixel_num_, 2]
 * [height, width]
 */
template <typename Dtype>
void CrfNormLossLayer<Dtype>::Calc_D_cpu(const Blob<Dtype>* bottom){
	Dtype* D_data = D_.mutable_cpu_data();
	const Dtype* bottom_data = bottom->cpu_data();

	const int stride = superpixel_num_ * superpixel_num_;

	// Set D to zeros
	caffe_set(D_.count(), Dtype(0), D_data);

	for (int n = 0; n < bottom->num(); n++){
		for (int h = 0; h < superpixel_num_; h++){
			for (int w = 0; w < h; w++){
				const int idx1_h = D_.offset(n, 0, h, w);
				const int idx2_h = D_.offset(n, 0, w, h);
				const int idx1_w = idx1_h + stride;
				const int idx2_w = idx2_h + stride;
				const int idx1 = bottom->offset(n, 0, h);
				const int idx2 = bottom->offset(n, 0, w);

				const Dtype h1 = bottom_data[idx1];
				const Dtype w1 = bottom_data[idx1+1];
				const Dtype h2 = bottom_data[idx2];
				const Dtype w2 = bottom_data[idx2+1];

				const Dtype dh = (h2 - h1) / f_;
				const Dtype dw = (w2 - w1) / f_;

				D_data[idx1_h] = dh;
				D_data[idx1_w] = dw;
				D_data[idx2_h] = -dh;
				D_data[idx2_w] = -dw;
			}
		}
	}
}

/*
 * Calc the G matrix according to the superpixel pooled normal map
 * The shape of the bottom is [n, 3, sp_num, 1]
 */
template <typename Dtype>
void CrfNormLossLayer<Dtype>::Calc_G_cpu(const Blob<Dtype>* bottom){
	const Dtype* bottom_data = bottom->cpu_data();
	Dtype* G_data = G_.mutable_cpu_data();

	if (use_gradient_){
		// Direct copy the bottom into G_
		CHECK_EQ(bottom->count(), G_.count());
		caffe_copy(G_.count(), bottom_data, G_data);
		return;
	}
	const int bottom_stride = bottom->count(2);
	const int G_stride = G_.count(2);

	// Set G to zeros
	caffe_set(G_.count(), Dtype(0), G_data);

	// Iter the bottom
	for (int n = 0; n < bottom->num(); n++){
		for (int h = 0; h < bottom->height(); h++){
			// Get the x, y, z
			const int bottom_idx = bottom->offset(n, 0, h);
			const int G_idx = G_.offset(n, 0, h);
			Dtype x = bottom_data[bottom_idx];
			Dtype y = bottom_data[bottom_idx+bottom_stride];
			Dtype z = bottom_data[bottom_idx+2*bottom_stride];

			Dtype dx = - x / z;
			Dtype dy = - y / z;

			if (fabs(z) < 0.1){
				dx = 0;
				dy = 0;
			}

			G_data[G_idx] = dy;
			G_data[G_idx+G_stride] = dx;
		}
	}
}

template <typename Dtype>
void CrfNormLossLayer<Dtype>::Calc_A_cpu(void){
	// A = I + D - R - K + M + N
	// Set A to 0
	caffe_set(A_.count(), Dtype(0), A_.mutable_cpu_data());
	const int num = A_.num();
	const int channels = A_.channels();
	const int height = A_.height();
	const int width = A_.width();
	const int d_stride = D_.count(2);
	const int g_stride = G_.count(2);
	const Dtype* r_data = R_.cpu_data();
	const Dtype* d_data = D_.cpu_data();
	const Dtype* g_data = G_.cpu_data();
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

	// If disable the surface normal guidance, return directly
	if (disable_normal_guidance_) return;

	// A = A - K
	// K_ij = 0.5 * R_ij * (D_ij * G_i + D_ji * G_j)
	// K_ij = 0.5 * R_ij * (D_ij * G_i - D_ij * G_j)
	// Iter tha A
	for (int n = 0; n < A_.num(); n++){
		for (int i = 0; i < superpixel_num_; i++){
			for (int j = 0; j < superpixel_num_; j++){
				const int AR_idx = A_.offset(n, 0, i, j);
				const int D_idx = D_.offset(n, 0, i, j);
				const int G_idx_i = G_.offset(n, 0, i);
				const int G_idx_j = G_.offset(n, 0, j);

				a_data[AR_idx] -= 0.5 * r_data[AR_idx] * (d_data[D_idx] * g_data[G_idx_i] + d_data[D_idx+d_stride] * g_data[G_idx_i+g_stride] - d_data[D_idx] * g_data[G_idx_j] - d_data[D_idx+d_stride] * g_data[G_idx_j+g_stride]);
			}
		}
	}

	// A = A + M + N
	// M_ii = sigma_j (R_ij * D_ij * G_i)
	// N_ii = 0.5 * sigma_j (R_ij * (D_ij * G_i)^2 )
	for (int n = 0; n < A_.num(); n++){
		for (int i = 0; i < superpixel_num_; i++){
			const int A_idx = A_.offset(n, 0, i, i);
			const int G_idx_i = G_.offset(n, 0, i);

			for (int j = 0; j < superpixel_num_; j++){
				const int D_idx = D_.offset(n, 0, i, j);
				const int R_idx = R_.offset(n, 0, i, j);
				const Dtype val = d_data[D_idx] * g_data[G_idx_i] + d_data[D_idx+d_stride] * g_data[G_idx_i+g_stride];
				a_data[A_idx] += r_data[R_idx] * val + 0.5 * r_data[R_idx] * val * val;
			}
		}
	}

	// If in scale invariant mode
	// A = A - Q
	if (unary_mode_ == ScaleInvariant){
		const Dtype* q_data = Q_.cpu_data();
		caffe_sub(A_.count(), a_data, q_data, a_data);
	}
}

template <typename Dtype>
void CrfNormLossLayer<Dtype>::Inference_cpu(const Blob<Dtype>* Z){
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
void CrfNormLossLayer<Dtype>::Inference_scaleinvariant_cpu(const Blob<Dtype>* Z){
	// pred = A_inv_ * P * Z
	const int dim = A_inv_.height();
	const Dtype* a_data = A_inv_.cpu_data();
	const Dtype* z_data = Z->cpu_data();
	const Dtype* p_data = P_.cpu_data();

	// Creat a buffer to make sure the blas safe
	buf_.ReshapeLike(Pred_);
	Dtype* buf_data = buf_.mutable_cpu_data();

	Dtype* pred_data = Pred_.mutable_cpu_data();

	for (int n = 0; n < A_inv_.num(); n++){
		const Dtype* a_data_n = a_data + A_inv_.offset(n);
		const Dtype* p_data_n = p_data + P_.offset(n);
		for (int c = 0; c < Z->channels(); c++){
			const Dtype* z_data_nc = z_data + Z->offset(n, c);
			Dtype* buf_data_nc = buf_data + buf_.offset(n, c);
			Dtype* pred_data_nc = pred_data + Pred_.offset(n, c);
			// buf = P * Z
			caffe_cpu_symv(dim, Dtype(1), p_data_n, z_data_nc, Dtype(0), buf_data_nc);
			// pred = A_inv_ * pred
			caffe_cpu_symv(dim, Dtype(1), a_data_n, buf_data_nc, Dtype(0), pred_data_nc);
		}
	}

}

template <typename Dtype>
void CrfNormLossLayer<Dtype>::Calc_A_inv_cpu(void){
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
void CrfNormLossLayer<Dtype>::Euclidean_loss_cpu(const Blob<Dtype>* gt, Blob<Dtype>* top){
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
void CrfNormLossLayer<Dtype>::Berhu_loss_cpu(const Blob<Dtype>* gt, Blob<Dtype>* top){
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

template <typename Dtype>
void CrfNormLossLayer<Dtype>::ScaleInvariant_loss_cpu(const Blob<Dtype>* gt, Blob<Dtype>* top){

  CHECK_EQ(gt->channels(), 1);
  int count = gt->count();
  caffe_sub(
      count,
      Pred_.cpu_data(),
      gt->cpu_data(),
      Pred_.mutable_cpu_diff());

  if (has_min_label_ || has_max_label_){
	  const Dtype* data_label = gt->cpu_data();
	  Dtype* data_diff = Pred_.mutable_cpu_diff();
	  for(int n = 0; n < Pred_.num(); n++){
		  // The channels must be 1 here
		  for(int c = 0; c < Pred_.channels(); c++){
			  for(int h = 0; h < Pred_.height(); h++){
				  const Dtype* data_label_w = data_label + gt->offset(n, c, h);
				  Dtype* data_diff_w = data_diff + Pred_.offset(n, c, h);
				  for(int w = 0; w < Pred_.width(); w++){
					  Dtype val = data_label_w[w];
					  if (val < min_label_ || val > max_label_){
						  // Set the diff to 0
						  data_diff_w[w] = Dtype(0);
					  }
				  }
			  }
		  }
	  }
  }

  Dtype dot = caffe_cpu_dot(count, Pred_.cpu_diff(), Pred_.cpu_diff());
  // The first term of the loss, basically is the L2 loss
  Dtype loss = dot / gt->count() / Dtype(2);

  // Calc the second term of the loss
  // Calc the sum of the diff
  // The tmp vector for the sum of the each sample for the minibatch
  // The pixels of the image
  Dtype pixel_num = Dtype(gt->count(1));
  Dtype* vecSum_data = vecSum_.mutable_cpu_data();
  for (int n = 0; n < gt->num(); n++){
	  const Dtype* pdata = Pred_.cpu_diff() + Pred_.offset(n);
	  vecSum_data[n] = caffe_cpu_sum(pixel_num, pdata);
	  loss -= vecSum_data[n] * vecSum_data[n] / superpixel_num_ / superpixel_num_ / gt->num() * delta_ / Dtype(2);
  }

  top->mutable_cpu_data()[0] = loss;
}


template <typename Dtype>
Dtype CrfNormLossLayer<Dtype>::L2Norm(const int count, const int stride, const Dtype* data){
	return sqrt(caffe_cpu_strided_dot(count, data, stride, data, stride));
}

/*
 * Calc the L2 distance square between two superpixels
 */
template <typename Dtype>
Dtype CrfNormLossLayer<Dtype>::L2DistSq(const Dtype* data1, const Dtype* data2){
	const Dtype h1 = data1[0] / height_;
	const Dtype w1 = data1[1] / width_;
	const Dtype h2 = data2[0] / height_;
	const Dtype w2 = data2[1] / width_;

	const Dtype dh = h1 - h2;
	const Dtype dw = w1 - w2;

	return dh*dh + dw*dw;
}

template <typename Dtype>
Dtype CrfNormLossLayer<Dtype>::Calc_project_depth_diff_cpu(const int n, const int idx1, const int idx2, const Blob<Dtype>* dep){
	const Dtype* dep_data = dep->cpu_data();
	const Dtype dep1 = dep_data[dep->offset(n, 0, idx1)];
	const Dtype dep2 = dep_data[dep->offset(n, 0, idx2)];

	const Dtype* d_data = D_.cpu_data();
	const Dtype* g_data = G_.cpu_data();

	const Dtype d12h = d_data[D_.offset(n, 0, idx1, idx2)];
	const Dtype d12w = d_data[D_.offset(n, 1, idx1, idx2)];

	const Dtype d21h = -d12h;
	const Dtype d21w = -d12w;

	const Dtype g1y = g_data[G_.offset(n, 0, idx1)];
	const Dtype g1x = g_data[G_.offset(n, 1, idx1)];

	const Dtype g2y = g_data[G_.offset(n, 0, idx2)];
	const Dtype g2x = g_data[G_.offset(n, 1, idx2)];

	// Calc the dep1 project depth
	const Dtype dep1_proj = dep2 + dep2*(d21h * g2y + d21w * g2x);
	const Dtype dep2_proj = dep1 + dep1*(d12h * g1y + d12w * g1x);

	// Calc the distance between two superpixels
	const Dtype dist = sqrt(d12h * d12h + d12w * d12w);
	// Normalize the dep project diff
	return (fabs(dep1_proj - dep1) + fabs(dep2_proj - dep2)) / dist;
}


template <typename Dtype>
void CrfNormLossLayer<Dtype>::Init_QP_cpu(void){
	if (unary_mode_ != ScaleInvariant) return;

	const Dtype val = delta_ / superpixel_num_;
	// Set the Q matrix
	Dtype* q_data = Q_.mutable_cpu_data();
	caffe_set(Q_.count(), val, q_data);

	// Set the P matrix
	// P = I - Q
	Dtype* p_data = P_.mutable_cpu_data();
	caffe_set(P_.count(), Dtype(0), p_data);
	caffe_sub(P_.count(), p_data, q_data, p_data);
	for (int n = 0; n < P_.num(); n++){
		for (int i = 0; i < P_.height(); i++){
			p_data[P_.offset(n, 0, i, i)] += Dtype(1);
		}
	}
}


#ifdef CPU_ONLY
STUB_GPU(CrfNormLossLayer);
#endif

INSTANTIATE_CLASS(CrfNormLossLayer);
REGISTER_LAYER_CLASS(CrfNormLoss);

}  // namespace caffe
