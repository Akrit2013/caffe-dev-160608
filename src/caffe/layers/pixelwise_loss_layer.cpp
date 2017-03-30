#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/pixelwise_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

#include <math.h>

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void PixelwiseLossLayer<Dtype>::LayerSetUp(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
	LossLayer<Dtype>::LayerSetUp(bottom, top);
	// Check the input blobs
	CHECK_GE(bottom.size(), 5) << "The layer contains 5 bottoms";
	// Parse the param
	PixelwiseLossParameter loss_param = this->layer_param_.pixelwise_loss_param();

	if (loss_param.has_loss()){
		switch (loss_param.loss()){
			case PixelwiseLossParameter_LossMode_L2:
				loss_mode_ = L2;
				break;
			case PixelwiseLossParameter_LossMode_Berhu:
				loss_mode_ = Berhu;
				break;
			case PixelwiseLossParameter_LossMode_ScaleInvariant:
				loss_mode_ = ScaleInvariant;
				break;
			default:
				loss_mode_ = L2;
		}
	}else{
		loss_mode_ = L2;
	}

	if (loss_param.has_focal()){
		f_ = loss_param.focal();
	}else{
		LOG(FATAL) << "The focal param must be set";
	}

	if (loss_param.has_normalize()){
		normalize_ = loss_param.normalize();
	}else{
		normalize_ = false;
	}

	if(loss_param.has_c_rate()){
		c_rate_ = loss_param.c_rate();
	}else{
		c_rate_ = 0.2;
	}

	if (loss_param.has_delta()){
		delta_ = loss_param.delta();
	}else{
		delta_ = 0.5;
	}

	if (loss_param.has_z_thd()){
		z_thd_ = loss_param.z_thd();
	}else{
		z_thd_ = 0;
	}

	if(loss_param.has_max_label()){
		has_max_label_ = true;
		max_label_ = loss_param.max_label();
	}else{
		has_max_label_ = false;
		max_label_ = 0;
	}
	if(loss_param.has_min_label()){
		has_min_label_ = true;
		min_label_ = loss_param.min_label();
	}else{
		has_min_label_ = false;
		min_label_ = 0;
	}
	if(loss_param.has_bp_depth()){
		bp_depth_ = loss_param.bp_depth();
	}else{
		bp_depth_ = false;
	}
	if(loss_param.has_lambda()){
		lambda_ = loss_param.lambda();
	}else{
		lambda_ = 1.0;
	}
	if(loss_param.has_lr_z()){
		lr_z_ = loss_param.lr_z();
	}else{
		lr_z_ = 1.0;
	}
	if(loss_param.has_h_rate()){
		h_rate_ = loss_param.h_rate();
	}else{
		h_rate_ = -1;
	}
	if(h_rate_ > 0){
		has_h_rate_ = true;
	}else{
		has_h_rate_ = false;
	}
	if(loss_param.has_radius()){
		radius_ = loss_param.radius();
	}else{
		radius_ = -1;
	}
	if(radius_ > 0){
		has_radius_ = true;
	}else{
		has_radius_ = false;
	}


	if(loss_param.has_c_rate_mode()){
		switch (loss_param.c_rate_mode()){
			case PixelwiseLossParameter_CRateMode_MAX:
				c_rate_mode_ = MAX;
				break;
			case PixelwiseLossParameter_CRateMode_AVE:
				c_rate_mode_ = AVE;
				break;
			default:
				LOG(FATAL) << "Unsupport c_rate_mode";
				break;
		}
	}else{
		c_rate_mode_ = MAX;
	}
	superpixel_num_ = bottom[0]->height();
	height_ = bottom[1]->height();
	width_ = bottom[1]->width();

	// Check the input
	// Check the number of input blobs
	CHECK_EQ(bottom[0]->channels(), 1) << "The bottom[0] is the depth prediciton";
	CHECK_EQ(bottom[1]->channels(), 1) << "The bottom[0] is the depth label";
	CHECK_EQ(bottom[2]->height(), height_);
	CHECK_EQ(bottom[2]->width(), width_);

	CHECK_EQ(bottom[2]->channels(), 1) << "The bottom[2] is the superpixel segment result";
	CHECK_EQ(bottom[3]->height(), superpixel_num_);
	CHECK_EQ(bottom[3]->width(), 2) << "The bottom[3] is the centroid coordination provided by SuperpixelCentroidLayer";

	if (bottom[4]->channels() == 3){
		use_gradient_ = false;
		LOG(INFO) << "PixelwiseLossLayer: Use surface normal BP mode";
	}else if (bottom[4]->channels() == 2){
		use_gradient_ = true;
		LOG(INFO) << "PixelwiseLossLayer: Use surface gradient BP mode";
	}else{
		LOG(FATAL) << "PixelwiseLossLayer: The channels of bottom[4] must be 2 or 3";
	}

	CHECK_EQ(bottom[4]->height(), superpixel_num_) << "The bottom[4] is the superpixel poolded surface normal";

//    CHECK_LE(top.size(), 2) << "Only support at most 2 top blobs";
}

template <typename Dtype>
void PixelwiseLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);

  // Reshape the top[1]
  if(top.size()>=2){
	  top[1]->ReshapeLike(*bottom[1]);
  }
  Pred_.ReshapeLike(*bottom[1]);
  dep_diff_.ReshapeLike(*bottom[0]);
  // Reshape the bad_pixel_
  bad_pixel_.Reshape(bottom[0]->num(), 1, bottom[0]->height(), 1); 
  normAccum_.ReshapeLike(*bottom[4]);
  // Re-define the var
  superpixel_num_ = bottom[0]->height();
  height_ = bottom[1]->height();
  width_ = bottom[1]->width();
  if (loss_mode_ == ScaleInvariant){
	  vecSum_.Reshape(bottom[0]->num(), 1, 1, 1);
  }

  // NOTE: The top[2] is a debug top, which will output the internal
  // blobs as needed
  /*
  if (top.size() >= 3){
	  top[2]->Reshape(bottom[1]->num(), 3, height_, width_);
  }
  */
}

template <typename Dtype>
void PixelwiseLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  // Inference the pixelwise depth
  Pixelwise_inference_cpu(bottom);
  
  // Copy the result if needed
  if (top.size() >= 2){
	  caffe_copy(Pred_.count(), Pred_.cpu_data(), top[1]->mutable_cpu_data());
  }
  // Calc the loss according to the Pred_ and the bottom[0]
  // The diff will be stored in Pred_.diff
  switch (loss_mode_){
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
		  LOG(FATAL)<<"Unknow loss_mode_ in PixelwiseLossLayer";
		  break;
  }
}

template <typename Dtype>
void PixelwiseLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	// BP the depth 
	if (bp_depth_){
		LOG(FATAL) << "The depth BP of PixelwiseLossLayer is not implemented yet";
	}
	const Dtype loss_weight = top[0]->cpu_diff()[0];
	Dtype beta;
	if (normalize_) {
		beta = loss_weight / bottom[0]->count();
	}else{
		beta = loss_weight / bottom[0]->num();
	}

	const Blob<Dtype>* centroid_bottom = bottom[3];
	Blob<Dtype>* norm_bottom = bottom[4];
	const Blob<Dtype>* sp_bottom = bottom[2];
	const Blob<Dtype>* depth_bottom = bottom[0];
	const Blob<Dtype>* depth_gt_bottom = bottom[1];

	// BP the surface normal of each superpixel
    const Dtype* diff_data = Pred_.cpu_diff();
	const Dtype* centroid_data = centroid_bottom->cpu_data();
	const Dtype* sp_data = sp_bottom->cpu_data();
	const Dtype* norm_data = norm_bottom->cpu_data();
	const Dtype* dep_data = depth_bottom->cpu_data();
	const Dtype* dep_gt_data = depth_gt_bottom->cpu_data();

	Dtype* accum_data = normAccum_.mutable_cpu_data();
	Dtype* counter_data = normAccum_.mutable_cpu_diff();
	
	Dtype* bottom_diff = norm_bottom->mutable_cpu_diff();

	// Calc the dep diff of each superpixel

	caffe_set(normAccum_.count(), Dtype(0), accum_data);
	caffe_set(normAccum_.count(), Dtype(0), counter_data);

	const int stride = norm_bottom->count(2);
	CHECK_EQ(stride, superpixel_num_);

	for (int n = 0; n < Pred_.num(); n++){
		for (int h = 0; h < height_; h++){
			const int idx_nh = sp_bottom->offset(n, 0, h);
			for (int w = 0; w < width_; w++){
				// Get the sp id
				const int idx = idx_nh + w;
				const int sp_id = sp_data[idx];
				const int dep_idx = depth_bottom->offset(n, 0, sp_id);
				const int centroid_idx = centroid_bottom->offset(n, 0, sp_id);
				const int norm_idx = norm_bottom->offset(n, 0, sp_id);

				const Dtype dep = dep_data[dep_idx];
				const Dtype coord_h = centroid_data[centroid_idx];
				const Dtype coord_w = centroid_data[centroid_idx + 1];

				const int central_dep_gt_idx = depth_gt_bottom->offset(n, 0, int(coord_h), int(coord_w));
				const Dtype central_dep_gt = dep_gt_data[central_dep_gt_idx];
				const Dtype central_dep_diff = dep - central_dep_gt;

				const Dtype diff = diff_data[idx] - central_dep_diff;

				const Dtype coord_diff_h = Dtype(h) - coord_h;
				const Dtype coord_diff_w = Dtype(w) - coord_w;

				if (has_radius_){
					const Dtype dist = sqrt(coord_diff_h * coord_diff_h + coord_diff_w * coord_diff_w);
					if (dist > radius_){
						continue;
					}
				}


				if (use_gradient_){
					Dtype diff_x = diff * dep / f_ * coord_diff_w;
					Dtype diff_y = diff * dep / f_ * coord_diff_h;

					accum_data[norm_idx] += diff_x;
					accum_data[norm_idx + stride] += diff_y;
				}else{
					const Dtype x = norm_data[norm_idx];
					const Dtype y = norm_data[norm_idx + stride];
					Dtype z = norm_data[norm_idx + stride * 2];

					if (fabs(z) < z_thd_){
						z = z > 0 ? z_thd_ : - z_thd_;
					}

					// The diff of regulation term
					const Dtype regula = 2 * lambda_ * (x * x + y * y + z * z - 1);

					Dtype diff_x = - diff * dep / f_ * coord_diff_w / z + regula * x;
					Dtype diff_y = - diff * dep / f_ * coord_diff_h / z + regula * y;
					Dtype diff_z = diff * dep / f_ * (coord_diff_h * y + coord_diff_w * x) / z / z + regula * z;

					// Make sure the Z must be larger than 0
					// Since when update z = z - diff_z
					// so when z < 0, the diff_z must < 0
					if (z < z_thd_) {
						diff_z =  -fabs(diff_z);
					}

					accum_data[norm_idx] += diff_x;
					accum_data[norm_idx + stride] += diff_y;
					accum_data[norm_idx + stride * 2] += lr_z_ * diff_z;
				}

				counter_data[norm_idx] += 1;
			}
		}
	}
	// Average the diff for each superpixel
	for (int n = 0; n < norm_bottom->num(); n++){
		for (int h = 0; h < norm_bottom->height(); h++){
			const int norm_idx = norm_bottom->offset(n, 0, h);
			const Dtype accum_x_diff = accum_data[norm_idx];
			const Dtype accum_y_diff = accum_data[norm_idx + stride];
			const Dtype accum_z_diff = accum_data[norm_idx + stride * 2];
			const int counter = counter_data[norm_idx];

			bottom_diff[norm_idx] = beta * accum_x_diff / counter;
			bottom_diff[norm_idx + stride] = beta * accum_y_diff / counter;
			if (!use_gradient_){
				bottom_diff[norm_idx + stride * 2] = beta * accum_z_diff / counter;
			}

			if (counter == 0){
				LOG(FATAL) << "The accum counter is 0!";
			}
		}
	}

	// Refine the diff value, eliminate the diff which is too large
	// Get the max(abs(diff))
	
	if (has_h_rate_){
		const Dtype ave_diff = caffe_cpu_asum(norm_bottom->count(), bottom_diff) / Dtype(norm_bottom->count());
		const Dtype H = h_rate_ * ave_diff;

		// Iter the diff
		for (int i = 0; i < norm_bottom->count(); i++){
			if (fabs(bottom_diff[i]) > H){
				bottom_diff[i] = bottom_diff[i] > 0 ? H: -H;
			}
		}
	}
}


template <typename Dtype>
void PixelwiseLossLayer<Dtype>::Euclidean_loss_cpu(const Blob<Dtype>* gt, Blob<Dtype>* top){
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
void PixelwiseLossLayer<Dtype>::Berhu_loss_cpu(const Blob<Dtype>* gt, Blob<Dtype>* top){
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
void PixelwiseLossLayer<Dtype>::ScaleInvariant_loss_cpu(const Blob<Dtype>* gt, Blob<Dtype>* top){

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

  Dtype dot = caffe_cpu_dot(count, Pred_.cpu_data(), Pred_.cpu_diff());
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
Dtype PixelwiseLossLayer<Dtype>::L2Norm(const int count, const int stride, const Dtype* data){
	return sqrt(caffe_cpu_strided_dot(count, data, stride, data, stride));
}

/*
 * Calc the L2 distance square between two superpixels
 */
template <typename Dtype>
Dtype PixelwiseLossLayer<Dtype>::L2DistSq(const Dtype* data1, const Dtype* data2){
	const Dtype h1 = data1[0] / height_;
	const Dtype w1 = data1[1] / width_;
	const Dtype h2 = data2[0] / height_;
	const Dtype w2 = data2[1] / width_;

	const Dtype dh = h1 - h2;
	const Dtype dw = w1 - w2;

	return dh*dh + dw*dw;
}

/*
 * Inference the pixel wise depth prediciton and put it into Pred_
 */
template <typename Dtype>
void PixelwiseLossLayer<Dtype>::Pixelwise_inference_cpu(const vector<Blob<Dtype>*>& bottom){
	const Blob<Dtype>* depth_bottom = bottom[0];
	const Blob<Dtype>* sp_bottom = bottom[2];
	const Blob<Dtype>* norm_bottom = bottom[4];
	const Blob<Dtype>* centroid_bottom = bottom[3];

	const Dtype* dep_data = depth_bottom->cpu_data();
	const Dtype* sp_data = sp_bottom->cpu_data();
	const Dtype* norm_data = norm_bottom->cpu_data();
	const Dtype* centroid_data = centroid_bottom->cpu_data();

	Dtype* pred_data = Pred_.mutable_cpu_data();

	const int stride = superpixel_num_;
	// Iter the pred
	for (int n = 0; n < Pred_.num(); n++){
		for (int h = 0; h < height_; h++){
			const int pred_idx_nh = Pred_.offset(n, 0, h);
			for (int w = 0; w < width_; w++){
				const int sp_id = sp_data[pred_idx_nh + w];
				const int centroid_idx = centroid_bottom->offset(n, 0, sp_id);
				const Dtype coord_h = centroid_data[centroid_idx];
				const Dtype coord_w = centroid_data[centroid_idx + 1];
				const int norm_idx = norm_bottom->offset(n, 0, sp_id);

				bool valid_norm = true;
				Dtype dx = 0;
				Dtype dy = 0;
				if (use_gradient_){
					dx = norm_data[norm_idx];
					dy = norm_data[norm_idx + stride];
				}else{
					const Dtype x = norm_data[norm_idx];
					const Dtype y = norm_data[norm_idx + stride];
					Dtype z = norm_data[norm_idx + 2 * stride];

					// NOTE: Here the norm is not normalized, so there is a
					// chance that all x, y, z are small value
					if (fabs(z) < z_thd_){
						if (this->phase_ == TRAIN){
							z = z > 0 ? z_thd_ : -z_thd_;
						}else{
							valid_norm = false;
						}
					}	

					dx = - x / z;
					dy = - y / z;
				}

				const int dep_idx = depth_bottom->offset(n, 0, sp_id);
				const Dtype dep = dep_data[dep_idx];
				Dtype dep_proj = dep;
				if (valid_norm){
					// The coord diff between pixel and centroid
					const Dtype coord_diff_h = Dtype(h) - coord_h;
					const Dtype coord_diff_w = Dtype(w) - coord_w;
					dep_proj += dep / f_ * (coord_diff_h * dy + coord_diff_w * dx);
				}

				pred_data[pred_idx_nh + w] = dep_proj;
			}
		}
	}
}

#ifdef CPU_ONLY
STUB_GPU(PixelwiseLossLayer);
#endif

INSTANTIATE_CLASS(PixelwiseLossLayer);
REGISTER_LAYER_CLASS(PixelwiseLoss);

}  // namespace caffe
