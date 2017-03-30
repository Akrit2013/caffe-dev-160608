#include "caffe/layer.hpp"
#include "caffe/net.hpp"
#include "caffe/layers/focaltrans_layer.hpp"
#include "math.h"

#define PI 3.1415926

namespace caffe {

template <typename Dtype>
void FocalTransformLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	DLOG(INFO)<<"FocalTransformLayer setup start";
	CHECK_EQ(bottom.size(), top.size());
	// Currently, to simplfy the problem, it only support 2 input layers
	// BUt it is easy to modify the code to support more layers
	CHECK_EQ(bottom.size(), 2);

	// Get the transform parameter
	FocalTransformationParameter transform_param = this->layer_param_.focaltransform_param();
	if(transform_param.has_crop_rate()){
		crop_rate_ = transform_param.crop_rate();
	}else{
		crop_rate_ = 1;
	}
	// Check if the crop_prob is set
	if(transform_param.has_crop_prob()){
		crop_prob_ = transform_param.crop_prob();
	}else{
		crop_prob_ = 0;
	}
	// Check if the min_hview is set
	if(transform_param.has_min_hview()){
		min_hview_ = transform_param.min_hview();
	}else{
		min_hview_ = 0;
	}
	// Check if need to rotate
	if(transform_param.has_rotate_degree()){
		rotate_degree_ = transform_param.rotate_degree();
	}else{
		rotate_degree_ = 0;
	}
	// Check the label type
	if(transform_param.has_label_type()){
		switch(transform_param.label_type()){
			case FocalTransformationParameter_LabelType_RAD:
				label_type_ = radian;
				break;
			case FocalTransformationParameter_LabelType_DEG:
				label_type_ = degree;
				break;
			default:
				label_type_ = radian;
				break;
		}
	}else{
		label_type_ = radian;
	}
	Reshape(bottom, top);
	// Init the transformer
	transformer_.reset(new RandCrop<Dtype>(bottom[0], top[0]));
	transformer_->set_crop_rate(crop_rate_);
	transformer_->set_crop_prob(crop_prob_);
	transformer_->set_center_crop(true);
	transformer_->set_rotate_degree(rotate_degree_);
	// Reshape the blobs
}

template <typename Dtype>
void FocalTransformLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	// Reshape the top blobs, make sure the shape of each top blob equal with
	// the corresponding bottom blob
	CHECK_EQ(bottom.size(), top.size());
	for(int i = 0; i < bottom.size(); ++i){
		top[i]->ReshapeLike(*bottom[i]);
	}
}

template <typename Dtype>
void FocalTransformLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	DLOG(INFO)<<"FocalTransformLayer forward";
	// Crop the bottom[0]
	// Refresh the transform parameters very time
	vector<vector<double> > crop_param = transformer_->randparam();

	CHECK_EQ(crop_param.size(), bottom[1]->num());
	CHECK_EQ(bottom[1]->num(), bottom[0]->num());
	// Check the crop param, and change the focal length accordingly
	const Dtype* data_label = bottom[1]->cpu_data();
	Dtype* data_label_top = top[1]->mutable_cpu_data();

	for (int i = 0; i < crop_param.size(); ++i){
		double curr_crop_rate = crop_param[i][0];
		int label_offset = bottom[1]->offset(i);
		int top_offset = top[1]->offset(i);

		const Dtype hview_lab = *(data_label+label_offset);
		// First, convert the org_hview according to the curr_crop_rate
		Dtype hview_org = hview_lab;
		Dtype hview_crop= hview_lab;

		if (curr_crop_rate == 1){
			hview_crop = hview_lab;
		}else{

			switch (label_type_){
				case degree:
					// Convert the degree to radian
					hview_org = hview_lab / 180.0 * PI;
					break;
				case radian:
					// Do nothing
					hview_org = hview_lab;
					break;
			}
			// Convert the hview according to the crop_param
			double hview_after = 2.0*atan(curr_crop_rate*tan(hview_org/2.0));

			switch (label_type_){
				case degree:
					// Convert the degree to radian
					hview_crop = hview_after / PI * 180.0;
					break;
				case radian:
					// Do nothing
					hview_crop = hview_after;
					break;
			}

			// Check if the crop exceed the limitation
			if (hview_crop < min_hview_){
				crop_param[i][0] = 1;
				hview_crop = hview_lab;
			}
		}
		// Set the top label blob
		caffe_copy(1, &hview_crop, data_label_top+top_offset);
/* For debug
		const Dtype hview_top = *(data_label_top+label_offset);
		std::cout<<"hview:"<<hview_lab<<"hview_crop:"<<hview_crop<<"hview_top:"<<hview_top<<std::endl;
		*/
	}
	// Start the transform the blobs
	transformer_->crop(crop_param);
}

#ifdef CPU_ONLY
STUB_GPU(FocalTransformLayer);
#endif

INSTANTIATE_CLASS(FocalTransformLayer);
REGISTER_LAYER_CLASS(FocalTransform);

}  // namespace caffe
