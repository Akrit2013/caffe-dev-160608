#ifndef CAFFE_SYNCTRANSFORM_LAYER_HPP_
#define CAFFE_SYNCTRANSFORM_LAYER_HPP_

#include "caffe/layers/synctrans_layer.hpp"

namespace caffe {

template <typename Dtype>
void SyncTransformLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	CHECK_EQ(bottom.size(), top.size());
	// Get the transform parameter
	SyncTransformationParameter transform_param = this->layer_param_.synctransform_param();
	if(transform_param.has_crop_rate()){
		crop_rate_ = transform_param.crop_rate();
	}else{
		crop_rate_ = 1;
	}
	// Check if need to random mirror the image
	if(transform_param.has_mirror()){
		has_mirror_ = transform_param.mirror();
	}else{
		has_mirror_ = false;
	}
	// Check if the label image is a normal map
	if(transform_param.has_norm_map()){
		norm_map_ = transform_param.norm_map();
	}else{
		norm_map_ = true;
	}
	// Check if color channel need to be tuned
	if(transform_param.has_color_offset()){
		color_offset_ = transform_param.color_offset();
	}else{
		color_offset_ = 0;
	}
	// Check if need to random rotate the image
	if(transform_param.has_rotate_degree()){
		rotate_degree_ = transform_param.rotate_degree();
	}else{
		rotate_degree_ = 0;
	}
	// Can not both have crop and crop_w && crop_h
	CHECK(!(transform_param.has_crop() && (transform_param.has_crop_w() || transform_param.has_crop_h())));
	crop_h_ = 0;
	crop_w_ = 0;
	if(transform_param.has_crop()){
		crop_h_ = transform_param.crop();
		crop_w_ = transform_param.crop();
	}else if(transform_param.has_crop_h()){
		crop_h_ = transform_param.crop_h();
	}else if(transform_param.has_crop_w()){
		crop_w_ = transform_param.crop_w();
	}
	// Init the transformer
	transformer_.reset(new Transformer<Dtype>(bottom, top));
	transformer_->set_crop_rate(crop_rate_);
	transformer_->set_rand_mirror(has_mirror_);
	transformer_->set_rand_mirror(norm_map_);
	transformer_->set_color_offset(color_offset_);
	transformer_->set_rotate_degree(rotate_degree_);
	transformer_->set_crop_h(crop_h_);
	transformer_->set_crop_w(crop_w_);
	// Reshape the blobs
	Reshape(bottom, top);
}

template <typename Dtype>
void SyncTransformLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	// Reshape the top blobs, make sure the shape of each top blob equal with
	// the corresponding bottom blob
	CHECK_EQ(bottom.size(), top.size());
	for(int i = 0; i < bottom.size(); ++i){
		top[i]->Reshape(bottom[i]->num(), bottom[i]->channels(), bottom[i]->height()-crop_h_*2, bottom[i]->width()-crop_w_*2);
		// top[i]->ReshapeLike(*bottom[i]);
	}
}

template <typename Dtype>
void SyncTransformLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	// Start the transform the blobs
	transformer_->transform();
}

#ifdef CPU_ONLY
STUB_GPU(SyncTransformLayer);
#endif

INSTANTIATE_CLASS(SyncTransformLayer);
REGISTER_LAYER_CLASS(SyncTransform);

}  // namespace caffe


#endif	// CAFFE_SYNCTRANSFORM_LAYER_HPP_
