/*************************************************************************
	> File Name: unpooling_layer.cpp
	> Author: YanHan
	> Mail:
	> Created Time: 2015年11月16日 星期一 15时42分28秒
NOTE:
1. This layer use the open lib to resize the feature size. zoom in or zoom out
2. Currently, This layer dose not have Backward function.
 ************************************************************************/

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/resize_layer.hpp"
#include "caffe/util/mytools.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void ResizeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  ResizeParameter resize_param = this->layer_param_.resize_param();
  // Currently, only support 1 bottom and 1 top
  CHECK(bottom.size()==1) << "The ResizeLayer only can have 1 bottom blob";
  CHECK(top.size()==1) << "The ResizeLayer only can have 1 top blob";

  // Check the parameters
  CHECK(!((resize_param.has_zoom() || resize_param.has_zoom_w() || resize_param.has_zoom_h()) && (resize_param.has_height() || resize_param.has_width()))) << "zoom and height, width can not co-exist";

  CHECK(!(resize_param.has_zoom() && (resize_param.has_zoom_h() || resize_param.has_zoom_w()))) << "zoom and zoom_w, zoom_h can not co-exist";

  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();

  // Init the param
  if(resize_param.has_zoom()){
	  float zoom = resize_param.zoom();
	  resize_height_ = height_ * zoom;
	  resize_width_ = width_ * zoom;
  }else if (resize_param.has_zoom_w() && resize_param.has_zoom_h()){
	  float zoom_h = resize_param.zoom_h();
	  float zoom_w = resize_param.zoom_w();
	  resize_height_ = static_cast<int>(height_ * zoom_h);
	  resize_width_ = static_cast<int>(width_ * zoom_w);
  }else if (resize_param.has_height() && resize_param.has_width()){
	  resize_height_ = resize_param.height();
	  resize_width_ = resize_param.width();
  }

  if (resize_param.has_inter()){
	  switch (resize_param.inter()){
		  case ResizeParameter_Interpolation_INTER_NEAREST:
			  interpolation_ = cv::INTER_NEAREST;
			  break;
		  case ResizeParameter_Interpolation_INTER_LINEAR:
			  interpolation_ = cv::INTER_LINEAR;
			  break;
		  case ResizeParameter_Interpolation_INTER_AREA:
			  interpolation_ = cv::INTER_AREA;
			  break;
		  case ResizeParameter_Interpolation_INTER_CUBIC:
			  interpolation_ = cv::INTER_CUBIC;
			  break;
		  case ResizeParameter_Interpolation_INTER_LANCZOS4:
			  interpolation_ = cv::INTER_LANCZOS4;
			  break;
		  default:
			  LOG(FATAL)<<"Unknown Interpolation Mode";
	  }
  }else{
	 interpolation_ = cv::INTER_LINEAR;
  }
}



template <typename Dtype>
void ResizeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();

  ResizeParameter resize_param = this->layer_param_.resize_param();
  if(resize_param.has_zoom()){
	  float zoom = resize_param.zoom();
	  resize_height_ = height_ * zoom;
	  resize_width_ = width_ * zoom;
  }else if (resize_param.has_zoom_w() && resize_param.has_zoom_h()){
	  float zoom_h = resize_param.zoom_h();
	  float zoom_w = resize_param.zoom_w();
	  resize_height_ = static_cast<int>(height_ * zoom_h);
	  resize_width_ = static_cast<int>(width_ * zoom_w);
  }

  top[0]->Reshape(bottom[0]->num(), channels_, resize_height_, resize_width_);
}



template <typename Dtype>
void ResizeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype *bottom_data = bottom[0]->cpu_data();
  Dtype *top_data = top[0]->mutable_cpu_data();

  vector<int> bottom_shape;
  bottom_shape.push_back(bottom[0]->channels());
  bottom_shape.push_back(bottom[0]->height());
  bottom_shape.push_back(bottom[0]->width());

  for (int j = 0; j < bottom[0]->num(); ++j){
	  int bottom_offset = bottom[0]->offset(j);
	  int top_offset = top[0]->offset(j);
	  resize(bottom_data+bottom_offset, top_data+top_offset, bottom_shape);
  }
}


template <typename Dtype>
void ResizeLayer<Dtype>::resize(const Dtype* data_bottom, Dtype* data_top, vector<int> bottom_shape){
	// First, convert the blob data format into a cv::Mat
	cv::Mat	matImg;

	BlobToCVMat(data_bottom, matImg, bottom_shape);
	cv::Mat	matImg_resize;
	cv::resize(matImg, matImg_resize, cv::Size(resize_width_, resize_height_), 0, 0, interpolation_);

	// Change the cv::Mat back to blob
	CVMatToBlob(matImg_resize, data_top);
}

//We need only CPU version.
// STUB_GPU(UnPoolingLayer);

INSTANTIATE_CLASS(ResizeLayer);
REGISTER_LAYER_CLASS(Resize);
} //namespace caffe
