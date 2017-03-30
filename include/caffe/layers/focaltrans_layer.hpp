#ifndef CAFFE_FOCALTRANS_LAYER_HPP_
#define CAFFE_FOCALTRANS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/rand_crop.hpp"

namespace caffe {


/**
 * Added by YanHan
 * This layer randomly crop the datalayer and change the corresponding focal
 * according to the crop rate
 * Currently, It only support 2 input layers, the first layer is the data image
 * and the second layer is the label focal value. BUT it is quite easy to
 * modify it to support mult input layers.
 */
template <typename Dtype>
class FocalTransformLayer: public Layer<Dtype> {
	public:
		explicit FocalTransformLayer(const LayerParameter& param)
			: Layer<Dtype>(param) {}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
				const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
				const vector<Blob<Dtype>*>& top);

		// Define the type the label can be
		enum HviewType {
			degree,
			radian
		};
	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
				const vector<Blob<Dtype>*>& top);
		// Do not need to backward this layer
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
				const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}

		double crop_rate_;
		double crop_prob_;
		double rotate_degree_;
		Dtype min_hview_;
		HviewType label_type_;

		shared_ptr<RandCrop<Dtype> > transformer_;
};

}  // namespace caffe

#endif  // CAFFE_FOCALTRANS_LAYER_HPP_
