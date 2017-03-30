#ifndef CAFFE_SYNCTRANS_LAYER_HPP_
#define CAFFE_SYNCTRANS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/base_conv_layer.hpp"
#include "caffe/transformer.hpp"

namespace caffe {


/**
 * Added by YanHan
 * This layer transform all of the bottom layers using the same
 * transform parameter, and feed them into the corrensponding tops
 */
template <typename Dtype>
class SyncTransformLayer: public Layer<Dtype> {
	public:
		explicit SyncTransformLayer(const LayerParameter& param)
			: Layer<Dtype>(param) {}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
				const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
				const vector<Blob<Dtype>*>& top);
	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
				const vector<Blob<Dtype>*>& top);
		// Do not need to backward this layer
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
				const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}

		float crop_rate_;
		bool  has_mirror_;
		bool  norm_map_;
		float color_offset_;
		float rotate_degree_;
		int crop_h_;
		int crop_w_;

		shared_ptr<Transformer<Dtype> > transformer_;
};

}  // namespace caffe

#endif  // CAFFE_SYNCTRANS_LAYER_HPP_
