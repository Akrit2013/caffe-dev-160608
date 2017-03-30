#ifndef CAFFE_SYNCTRANS_FAST_LAYER_HPP_
#define CAFFE_SYNCTRANS_FAST_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include <boost/random.hpp>


namespace caffe {

/**
 * Added by YanHan
 * This layer transform all of the bottom layers using the same
 * transform parameter, and feed them into the corrensponding tops
 */
template <typename Dtype>
class SyncTransFastLayer: public Layer<Dtype> {
	public:
		explicit SyncTransFastLayer(const LayerParameter& param)
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

		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
				const vector<Blob<Dtype>*>& top);

		// Generate the random parameters according to the setting
		void random_params(void);

	protected:
		// If perform the central crop
		bool central_crop_;
		// If need to mirror
		bool mirror_;

		// The crop height and width
		int crop_height_;
		int crop_width_;
		// The color offset
		float color_offset_;

		// This controls the random crop and resized back
		float crop_rate_;

		// The index of the bottom to perform the color offset
		vector<int> color_index_;
		// A list of bool indicate which bottom need color transform
		// The length of the list equal to the bottom.size
		vector<bool> color_list_;

		// The index of the bottom is the normal map type
		vector<int> norm_index_;
		// A list of bool indicate which bottom is the normal map type
		vector<bool> norm_list_;

		// The index of the bottom is the depth map type
		vector<int> depth_index_;
		// A list of bool indicate which bottom is the depth map type
		vector<bool> depth_list_;

		// The index of the bottom is the grad map type
		vector<int> grad_index_;
		// A list of bool indicate which bottom is the grad map
		vector<bool> grad_list_;

		// The index of bottom is the pairwise map
		vector<int>	pairwise_index_;
		// A list of bool indicate if the bottom is the pairwise map
		vector<bool> pairwise_list_;

		// The index of bottom is the superpixel map
		vector<int> superpixel_index_;
		// A list of bool indicate if the bottom is the superpixel map
		vector<bool> superpixel_list_;

		// The random parameters for crop height and width offset for the minibatch
		// which is a float between 0 and 1
		Blob<Dtype> batch_crop_h_offset_;
		Blob<Dtype> batch_crop_w_offset_;
		// The random paramter for mirror of each images in the minibatch
		// 1 indicate mirror the image, 0 indicate not
		Blob<Dtype> batch_mirror_;
		// The random paramter for color offset of each image in the minibatch
		// Different from other blobs, the size of this blobs is (n, c, 1, 1)
		// The n is the size of the minibatch, and the c is the channels of the
		// input image
		// NOTE: When multiple color index set, it will use the channel of the
		// first color image
		Blob<Dtype> batch_color_offset_;

		// This param controlled by the crop_rate_, which random crop the image
		// and resize it back. This crop will stack after the crop_height_ or
		// crop_width_
		// These values are all between 0 and 1
		Blob<Dtype> batch_rand_crop_h_offset_;
		Blob<Dtype> batch_rand_crop_w_offset_;
		Blob<Dtype> batch_rand_crop_rate_;
};

}  // namespace caffe

#endif  // CAFFE_SYNCTRANS_FAST_LAYER_HPP_
