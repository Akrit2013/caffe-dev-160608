#ifndef CAFFE_RAND_CROP_
#define CAFFE_RAND_CROP_
/**
 * This class can be used in FocalTransformLayer.
 * It randomly crop the bottom image (each image in a minibatch has its own crop
 * param).
 * This class works in the following steps:
 * 1. Set the necessary parameters, like crop rate, crop prob, etc
 * 2. The class will generate a vector<vector<float>> randomly, which contains all crop
 *    parameters in this bottom minibatch
 * 3. The vector<float> can be modified by user and then, it will be passed to
 *    this crop object again to do the real crop
 *
 * NOTE:
 * We use a vector<float> indicate the crop parameters for each image, which is
 * a 3-dim float vector: [crop_rate, h_offset_rate, w_offset_rate]
 * It will use muti thread from boost lib. Each transformation will use an indenpend
 * thread to speed up the job
 */

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include <boost/thread/thread.hpp>


namespace caffe{

// The crop thread class
template <typename Dtype>
class CropThread {
	public:
		CropThread(const Dtype* data_bottom,
				Dtype* data_top,
				vector<double> crop_param,
				vector<int> bottom_shape):
			data_bottom_(data_bottom),
			data_top_(data_top),
			param_(crop_param),
			bottom_shape_(bottom_shape)	{
		}

		// The main thread funcion
		void operator()();

	protected:
		// Calc the rotated image's offset, which will be used
		// to crop out the black edge
		// It will take the height and width of the image, and the rotate degree
		// as the input param
		// the rstH_offset and rstW_offset is the output param
		void calc_rotate_offset(double imgH, double imgW, double degree, int* rstH_offset, int* rstW_offset);

	protected:
		const Dtype*	data_bottom_;
		Dtype*			data_top_;
		const vector<double> param_;
		// Indicate the bottom blob shape: (channel, height, width)
		const vector<int>  bottom_shape_;
};


template <typename Dtype>
class RandCrop {
	public:
		RandCrop(const Blob<Dtype>* bottom,
				Blob<Dtype>* top): bottom_(bottom), top_(top) {
			CHECK_EQ(bottom->num(), top->num());
		}
		// Set the crop rate
		void set_crop_rate(double crop_rate) {crop_rate_ = crop_rate;}
		// Set the crop prob
		void set_crop_prob(double crop_prob) {crop_prob_ = crop_prob;}
		// Set the crop is center crop
		void set_center_crop(bool center_crop) {center_crop_ = center_crop;}
		// Set the rotate degree
		void set_rotate_degree(double rotate_degree) {rotate_degree_ = rotate_degree;}
		// Start to transform the blobs
		void crop(std::vector<std::vector<double> > crop_param);
		// Re-generate the transformer parameters
		// And return the crop param for each image in the minibatch
		// The format of this param is
		// [crop_rate, offset_h_rate, offset_w_rate, rotate_degree]
		std::vector<std::vector<double> > randparam(void);


	protected:
		const Blob<Dtype>* bottom_;
		Blob<Dtype>* top_;
		// The thread pool
		vector<shared_ptr<boost::thread> > thread_pool_;

		double	crop_rate_; 
		double  crop_prob_;
		double	rotate_degree_;
		bool    center_crop_;
		std::vector<std::vector<double> > crop_param_;
};



}	// namespace caffe

#endif		// CAFFE_RAND_CROP_
