#ifndef CAFFE_TRANSFORMER_
#define CAFFE_TRANSFORMER_
/**
 * This class can be used in SyncTransformLayer. It will trainsfer multi bottom
 * blobs with the same setting and feed them into top blobs
 *
 * It will use muti thread from boost lib. Each transformation will use an indenpend
 * thread to speed up the job
 */

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include <boost/thread/thread.hpp>


namespace caffe{

// The structure which will store all of the transform parameters
class TransformerParameter {
	public:
		// Init the parameters
		TransformerParameter(){
			has_crop = false;
			crop_size_rate = 0;
			crop_offset_h_rate = 0;
			crop_offset_w_rate = 0;
			mirror = false;
			is_normal_map = false;
			is_rgb_map = false;
			is_pixel_map = false;
			has_color_tune = false;
			channel1_mult = 1;
			channel2_mult = 1;
			channel3_mult = 1;
		}

		bool		has_crop;
		bool		mirror;
		bool		has_color_tune;
		bool		has_rotate;

		double		crop_size_rate;
		double		crop_offset_w_rate;
		double		crop_offset_h_rate;
		
		double		rotate_degree;
		double		rotate_sin;
		double		rotate_cos;

		int			crop_h;
		int			crop_w;

		// If the image is normal map, will tune the normal
		// value when we mirror the image
		bool		is_normal_map;
		// If the image is rgb map, the color_offset param
		// will be effective
		bool		is_rgb_map;
		// The pixel map means it is consised of normal pixel
		// and the pixel will not be altered
		bool		is_pixel_map;
		// Rand color mult rate of rgb channels
		double		channel1_mult;
		double		channel2_mult;
		double		channel3_mult;
};

// The transformer thread class
template <typename Dtype>
class TransformerThread {
	public:
		TransformerThread(const Dtype* data_bottom,
				Dtype* data_top,
				TransformerParameter param,
				vector<int> bottom_shape):
			data_bottom_(data_bottom),
			data_top_(data_top),
			param_(param),
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
		const TransformerParameter param_;
		// Indicate the bottom blob shape: (channel, height, width)
		const vector<int>  bottom_shape_;
};


template <typename Dtype>
class Transformer {
	public:
		Transformer(const vector<Blob<Dtype>*>& bottom,
				const vector<Blob<Dtype>*>& top): bottom_(bottom), top_(top) {
			CHECK_EQ(bottom.size(), top.size());
		}
		// Set the crop rate
		void set_crop_rate(double crop_rate);
		// Enable random mirror or not
		void set_rand_mirror(bool bas_mirror);
		// Set the color offset
		void set_color_offset(double color_offset);
		// Set the max rotate degree
		void set_rotate_degree(double rotate_degree);
		// Set if the label image is a norm map
		void set_norm_map(bool norm_map);
		// Set the crop parameter
		void set_crop_h(int crop_h);
		void set_crop_w(int crop_w);
		// Start to transform the blobs
		void transform(void);
		// Re-generate the transformer parameters
		// And refresh the 
		void randparam(TransformerParameter&);

	protected:
		const vector<Blob<Dtype>*> bottom_;
		const vector<Blob<Dtype>*> top_;

		// The thread pool
		vector<shared_ptr<boost::thread> > thread_pool_;

		// The layer param
		bool	has_crop_;
		bool	has_mirror_;
		bool	norm_map_;
		double	crop_rate_; 
		double  color_offset_;
		double	rotate_degree_;

		int		crop_h_;
		int		crop_w_;
};



}	// namespace caffe

#endif		// CAFFE_TRANSFORMER_
