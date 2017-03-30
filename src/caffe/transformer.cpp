#include "caffe/transformer.hpp"
#include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/random.hpp>
#include "caffe/common.hpp"

#include <opencv2/opencv.hpp>
#include "caffe/util/mytools.hpp"
#include <math.h>

#define PI 3.14159265

boost::random::mt19937	rng_(time(0));

namespace caffe {

template <typename Dtype>
void Transformer<Dtype>::set_crop_rate(double crop_rate){
	if(crop_rate == 1){
		has_crop_ = false;
		crop_rate_ = 1;
	}else{
		has_crop_ = true;
		crop_rate_ = crop_rate;
	}
}

template <typename Dtype>
void Transformer<Dtype>::set_rand_mirror(bool has_mirror){
	has_mirror_ = has_mirror;
}

template <typename Dtype>
void Transformer<Dtype>::set_norm_map(bool norm_map){
	norm_map_ = norm_map;
}

template <typename Dtype>
void Transformer<Dtype>::set_color_offset(double color_offset){
	color_offset_ = color_offset;
}

template <typename Dtype>
void Transformer<Dtype>::set_rotate_degree(double rotate_degree){
	rotate_degree_ = rotate_degree;
}

template <typename Dtype>
void Transformer<Dtype>::set_crop_h(int crop_h){
	crop_h_ = crop_h;
}

template <typename Dtype>
void Transformer<Dtype>::set_crop_w(int crop_w){
	crop_w_ = crop_w;
}

template <typename Dtype>
void Transformer<Dtype>::randparam(TransformerParameter& param){
	// The random seed
	// ---- crop ----
	param.crop_w = crop_w_;
	param.crop_h = crop_h_;
	if(has_crop_){
		// Generate crop size
		param.has_crop = true;
		boost::uniform_real<double>	real(crop_rate_, 1);
		param.crop_size_rate = real(rng_);
		// Generate the crop offset
		boost::uniform_01<boost::mt19937&>  randoffset(rng_);
		param.crop_offset_h_rate = randoffset();
		param.crop_offset_w_rate = randoffset();
		DLOG(INFO)<<"crop rate:"<<param.crop_size_rate<<" w offset:"<<param.crop_offset_w_rate<<" h offset:"<<param.crop_offset_h_rate;
	}else{
		param.has_crop = false;
	}
	// --------mirror-------
	if(has_mirror_){
		// Random to choose whether mirror or not
		boost::uniform_real<double> rand_mirror(0,1);
		if(rand_mirror(rng_)>0.5){
			param.mirror = true;
		}else{
			param.mirror = false;
		}
	}else{
		param.mirror = false;
	}
	// -------color offset-------
	if(color_offset_ != 0){
		// Random generate the color offset of the rgb channels
		double offset;
		param.has_color_tune = true;
		// Channel 1
		boost::uniform_real<double> rand_channel1_offset(0,color_offset_);
		boost::uniform_real<double> rand_channel1_sign(0, 1);
		offset = rand_channel1_offset(rng_);
		if(rand_channel1_sign(rng_)>0.5){
			param.channel1_mult = 1 + offset;
		}else{
			param.channel1_mult = 1 - offset;
		}
		// Channel 2
		boost::uniform_real<double> rand_channel2_offset(0,color_offset_);
		boost::uniform_real<double> rand_channel2_sign(0, 1);
		offset = rand_channel2_offset(rng_);
		if(rand_channel2_sign(rng_)>0.5){
			param.channel2_mult = 1 + offset;
		}else{
			param.channel2_mult = 1 - offset;
		}
		// Channel 3
		boost::uniform_real<double> rand_channel3_offset(0,color_offset_);
		boost::uniform_real<double> rand_channel3_sign(0, 1);
		offset = rand_channel3_offset(rng_);
		if(rand_channel3_sign(rng_)>0.5){
			param.channel3_mult = 1 + offset;
		}else{
			param.channel3_mult = 1 - offset;
		}
	}else{
		param.has_color_tune = false;
	}
	// ---------rotate image----------
	if(rotate_degree_ != 0){
		// Random generate the rotate degree
		param.has_rotate = true;
		// random degree
		boost::uniform_real<double> rand_rotate_degree(0, rotate_degree_);
		param.rotate_degree = rand_rotate_degree(rng_);
		// random sign
		boost::uniform_real<double> rand_rotate_sign(0, 1);
		if(rand_rotate_sign(rng_)>0.5){
			param.rotate_degree = -param.rotate_degree;
		}

		double radians = param.rotate_degree * PI / 180.0;
		param.rotate_sin = sin(radians);
		param.rotate_cos = cos(radians);
	}else{
		param.has_rotate = false;
	}
		
}


template <typename Dtype>
void Transformer<Dtype>::transform(void){
	/* This function will start a couple of threads which will transform each
	 * bottom blob according to the TransformerParameter and save the result
	 * in corresponding top blob.
	 */
	// Create threads for each image in the minibatch
	for (int j = 0; j < bottom_[0]->num(); ++j){
		// Generate random parameters for each data in minibatch
		TransformerParameter param;
		randparam(param);
		for(int i = 0; i < bottom_.size(); ++i){
			// Get the offset of each memory slice of the current blob
			const Dtype* data_bottom = bottom_[i]->cpu_data();
			Dtype* data_top = top_[i]->mutable_cpu_data();
			int bottom_offset = bottom_[i]->offset(j);
			int top_offset = top_[i]->offset(j);

			vector<int> bottom_shape;
			bottom_shape.push_back(bottom_[i]->channels());
			bottom_shape.push_back(bottom_[i]->height());
			bottom_shape.push_back(bottom_[i]->width());

			// Here we assume the bottom[0] is data image
			// and the bottom[1] is the label image, which is the
			// normal mat
			// Also assume other bottoms are images
			if(i==1){
				if(bottom_[i]->channels()==3){
					if(norm_map_==true){
						param.is_normal_map = true;
						param.is_rgb_map = false;
						param.is_pixel_map = false;
					}else{
						param.is_normal_map = false;
						param.is_rgb_map = true;
						param.is_pixel_map = false;
					}
				}else{
					param.is_normal_map = false;
					param.is_rgb_map = false;
					param.is_pixel_map = true;
				}
			}else{
				if(bottom_[i]->channels()==3){
					param.is_normal_map = false;
					param.is_rgb_map = true;
					param.is_pixel_map = false;
				}else{
					param.is_pixel_map = true;
					param.is_normal_map = false;
					param.is_rgb_map = false;
				}
			}

			shared_ptr<boost::thread> hThread;
			hThread.reset(new boost::thread(TransformerThread<Dtype>(data_bottom+bottom_offset, data_top+top_offset, param, bottom_shape)));
			thread_pool_.push_back(hThread);
		}
	}
	// Wait until all the threads are finished 
	for(int i = 0; i < thread_pool_.size(); ++i){
		thread_pool_[i]->join();
	}
}


template <typename Dtype>
void TransformerThread<Dtype>::operator()(){
	/* This function is a single thread, and it will do the transform job
	 * according to the TransformerParameter from bottom to top
	 *
	 * This function will turn the blob into opencv::mat first, and do the
	 * transform on the mat, in the end, the result will be changed back
	 * to blob data format.
	 */
	int top_height = bottom_shape_[1]-param_.crop_h*2;
	int top_width = bottom_shape_[2]-param_.crop_w*2;
	// First, convert the blob data format into a cv::Mat
	cv::Mat	matImg;
	cv::Mat matImg_precrop;
	BlobToCVMat(data_bottom_, matImg, bottom_shape_);

	if(param_.crop_h != 0 || param_.crop_w != 0){
		matImg_precrop = matImg(cv::Rect(param_.crop_w, param_.crop_h, top_width, top_height));
	}else{
		matImg_precrop = matImg;
	}
	// Crop it first if needed

	// Write the mat for debug
	/*
	cv::Mat matstore;
	if(bottom_shape_[0]==1){
		LOG(INFO)<<"*************The channels is 1**************";
		cv::normalize(matImg, matstore, 0, 255, cv::NORM_MINMAX, CV_8UC1);
		//cv::imshow("label",matstore);
		cv::imwrite("label.bmp", matstore);
	}else{
		LOG(INFO)<<"*************The channels is 3************";
		//cv::imshow("data",matImg);
		cv::normalize(matImg, matstore, 0, 255, cv::NORM_MINMAX, CV_8UC3);
		cv::imwrite("data.bmp", matstore);
	}
	*/

	// The image rotation must be done before the crop
	cv::Mat matImg_rotate;
	if(param_.has_rotate){
		// The raw rotated image, before crop
		cv::Mat matImg_rotate_raw;
		// Rotate the rgb image and the norm image
		// Generate the rotate matrix
		// Positive values mean counter-clockwise rotation
		cv::Mat rotmat = cv::getRotationMatrix2D(cv::Point2f(matImg_precrop.cols/2.0, matImg_precrop.rows/2.0), param_.rotate_degree, 1);
		// The interlance mode can be set by the last params
		cv::warpAffine(matImg_precrop, matImg_rotate_raw, rotmat, cv::Size(matImg_precrop.cols, matImg_precrop.rows), cv::INTER_LINEAR, cv::BORDER_CONSTANT);
		// The norm map should be changed according to the rotation degree
		if(param_.is_normal_map){
			// Positive values mean counter-clockwise rotation
			// X = xcos - ysin
			// Y = xsin + ycos
			// The channels order is RGB, which means XYZ
			// First, split the channels
			vector<cv::Mat> bgr_channel;
			cv::split(matImg_rotate_raw, bgr_channel);
			cv::Mat x_channel = bgr_channel[0];
			cv::Mat y_channel = bgr_channel[1];
			
			cv::Mat x_channel_new = x_channel * param_.rotate_cos + y_channel * param_.rotate_sin;
			cv::Mat y_channel_new = -x_channel * param_.rotate_sin + y_channel * param_.rotate_cos;

			// Merge the new channels
			bgr_channel[0] = x_channel_new;
			bgr_channel[1] = y_channel_new;
			cv::Mat merged_channels;
			cv::merge(bgr_channel, merged_channels);
			matImg_rotate_raw = merged_channels;
		}
		// Crop the image to avoid the empty edge
		// Calc the crop offset first
		int crop_H_offset;
		int crop_W_offset;
		calc_rotate_offset(matImg_rotate_raw.rows, matImg_rotate_raw.cols, param_.rotate_degree, &crop_H_offset, &crop_W_offset);

		// For debug
		// LOG(INFO)<<"H offset:"<<crop_H_offset<<"W offset:"<<crop_W_offset<<"rotate degree:"<<param_.rotate_degree;
		// Crop the rotated image to avoid the black edge
		int rotate_crop_width = matImg_rotate_raw.cols - 2 * crop_W_offset;
		int rotate_crop_Height = matImg_rotate_raw.rows - 2 * crop_H_offset;
		matImg_rotate = matImg_rotate_raw(cv::Rect(crop_W_offset, crop_H_offset, rotate_crop_width, rotate_crop_Height)); 

		// Write the image For debug
		/*
		cv::normalize(matImg_rotate_raw, matstore, 0, 255, cv::NORM_MINMAX, CV_8UC3);
		cv::imwrite("rotated_data_raw.bmp", matstore);
		*/

	}else{
		matImg_rotate = matImg_precrop;
	}
	/* // Write the image For debug
	cv::normalize(matImg_rotate, matstore, 0, 255, cv::NORM_MINMAX, CV_8UC3);
	cv::imwrite("rotated_data.bmp", matstore);
	*/

	// Crop the mat as we need
	// The crop pixel value should be calculated first
	cv::Mat	matImg_crop;
	int org_height = matImg_rotate.rows;
	int org_width = matImg_rotate.cols;
	if(param_.has_crop){
		const int crop_height = round(param_.crop_size_rate * org_height);
		const int crop_width = round(param_.crop_size_rate * org_width);
		const int crop_offset_h = round(param_.crop_offset_h_rate * (org_height - crop_height));
		const int crop_offset_w = round(param_.crop_offset_w_rate * (org_width - crop_width));
		CHECK_LE(crop_height + crop_offset_h, org_height);
		CHECK_LE(crop_width + crop_offset_w, org_width);
		// Crop the image according to the crop pixel value
		matImg_crop = matImg_rotate(cv::Rect(crop_offset_w, crop_offset_h, crop_width, crop_height));
	}else{
		matImg_crop = matImg_rotate;
	}

	/*  // Write the image For debug
	cv::normalize(matImg_crop, matstore, 0, 255, cv::NORM_MINMAX, CV_8UC3);
	cv::imwrite("croped_data.bmp", matstore);
	*/
	

	// Flip the image if needed
	cv::Mat matImg_flipped;
	if(param_.mirror){
		if(param_.is_normal_map){
			// Change the normal vector X direction before flip
			vector<cv::Mat> bgr_channel;
			cv::split(matImg_crop, bgr_channel);
			bgr_channel[0] = bgr_channel[0] * -1;
			cv::merge(bgr_channel, matImg_crop);
		}
		// Mirror the image in hor way
		cv::flip(matImg_crop, matImg_flipped, 1);
	}else{
		matImg_flipped = matImg_crop;
	}
	// Tune the color channel if needed
	cv::Mat matImg_color_tune;
	if(param_.has_color_tune && param_.is_rgb_map){
		// Split the channels
		vector<cv::Mat> bgr_channel;
		cv::split(matImg_flipped, bgr_channel);
		if(bgr_channel.size() == 1){
			bgr_channel[0] = bgr_channel[0] * param_.channel1_mult;
		}else{
			bgr_channel[0] = bgr_channel[0] * param_.channel1_mult;
			bgr_channel[1] = bgr_channel[1] * param_.channel2_mult;
			bgr_channel[2] = bgr_channel[2] * param_.channel3_mult;
		}
		cv::merge(bgr_channel, matImg_color_tune);
	}else{
		matImg_color_tune = matImg_flipped;
	}

	// Resize the mat back to the original size
	cv::Mat	matImg_resize_back;
	//cv::resize(matImg_crop, matImg_resize_back, cv::Size(top_width, top_height), 0, 0, CV_INTER_CUBIC);
	cv::resize(matImg_color_tune, matImg_resize_back, cv::Size(top_width, top_height), 0, 0);

	/*  // Write the image for debug
	cv::normalize(matImg_resize_back, matstore, 0, 255, cv::NORM_MINMAX, CV_8UC3);
	cv::imwrite("resized_data.bmp", matstore);
	*/
	

	// Change the cv::Mat back to blob
	CVMatToBlob(matImg_resize_back, data_top_);
	
}

template <typename Dtype>
void TransformerThread<Dtype>::calc_rotate_offset(double imgH, double imgW, double degree, int* rstH_offset, int* rstW_offset){
	double alpha = fabs(PI*degree/180);
	double beta = atan(imgH/imgW);
	double m = imgH / 2 * tan(alpha / 2);
	double z = imgW / 2 - m;
	double x = sin(alpha) * z / sin(PI-beta-alpha);
	*rstH_offset = sin(beta) * x;
	*rstW_offset = cos(beta) * x;
}


INSTANTIATE_CLASS(Transformer);
}	// namespace caffe
