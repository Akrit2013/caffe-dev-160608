#include "caffe/rand_crop.hpp"
#include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/random.hpp>
#include "caffe/common.hpp"

#include <opencv2/opencv.hpp>
#include "caffe/util/mytools.hpp"
#include <math.h>

#define PI 3.14159265

namespace caffe {

template <typename Dtype>
std::vector<std::vector<double> > RandCrop<Dtype>::randparam(void){
	// The random seed
	boost::random::mt19937	rng_(time(0));
	// Clear the original param
	crop_param_.clear();
	// ---- crop ----
	// Generate the crop param for each image in the minibatch
	for (int i = 0; i < bottom_->num(); ++i){
		// First, judge whether this image need to be cropped
		boost::uniform_01<boost::mt19937&>  crop_prob_gen(rng_);
		double curr_crop_rate = 0;
		double curr_offset_h_rate = 0;
		double curr_offset_w_rate = 0;
		double curr_rotate_degree = 0;

		if (crop_prob_gen() < crop_prob_){
			// Need to crop
			boost::uniform_real<double>	real(crop_rate_, 1);
			boost::uniform_real<double>	rot_real(0, rotate_degree_);

			curr_crop_rate = real(rng_);
			// Generate the crop offset
			if (center_crop_){
				curr_offset_h_rate = 0.5;
				curr_offset_w_rate = 0.5;
				curr_rotate_degree = rot_real(rng_);
				if(crop_prob_gen() < 0.5){
					curr_rotate_degree = -curr_rotate_degree;
				}
			}else{
				boost::uniform_01<boost::mt19937&>  randoffset(rng_);
				curr_offset_h_rate = randoffset();
				curr_offset_w_rate = randoffset();		
				// Disable the rotate when it is not center crop for now
				curr_rotate_degree = 0;
			}
		}else{
			// Need not to crop
			curr_crop_rate = 1;
			curr_offset_h_rate = 0;
			curr_offset_w_rate = 0;
			curr_rotate_degree = 0;
		}
		std::vector<double> vec_param;
		vec_param.push_back(curr_crop_rate);
		vec_param.push_back(curr_offset_h_rate);
		vec_param.push_back(curr_offset_w_rate);
		vec_param.push_back(curr_rotate_degree);
		crop_param_.push_back(vec_param);
	
		DLOG(INFO)<<"crop rate:"<<curr_crop_rate<<" w offset:"<<curr_offset_w_rate<<" h offset:"<<curr_offset_h_rate;
	}
	return crop_param_;
}


template <typename Dtype>
void RandCrop<Dtype>::crop(std::vector<std::vector<double> > crop_param){
	/* This function will start a couple of threads which will transform each
	 * bottom blob according to the crop_param and save the result
	 * in corresponding top blob.
	 */
	CHECK_EQ(crop_param.size(), bottom_->num());

	// Create threads for each image in the minibatch
	for (int j = 0; j < bottom_->num(); ++j){
		// Get the offset of each memory slice of the current blob
		const Dtype* data_bottom = bottom_->cpu_data();
		Dtype* data_top = top_->mutable_cpu_data();
		int bottom_offset = bottom_->offset(j);
		int top_offset = top_->offset(j);

		vector<int> bottom_shape;
		bottom_shape.push_back(bottom_->channels());
		bottom_shape.push_back(bottom_->height());
		bottom_shape.push_back(bottom_->width());

		shared_ptr<boost::thread> hThread;
		hThread.reset(new boost::thread(CropThread<Dtype>(data_bottom+bottom_offset, data_top+top_offset, crop_param[j], bottom_shape)));
		thread_pool_.push_back(hThread);
	}
	// Wait until all the threads are finished 
	for(int i = 0; i < thread_pool_.size(); ++i){
		thread_pool_[i]->join();
	}
}


template <typename Dtype>
void CropThread<Dtype>::operator()(){
	/* This function is a single thread, and it will do the transform job
	 * according to the crop_param from bottom to top
	 *
	 * This function will turn the blob into opencv::mat first, and do the
	 * transform on the mat, in the end, the result will be changed back
	 * to blob data format.
	 */
	int top_height = bottom_shape_[1];
	int top_width = bottom_shape_[2];

	// Check if the crop is needed
	if (param_[0] == 1){
		// No need to crop, directly copy the data
		const int len = bottom_shape_[0]*bottom_shape_[1]*bottom_shape_[2];
		caffe_copy(len, data_bottom_, data_top_);
	}else{

		// First, convert the blob data format into a cv::Mat
		cv::Mat	matImg;
		BlobToCVMat(data_bottom_, matImg, bottom_shape_);

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

		// Crop the mat as we need
		// The crop pixel value should be calculated first
		cv::Mat	matImg_crop;
		cv::Mat matImg_rotate;
		int org_height = matImg.rows;
		int org_width = matImg.cols;
		double crop_size_rate = param_[0];
		double crop_offset_h_rate = param_[1];
		double crop_offset_w_rate = param_[2];
		double rotate_degree = param_[3];

		const int crop_height = round(crop_size_rate * org_height);
		const int crop_width = round(crop_size_rate * org_width);
		const int crop_offset_h = round(crop_offset_h_rate * (org_height - crop_height));
		const int crop_offset_w = round(crop_offset_w_rate * (org_width - crop_width));
		// Calc the rotate crop offset
		int rot_offset_h;
		int rot_offset_w;
		calc_rotate_offset(matImg.rows, matImg.cols, rotate_degree, &rot_offset_h, &rot_offset_w);
		// Judge if the rotation is too much and exceed the targeted crop rate
		if (rot_offset_h < crop_offset_h && rot_offset_w < crop_offset_w && rotate_degree != 0){
			// Only under this condition, rotate first, and then crop the image
			cv::Mat rotmat = cv::getRotationMatrix2D(cv::Point2f(matImg.cols/2.0, matImg.rows/2.0), rotate_degree, 1);
			cv::warpAffine(matImg, matImg_rotate, rotmat, cv::Size(matImg.cols, matImg.rows), cv::INTER_LINEAR, cv::BORDER_CONSTANT);
		}else{
			// Skip the rotation
			matImg_rotate = matImg;
		}

		// Crop after the rotation
		CHECK_LE(crop_height + crop_offset_h, org_height);
		CHECK_LE(crop_width + crop_offset_w, org_width);
		// Crop the image according to the crop pixel value
		matImg_crop = matImg_rotate(cv::Rect(crop_offset_w, crop_offset_h, crop_width, crop_height));

		// Resize the mat back to the original size
		cv::Mat	matImg_resize_back;
		//cv::resize(matImg_crop, matImg_resize_back, cv::Size(top_width, top_height), 0, 0, CV_INTER_CUBIC);
		cv::resize(matImg_crop, matImg_resize_back, cv::Size(top_width, top_height), 0, 0);

		/*  // Write the image for debug
		cv::normalize(matImg_resize_back, matstore, 0, 255, cv::NORM_MINMAX, CV_8UC3);
		cv::imwrite("resized_data.bmp", matstore);
		*/
		

		// Change the cv::Mat back to blob
		CVMatToBlob(matImg_resize_back, data_top_);
	}
}

template <typename Dtype>
void CropThread<Dtype>::calc_rotate_offset(double imgH, double imgW, double degree, int* rstH_offset, int* rstW_offset){
	double alpha = fabs(PI*degree/180);
	double beta = atan(imgH/imgW);
	double m = imgH / 2 * tan(alpha / 2);
	double z = imgW / 2 - m;
	double x = sin(alpha) * z / sin(PI-beta-alpha);
	*rstH_offset = sin(beta) * x;
	*rstW_offset = cos(beta) * x;
}


INSTANTIATE_CLASS(RandCrop);
}	// namespace caffe
