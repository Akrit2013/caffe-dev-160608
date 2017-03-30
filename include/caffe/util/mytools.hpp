/* This is the model which contains some useful function should be used in
 * my own code.
 * Added by YanHan
 */
#ifndef	CAFFE_MYTOOLS_HPP_
#define CAFFE_MYTOOLS_HPP_

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include <opencv2/opencv.hpp>
#include <vector>


namespace caffe {

template <typename Dtype>
void BlobToCVMat(const Blob<Dtype>& src, cv::Mat& dst);

template <typename Dtype>
void CVMatToBlob(const cv::Mat& src, Blob<Dtype>& dst);

/* This function can only convert a part of the blob, such as a
 * sample in the minibatch. So the shape of the image should
 * be provided.
 * The vector<int>  bottom_shape is a 3 dim vector, which contrains
 * (channels, height, width)
 */
template <typename Dtype>
void BlobToCVMat(const Dtype* src, cv::Mat& dst, vector<int> blob_shape);

/* This function do not need the shape of the dst blob, it just
 * assume the shape of blob is equal to the shape of the cv::Mat
 */
template <typename Dtype>
void CVMatToBlob(const cv::Mat& src, Dtype* dst);


/* Transform the datum into an empty blob
 * the mirror param indicate whether to mirror the image
 */
template <typename Dtype>
void DatumToBlob(Datum& datum, Blob<Dtype>& blob, const bool mirror=false);

template <typename Dtype>
void BlobToDatum(const Blob<Dtype>& blob, Datum& datum, const bool use_uint8=false);

// =========== The following function are used in cuda ===============
/*
 * This function perform the bilinear interpolation for batch images
 * The params:
 *	src_data	The data pointer contains the src data
 *	dst_data	The data pointer to store the result (not in-place operation)
 *	channels	The number of the channels or the size of the batch
 *	height		The height of the src_data
 *	width		The width of the src_data image
 *	new_height	The height of the dst_data
 *	new_width	The width of the dst_data
 * NOTE:
 *	The size of the src_data should be channels * height * width * sizeof(Dtype)
 *	The size of the dst_data should be channels * new_height * new_width * sizeof(Dtype)
 *	The memory arrangement is (channels, height, width)
 *	The new_height > height and new_width > width
 */
template <typename Dtype>
void cuBilinearInterpolation(const Dtype* const src_data, Dtype* dst_data, const int channels,
		const int height, const int width, const int new_height, const int new_width);

/*
 * This function is a much complex version of the cuBilinearInterpolation
 * It take two exact same size memory, src_data and dst_data, which is (n, c, h, w)
 * The src_data will first cropped according to the three memory:
 * h_size_rate: The crop rate of the height
 * w_size_rate: The crop rate of the width
 * h_offset_rate: The offset rate of the cropped height
 * w_offset_rate: The offset rate of the cropped width
 * All these four them are in size (n, 1, 1, 1), that means each image in minibatch
 * have ites own crop parameters, the value contains must between 0 and 1
 *
 * This function will crop it first and resize them back to the original size
 */
template <typename Dtype>
void cuCropAndResizeBack(const Dtype* const src_data, Dtype* dst_data,
		const Dtype* const h_size_rate, const Dtype* const w_size_rate,
		const Dtype* const h_offset_rate, const Dtype* const w_offset_rate,
		const int num, const int channels, const int height, const int width);

/*
 * This function generate the pairwise map from the superpixel map and the segment
 * boundary map.
 */
template <typename Dtype>
void GenPairwiseMap_gpu(const Blob<Dtype>& blob_sup, const Blob<Dtype>& blob_seg, Blob<Dtype>& blob_pair);

/*
 * This function check if the NaN or exist in the given memory section
 * If exist, return true, else return false
 * It is useful in debug process
 */
template <typename Dtype>
bool CheckNan(const int count, const Dtype* data);


}		// namespace caffe


#endif	// CAFFE_MYTOOLS_HPP_
