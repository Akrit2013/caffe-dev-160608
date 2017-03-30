#include "caffe/util/mytools.hpp"
#include "caffe/util/io.hpp"


namespace caffe {


/* The data format of Blob is (n, c, h, w), and the data format of
 * cv::Mat is (h, w, c). We assume the n is 1 here, and transform
 * the blob format into mat format.
 */
template <typename Dtype>
void BlobToCVMat(const Blob<Dtype>& src, cv::Mat& dst){
	// Make sure the blob only contains 1 image
	CHECK_EQ(src.num(), 1);
	const Dtype* data_blob = src.cpu_data();
	vector<int> blob_shape;
	blob_shape.push_back(src.channels());
	blob_shape.push_back(src.height());
	blob_shape.push_back(src.width());

	BlobToCVMat(data_blob, dst, blob_shape);
}


template <typename Dtype>
void CVMatToBlob(const cv::Mat& src, Blob<Dtype>& dst){
	CHECK_EQ(dst.num(), 1);
	Dtype* data_blob = dst.mutable_cpu_data();
	CVMatToBlob(src, data_blob);
}


template <typename Dtype>
void BlobToCVMat(const Dtype* src, cv::Mat& dst, vector<int> blob_shape){
	// Create a new mat according to the blob_shape
	// TODO: Currently this function can only support channels == 1 or
	// channels == 3 which indicate the rgb image and the gray image
	const int channels = blob_shape[0];
	const int height = blob_shape[1];
	const int width = blob_shape[2];

	CHECK(channels == 1 || channels == 3) <<
			"Currently, only support channels == 1 or channels == 3";
	if(channels==1){
		dst.create(height, width, CV_32FC1);
	}else if(channels==3){
		dst.create(height, width, CV_32FC3);
	}else{
		LOG(ERROR)<<"ERROR: The channels should be 1 or 3, in this case it is:"
			<<channels<<std::endl;
	}
	// First, make sure the memory in this mat is continous
	if(!dst.isContinuous()){
		dst = dst.clone();
		LOG(WARNING)<<"WARNING: The mat is not continuous";
	}
	// Get the first byte address of the mat data
	float* data_mat = dst.ptr<float>();

	// Iter the blob to set the mat
	for (int c = 0; c < channels; ++c){
		for (int h = 0; h < height; ++h){
			for (int w = 0; w < width; ++w){
				// Calc the blob and the mat index
				int blob_index = (c*height+h)*width+w;
				int mat_index = (h*width+w)*channels+c;
				data_mat[mat_index] = static_cast<float>(src[blob_index]);
			}
		}
	}
}

/*
template <typename Dtype>
void BlobToCVMat(const Dtype* src, cv::Mat& dst, vector<int> blob_shape){
	// Create a new mat according to the blob_shape
	// TODO: Currently this function can only support channels == 1 or
	// channels == 3 which indicate the rgb image and the gray image
	const int channels = blob_shape[0];
	const int height = blob_shape[1];
	const int width = blob_shape[2];

	CHECK(channels == 1 || channels == 3) <<
			"Currently, only support channels == 1 or channels == 3";
	if(channels==1){
		dst.create(height, width, CV_32FC1);
	}else if(channels==3){
		dst.create(height, width, CV_32FC3);
	}else{
		LOG(ERROR)<<"ERROR: The channels should be 1 or 3, in this case it is:"
			<<channels<<std::endl;
	}

	// Iter the blob to set the mat
	for (int c = 0; c < channels; ++c){
		for (int h = 0; h < height; ++h){
			float* row_data = dst.ptr<float>(h);
			for (int w = 0; w < width; ++w){
				// Calc the blob and the mat index
				int blob_index = (c*height+h)*width+w;
				int mat_index = w*channels+c;
				row_data[mat_index] = static_cast<float>(src[blob_index]);
			}
		}
	}
}
*/
template <typename Dtype>
void CVMatToBlob(const cv::Mat& src, Dtype* dst){
	const int channels = src.channels();
	const int height = src.rows;
	const int width = src.cols;
	// First, make sure the memory in this mat is continous
	cv::Mat src2;
	src2 = src;
	if(!src.isContinuous()){
		src2 = src.clone();
		LOG(WARNING)<<"WARNING: The mat is not continuous";
	}

	float* data_mat = src2.ptr<float>();
	// Iter the mat to set the blob
	for (int c = 0; c < channels; ++c){
		for (int h = 0; h < height; ++h){
			for (int w = 0; w < width; ++w){
				// Calc the blob and the mat index
				int blob_index = (c*height+h)*width+w;
				int mat_index = (h*width+w)*channels+c;
				dst[blob_index] = static_cast<Dtype>(data_mat[mat_index]);
			}
		}
	}
}



template <typename Dtype>
void DatumToBlob(Datum& datum, Blob<Dtype>& blob, const bool mirror){
	const int channels = datum.channels();
	const int height = datum.height();
	const int width = datum.width();

	blob.Reshape(1, channels, height, width);
	// Try to decode the datum
	DecodeDatumNative(&datum);

	const std::string& data = datum.data();
	Dtype* blob_data = blob.mutable_cpu_data();

	if (data.size() != 0) {
		CHECK_EQ(data.size(), channels * height * width);
		for (int c = 0; c < channels; c++){
			for (int h = 0; h < height; h++){
				for (int w = 0; w < width; w++){
					int idx = (c * height + h) * width + w;
					int idx2 = 0;
					if (mirror){
						idx2 = (c * height + h) * width + width - 1 - w;
					}else{
						idx2 = idx;
					}
					blob_data[idx] = (uint8_t)data[idx2];
				}
			}
		}
		/*
		for (int i = 0; i < data.size(); ++i) {
			blob_data[i] = (uint8_t)data[i];
		}
		*/
	} else {
		CHECK_EQ(datum.float_data_size(), channels * height * width);
		for (int c = 0; c < channels; c++){
			for (int h = 0; h < height; h++){
				for (int w = 0; w < width; w++){
					int idx = (c * height + h) * width + w;
					int idx2 = 0;
					if (mirror){
						idx2 = (c * height + h) * width + width - 1 - w;
					}else{
						idx2 = idx;
					}
					blob_data[idx] = static_cast<float>(datum.float_data(idx2));
				}
			}
		}
		/*
		for (int i = 0; i < datum.float_data_size(); ++i) {
			blob_data[i] = static_cast<float>(datum.float_data(i));
		}
		*/
	}
}

template <typename Dtype>
void BlobToDatum(const Blob<Dtype>& blob, Datum& datum, const bool use_uint8){
	/* Currently only set the float data
	 */
	CHECK_EQ(blob.num(), 1) << "The number of the blob must be 1";
	datum.set_channels(blob.channels());
	datum.set_height(blob.height());
	datum.set_width(blob.width());
	datum.clear_label();
	datum.clear_data();
	datum.clear_float_data();
	datum.set_encoded(false);

	int datum_channels = datum.channels();
	int datum_height = datum.height();
	int datum_width = datum.width();
	int datum_size = datum_channels * datum_height * datum_width;

	const Dtype* blob_data = blob.cpu_data();

	if (use_uint8){
		string buffer(datum_size, ' ');
		for (int i = 0; i < datum_size; i++){
			uint8_t ch = static_cast<uint8_t>(blob_data[i] + 0.5);
			buffer[i] = ch;
		}
		datum.set_data(buffer);
	}else{
		for (int i = 0; i < blob.count(); i++){
			datum.add_float_data(blob_data[i]);
		}
	}
}

template <typename Dtype>
bool CheckNan(const int count, const Dtype* data){
	for (int i = 0; i < count; i++){
		if (isnan(data[i])) return true;
		if (isinf(data[i])) return true;
	}
	return false;
}


// Explicit instantiation
template void BlobToCVMat<float>(const Blob<float>& src, cv::Mat& dst);
template void BlobToCVMat<double>(const Blob<double>& src, cv::Mat& dst);

template void CVMatToBlob<float>(const cv::Mat& src, Blob<float>& dst);
template void CVMatToBlob<double>(const cv::Mat& src, Blob<double>& dst);

template void BlobToCVMat<float>(const float* src, cv::Mat& dst, vector<int> blob_shape);
template void BlobToCVMat<double>(const double* src, cv::Mat& dst, vector<int> blob_shape);

template void CVMatToBlob<float>(const cv::Mat& src, float* dst);
template void CVMatToBlob<double>(const cv::Mat& src, double* dst);

template void DatumToBlob<float>(Datum& datum, Blob<float>& blob, const bool mirror);
template void DatumToBlob<double>(Datum& datum, Blob<double>& blob, const bool mirror);

template void BlobToDatum<float>(const Blob<float>& blob, Datum& datum, const bool use_uint8);
template void BlobToDatum<double>(const Blob<double>& blob, Datum& datum, const bool use_uint8);

template bool CheckNan<float>(const int count, const float* data);
template bool CheckNan<double>(const int count, const double* data);

}		// namespace caffe
