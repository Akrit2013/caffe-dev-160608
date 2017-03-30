#include "caffe/layers/synctrans_fast_layer.hpp"
#include "caffe/util/mytools.hpp"


boost::random::mt19937	rng(time(0));
namespace caffe {

template <typename Dtype>
void SyncTransFastLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	CHECK_EQ(bottom.size(), top.size());
	// Get the transform parameter
	SyncTransFastParameter transform_param = this->layer_param_.synctransfast_param();
	// Check if need to random mirror the image
	// --------------- Check the optional param ----------------
	if(transform_param.has_mirror()){
		mirror_ = transform_param.mirror();
	}else{
		mirror_ = false;
	}
	// Check if color channel need to be tuned
	if(transform_param.has_color_offset()){
		color_offset_ = transform_param.color_offset();
	}else{
		color_offset_ = 0;
	}
	// Check the central_crop_
	if(transform_param.has_central_crop()){
		central_crop_ = transform_param.central_crop();
	}else{
		central_crop_ = false;
	}
	// Check the color index
	int color_index_num = transform_param.color_index_size();
	color_index_.clear();
	if (color_index_num != 0){
		color_index_.resize(color_index_num, 0);
		for (int i = 0; i < color_index_num; i++){
			color_index_[i] = transform_param.color_index(i);
		}
	}else{
		// Use the first blob as the color input by default
		color_index_.resize(1, 0);
	}

	// Check the crop_rate_ param
	if(transform_param.has_crop_rate()){
		crop_rate_ = transform_param.crop_rate();
		CHECK_GT(crop_rate_, 0);
		CHECK_LE(crop_rate_, 1);
	}else{
		crop_rate_ = 1;
	}

	// --------------- Check the repeated param ----------------

	int norm_index_num = transform_param.norm_index_size();
	norm_index_.clear();
	if (norm_index_num != 0){
		norm_index_.resize(norm_index_num);
		for (int i = 0; i < norm_index_num; i++){
			norm_index_[i] = transform_param.norm_index(i);
			// The normal map must have 3 channels
			CHECK_EQ(bottom[norm_index_[i]]->channels(), 3)<<"The normal map \
must have 3 channels";
		}
	}

	int pairwise_index_num = transform_param.pairwise_index_size();
	pairwise_index_.clear();
	if (pairwise_index_num != 0){
		pairwise_index_.resize(pairwise_index_num);
		for (int i = 0; i < pairwise_index_num; i++){
			pairwise_index_[i] = transform_param.pairwise_index(i);
		}
	}

	int superpixel_index_num = transform_param.superpixel_index_size();
	superpixel_index_.clear();
	if (superpixel_index_num != 0){
		superpixel_index_.resize(superpixel_index_num);
		for (int i = 0; i < superpixel_index_num; i++){
			superpixel_index_[i] = transform_param.superpixel_index(i);
		}
	}

	int depth_index_num = transform_param.depth_index_size();
	depth_index_.clear();
	if (depth_index_num != 0){
		depth_index_.resize(depth_index_num);
		for (int i = 0; i < depth_index_num; i++){
			depth_index_[i] = transform_param.depth_index(i);
		}
	}

	int grad_index_num = transform_param.grad_index_size();
	grad_index_.clear();
	if (grad_index_num != 0){
		grad_index_.resize(grad_index_num);
		for (int i = 0; i < grad_index_num; i++){
			grad_index_[i] = transform_param.grad_index(i);
		}
	}


	if(transform_param.has_crop_height()){
		crop_height_ = transform_param.crop_height();
	}else{
		crop_height_ = 0;
	}

	if(transform_param.has_crop_width()){
		crop_width_ = transform_param.crop_width();
	}else{
		crop_width_ = 0;
	}

	// ================== Check the capacity of the params ==============
	//
	// Make sure the crop_width_ and crop_height_ small than the input
	// width and height
	for (int i = 0; i < bottom.size(); i++){
		CHECK_GE(bottom[i]->height(), crop_height_);
		CHECK_GE(bottom[i]->width(), crop_width_);
	}

	// Check if the parameter are compatiable to each other
	// TODO: Currently, the superpixel map can not support the crop
	if ((crop_height_ != 0 || crop_width_ != 0 || crop_rate_ != 1) && (superpixel_index_num != 0 || pairwise_index_num !=0)){
		LOG(FATAL) << "ERROR: Currently, the superpixel, pariwise layer can not co-exist with the crop paramters";
	}
}

template <typename Dtype>
void SyncTransFastLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	// Reshape the top blobs, make sure the shape of each top blob equal with
	// the corresponding bottom blob
	CHECK_EQ(bottom.size(), top.size());
	const int batch_size = bottom[0]->num();

	// Init the batch parameters blobs
	batch_crop_h_offset_.Reshape(batch_size, 1, 1, 1);
	batch_crop_w_offset_.Reshape(batch_size, 1, 1, 1);
	batch_mirror_.Reshape(batch_size, 1, 1, 1);
	int color_channels = 0;
	color_channels = bottom[color_index_[0]]->channels();
	batch_color_offset_.Reshape(batch_size, color_channels, 1, 1);

	// Init the value of these blobs
	caffe_set(batch_crop_h_offset_.count(), Dtype(0), batch_crop_h_offset_.mutable_cpu_data());
	caffe_set(batch_crop_w_offset_.count(), Dtype(0), batch_crop_w_offset_.mutable_cpu_data());
	caffe_set(batch_mirror_.count(), Dtype(0), batch_mirror_.mutable_cpu_data());
	caffe_set(batch_color_offset_.count(), Dtype(1), batch_color_offset_.mutable_cpu_data());

	// The batch param for the crop_rate
	batch_rand_crop_rate_.Reshape(batch_size, 1, 1, 1);
	caffe_set(batch_rand_crop_rate_.count(), Dtype(1), batch_rand_crop_rate_.mutable_cpu_data());

	batch_rand_crop_h_offset_.Reshape(batch_size, 1, 1, 1);
	caffe_set(batch_rand_crop_h_offset_.count(), Dtype(0), batch_rand_crop_h_offset_.mutable_cpu_data());

	batch_rand_crop_w_offset_.Reshape(batch_size, 1, 1, 1);
	caffe_set(batch_rand_crop_w_offset_.count(), Dtype(0), batch_rand_crop_w_offset_.mutable_cpu_data());

	// Init the color_list_ according to the color_index_
	color_list_.clear();
	color_list_.resize(bottom.size(), false);
	for (int i = 0; i < color_index_.size(); i++){
		const int idx = color_index_[i];
		CHECK_LT(idx, bottom.size());
		CHECK_GE(idx, 0);
		color_list_[idx] = true;
	}

	// Init the norm_list_ according to the norm_index_
	norm_list_.clear();
	norm_list_.resize(bottom.size(), false);
	for (int i = 0; i < norm_index_.size(); i++){
		const int idx = norm_index_[i];
		CHECK_LT(idx, bottom.size());
		CHECK_GE(idx, 0);
		norm_list_[idx] = true;
	}
	// Init the pairwise_list_ according to the pairwise_index_
	pairwise_list_.clear();
	pairwise_list_.resize(bottom.size(), false);
	for (int i = 0; i < pairwise_index_.size(); i++){
		const int idx = pairwise_index_[i];
		CHECK_LT(idx, bottom.size());
		CHECK_GE(idx, 0);
		pairwise_list_[idx] = true;
		CHECK_EQ(bottom[idx]->width(), bottom[idx]->height()) << "The height and width of the pairwise map must be the same";
	}
	// Init the superpixel_list_ according to the superpixel_index_
	superpixel_list_.clear();
	superpixel_list_.resize(bottom.size(), false);
	for (int i = 0; i < superpixel_index_.size(); i++){
		const int idx = superpixel_index_[i];
		CHECK_LT(idx, bottom.size());
		CHECK_GE(idx, 0);
		superpixel_list_[idx] = true;
	}
	// Init the depth_list_ according to the depth_index_
	depth_list_.clear();
	depth_list_.resize(bottom.size(), false);
	for (int i = 0; i < depth_index_.size(); i++){
		const int idx = depth_index_[i];
		CHECK_LT(idx, bottom.size());
		CHECK_GE(idx, 0);
		depth_list_[idx] = true;
	}
	// Init the grad_list_ according to the grad_index_
	grad_list_.clear();
	grad_list_.resize(bottom.size(), false);
	for (int i = 0; i < grad_index_.size(); i++){
		const int idx = grad_index_[i];
		CHECK_LT(idx, bottom.size());
		CHECK_GE(idx, 0);
		grad_list_[idx] = true;
	}

	// Reshape the top shape
	for (int i = 0; i < bottom.size(); i++){
		int height = (crop_height_ == 0)? bottom[i]->height() : crop_height_;
		int width = (crop_width_ == 0)? bottom[i]->width() : crop_width_;
		int channels = (superpixel_list_[i])? 1 : bottom[i]->channels();
		// When the mirror is set, the input superpixel bottom must have 2 channels
		// instead of the normal one
		if (superpixel_list_[i] && bottom[i]->channels() != 2){
			LOG(FATAL) << "ERROR: When the mirror is set, the superpixel input must have 2 channels represent both the original and mirror superpixel map";
		}

		top[i]->Reshape(bottom[i]->num(), channels, height, width);
		CHECK_EQ(batch_size, bottom[i]->num()) << "The batch size of each bottom must be the same";
	}
}

template <typename Dtype>
void SyncTransFastLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	// Deside the random parameter for the transform
	random_params();

	// Start to transform the data for each blobs
	const int blobs_num = bottom.size();

	for (int i = 0; i < blobs_num; i++){
		const Dtype* bottom_data = bottom[i]->cpu_data();
		Dtype* top_data = top[i]->mutable_cpu_data();
		const int top_width = top[i]->width();
		// Iter the minibatch
		for (int n = 0; n < bottom[i]->num(); n++){
			// Deside the start and end index of the source according to the
			// crop info
			int h_start_idx = 0;
			int h_end_idx = bottom[i]->height();

			int w_start_idx = 0;
			int w_end_idx = bottom[i]->width();

			if (crop_height_ != 0){
				Dtype offset_rate = batch_crop_h_offset_.cpu_data()[n];
				int margin = bottom[i]->height() - crop_height_;
				CHECK_GE(margin, 0);
				int offset = round(offset_rate * margin);
				h_start_idx = offset;
				h_end_idx = h_start_idx + crop_height_;
				CHECK_LE(h_end_idx, bottom[i]->height());
			}

			if (crop_width_ != 0){
				Dtype offset_rate = batch_crop_w_offset_.cpu_data()[n];
				int margin = bottom[i]->width() - crop_width_;
				CHECK_GE(margin, 0);
				int offset = round(offset_rate * margin);
				w_start_idx = offset;
				w_end_idx = w_start_idx + crop_width_;
				CHECK_LE(w_end_idx, bottom[i]->width());
			}

			bool do_mirror;
			if (mirror_){
				do_mirror = batch_mirror_.cpu_data()[n] > 0.5 ? true: false;
			}else{
				do_mirror = false;
			}
			// If the current blobs indicate a superpixel blob and mirror
			// is required, copy the second channel, otherwise copy the first
			// channel
			if (superpixel_list_[i] == true){
				const Dtype* bottom_data_n;
				Dtype* top_data_n = top_data + top[i]->offset(n);
				if (do_mirror){
					bottom_data_n = bottom_data + bottom[i]->offset(n, 1);
				}else{
					bottom_data_n = bottom_data + bottom[i]->offset(n);
				}
				caffe_copy(top[i]->count(2), bottom_data_n, top_data_n);
				//TODO: Not support crop
				continue;
			}

			// If the current blob is a pairwise map, the mirror indicate
			// that use the upper trangle H < W instead of the lower trangle
			// H > W to fill the whole map
			if (pairwise_list_[i] == true){
				for(int c = 0; c < top[i]->channels(); c++){
					Dtype* top_data_nc = top_data + top[i]->offset(n, c);
					const Dtype* bottom_data_nc = bottom_data + bottom[i]->offset(n, c);
					if (do_mirror){
						for (int h = 0; h < top[i]->height(); h++){
							for (int w = 0; w < h; w++){
								top_data_nc[h*top[i]->width()+w] = bottom_data_nc[w*bottom[i]->width()+h];
								top_data_nc[w*top[i]->width()+h] = bottom_data_nc[w*bottom[i]->width()+h];
							}
						}
					}else{
						for (int h = 0; h < top[i]->height(); h++){
							for (int w = 0; w < h; w++){
								top_data_nc[h*top[i]->width()+w] = bottom_data_nc[h*bottom[i]->width()+w];
								top_data_nc[w*top[i]->width()+h] = bottom_data_nc[h*bottom[i]->width()+w];
							}
						}
					}
				}
				continue;
			}

			// Start to iter the channels
			for(int c = 0; c < top[i]->channels(); c++){
				// Calc the color multiple of the corrent channels
				Dtype color_multi = 1.0;
				if (color_offset_ != 0 && color_list_[i]){
					const Dtype* color_data = batch_color_offset_.cpu_data();
					color_multi += color_data[batch_color_offset_.offset(n, c)];
				}
				// Iter the height
				for (int h = h_start_idx; h < h_end_idx; h++){
					const Dtype* bottom_data_nch = bottom_data + bottom[i]->offset(n,c,h);
					Dtype* top_data_nch = top_data + top[i]->offset(n,c,h-h_start_idx);

					for (int w = w_start_idx; w < w_end_idx; w++){
						if (do_mirror){
							top_data_nch[top_width-1-(w-w_start_idx)] = color_multi * bottom_data_nch[w];
						}else{
							top_data_nch[w-w_start_idx] = color_multi * bottom_data_nch[w];
						}
					} // w
				} // h
			} // c
			// If the current image is a normal map or grad map and need to do mirror
			if (do_mirror && (norm_list_[i] || grad_list_[i])){
				// Change the direction of the X
				caffe_scal(top[i]->count(2), Dtype(-1), top_data + top[i]->offset(n, 0));
			}

			// If need to perform the random crop and resize back
			// It will load the data from the top into opencv mat, crop and resize
			// back and put it back to the top blob
			if (crop_rate_ != 1){
				vector<int> data_shape;
				const int top_channels = top[i]->channels();
				const int top_height = top[i]->height();
				const int top_width = top[i]->width();

				data_shape.push_back(top_channels);
				data_shape.push_back(top_height);
				data_shape.push_back(top_width);

				cv::Mat matImg;
				const Dtype* p_data = top_data + top[i]->offset(n);
				BlobToCVMat(p_data, matImg, data_shape);
				// Calc the offset h and w and the cropped height and width
				const Dtype* crop_rate_data = batch_rand_crop_rate_.cpu_data();
				const Dtype* crop_rate_h_data = batch_rand_crop_h_offset_.cpu_data();
				const Dtype* crop_rate_w_data = batch_rand_crop_w_offset_.cpu_data();
				const int rc_height = round(crop_rate_data[n] * top_height);
				const int rc_width = round(crop_rate_data[n] * top_width);
				const int rc_h_offset = round(crop_rate_h_data[n] * (top_height-rc_height));
				const int rc_w_offset = round(crop_rate_w_data[n] * (top_width-rc_width));
				CHECK_LE(rc_height+rc_h_offset, top_height);
				CHECK_LE(rc_width+rc_w_offset, top_width);
				// Crop the Mat
				cv::Mat matImg_cropped;
				matImg_cropped = matImg(cv::Rect(rc_w_offset, rc_h_offset, rc_width, rc_height));
				// Resize back to the original size
				cv::Mat matImg_resize_back;
				cv::resize(matImg_cropped, matImg_resize_back, cv::Size(top_width, top_height), 0, 0);

				// Copy back the mat to top_data
				CVMatToBlob(matImg_resize_back, top_data + top[i]->offset(n));

				// If the current blob is a depth map
				// rescale the pixel value according to the crop rate
				if (depth_list_[i]){
					caffe_scal(top[i]->count(1), Dtype(crop_rate_data[n]), top_data + top[i]->offset(n));
				}

			}

		} // n
	} // i

			
}

template <typename Dtype>
void SyncTransFastLayer<Dtype>::random_params(void){
	const int batch_size = batch_crop_h_offset_.count();
	// Generate the paramter for crop if needed
	if (crop_width_ != 0 || crop_height_ != 0){
		Dtype* h_data = batch_crop_h_offset_.mutable_cpu_data();
		Dtype* w_data = batch_crop_w_offset_.mutable_cpu_data();
		if (central_crop_){
			for (int i = 0; i < batch_size; i++){
				h_data[i] = 0.5;
				w_data[i] = 0.5;
			}
		}else{
			boost::uniform_real<Dtype>	real(0, 1);
			for (int i = 0; i < batch_size; i++){
				h_data[i] = real(rng);
				w_data[i] = real(rng);
			}
		}
	}

	// Generate the parameter for mirror if needed
	if (mirror_){
		boost::uniform_real<Dtype>	real(0, 1);
		Dtype* mirror_data = batch_mirror_.mutable_cpu_data();
		for (int i = 0; i < batch_size; i++){
			Dtype val = real(rng);
			if (val > 0.5){
				mirror_data[i] = 1;
			}else{
				mirror_data[i] = 0;
			}
		}
	}

	// Generate the param for color offset if needed
	if (color_offset_ != 0){
		boost::uniform_real<Dtype>	real(0, color_offset_);
		boost::uniform_real<Dtype>	binary(0, 1);
		Dtype* co_data = batch_color_offset_.mutable_cpu_data();
		for (int i = 0; i < batch_color_offset_.count(); i++){
			co_data[i] = real(rng);
			if (binary(rng) > 0.5){
				co_data[i] = -co_data[i];
			}
		}
	}

	// Generate the rand param for the crop_rate_
	if (crop_rate_ != 1){
		boost::uniform_real<Dtype> real(crop_rate_, 1);
		boost::uniform_real<Dtype> real2(0, 1);
		Dtype* crop_rate_data = batch_rand_crop_rate_.mutable_cpu_data();
		Dtype* crop_rate_h_data = batch_rand_crop_h_offset_.mutable_cpu_data();
		Dtype* crop_rate_w_data = batch_rand_crop_w_offset_.mutable_cpu_data();

		for (int i = 0; i < batch_size; i++){
			crop_rate_data[i] = real(rng);
			crop_rate_h_data[i] = real2(rng);
			crop_rate_w_data[i] = real2(rng);
		}
	}
}

#ifdef CPU_ONLY
STUB_GPU(SyncTransFastLayer);
#endif

INSTANTIATE_CLASS(SyncTransFastLayer);
REGISTER_LAYER_CLASS(SyncTransFast);

}  // namespace caffe
