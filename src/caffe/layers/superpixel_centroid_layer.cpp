#include <vector>
#include <cfloat>

#include "caffe/layers/superpixel_centroid_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SuperpixelCentroidLayer<Dtype>::LayerSetUp(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
	// The bottom[0] is a superpixel map, which channels is 1
	CHECK_EQ(bottom[0]->channels(), 1);

	// Parse the params
	SuperpixelCentroidParameter superpixel_centroid_param = this->layer_param_.superpixel_centroid_param();
	// NOTE: The num_output_ must be equal with the number of superpixels
	if (superpixel_centroid_param.has_num_output()){
		num_output_ = superpixel_centroid_param.num_output();
	}else{
		LOG(FATAL) << "The num_output must be set equal to the number of the superpixels";
	}

	if (superpixel_centroid_param.has_check()){
		check_ = superpixel_centroid_param.check();
	}else{
		check_ = true;
	}

	if (superpixel_centroid_param.has_normalize()){
		normalize_ = superpixel_centroid_param.normalize();
	}else{
		normalize_ = false;
	}
}

template <typename Dtype>
void SuperpixelCentroidLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	// Reshape the top
	top[0]->Reshape(bottom[0]->num(), 1, num_output_, 2);
	// Reshape the tmp blobs used in this layer
	sp_accum_.Reshape(bottom[0]->num(), 1, num_output_, 2);
	sp_num_.Reshape(bottom[0]->num(), 1, num_output_, 1);
	height_ = bottom[0]->height();
	width_ = bottom[0]->width();
}

template <typename Dtype>
void SuperpixelCentroidLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	const Dtype* sp_data = bottom[0]->cpu_data();
	Dtype* top_data = top[0]->mutable_cpu_data();
	Dtype* sp_accum_data = sp_accum_.mutable_cpu_data();
	Dtype* sp_num_data = sp_num_.mutable_cpu_data();
	// Check the max id of the superpixel map, which indicates the number of the
	// superpixel number
	const int max_label = int(caffe_amax(bottom[0]->count(), sp_data));
	if(max_label + 1 != num_output_){
		if (check_){
			LOG(FATAL) << "The num_output and max superpixel+1 not match: "<<num_output_<<" vs "<<max_label+1;
		}else{
			LOG(WARNING) << "The num_output and max superpixel+1 not match: "<<num_output_<<" vs "<<max_label+1;
		}
	}

	// Clear the sp_accum_ and sp_num_
	caffe_set(sp_accum_.count(), Dtype(0), sp_accum_data);
	caffe_set(sp_num_.count(), Dtype(0), sp_num_data);
	// Iterate the whole superpixel map, and accumulate the w and h for each superpixel
	for(int n = 0; n < bottom[0]->num(); n++){
		for (int c = 0; c < bottom[0]->channels(); c++){
			for (int h = 0; h < bottom[0]->height(); h++){
				int sp_offset = bottom[0]->offset(n, c, h);
				for (int w = 0; w < bottom[0]->width(); w++){
					int sp_id = int(sp_data[sp_offset+w]);
					if (sp_id < num_output_){
						const int sp_accum_offset = sp_accum_.offset(n, c, sp_id);
						const int sp_num_offset = sp_num_.offset(n, c, sp_id);
						// Accumulate the h and w coordinate
						sp_accum_data[sp_accum_offset] += h;
						sp_accum_data[sp_accum_offset+1] += w;
						// Add the counter
						sp_num_data[sp_num_offset] += 1;
					}
				}
			}
		}
	}
	// Average the sp_accum_ to calc the central coordinate
	for(int n = 0; n < bottom[0]->num(); n++){
		for(int c = 0; c < bottom[0]->channels(); c++){
			for(int h = 0; h < top[0]->height(); h++){
				// The h is the number of superpixel
				const int offset_accum = sp_accum_.offset(n, c, h);
				const int offset_num = sp_num_.offset(n, c, h);
				// W1: height
				// W2: width
				top_data[offset_accum] = sp_accum_data[offset_accum] / Dtype(sp_num_data[offset_num]);
				top_data[offset_accum+1] = sp_accum_data[offset_accum+1] / Dtype(sp_num_data[offset_num]);
				if (normalize_){
					top_data[offset_accum] = top_data[offset_accum] / height_;
					top_data[offset_accum+1] = top_data[offset_accum+1] / width_;
				}
			}
		}
	}
	
}

INSTANTIATE_CLASS(SuperpixelCentroidLayer);
REGISTER_LAYER_CLASS(SuperpixelCentroid);

}  // namespace caffe
