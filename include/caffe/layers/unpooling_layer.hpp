#ifndef CAFFE_UNPOOLING_LAYER_HPP_
#define CAFFE_UNPOOLING_LAYER_HPP_
/*
 * @brief Does unpooling operation on the network like Zeiler's paper in ECCV 2014
 * TODO(mariolew) Through documentation on the useage of unpooling layer.
 */
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe{

	template <typename Dtype>
		class UnpoolingLayer : public Layer <Dtype> {
			public:
				explicit UnpoolingLayer(const LayerParameter &param)
					: Layer<Dtype>(param) {}
				virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
						const vector<Blob<Dtype>*>& top);
				virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
						const vector<Blob<Dtype>*>& top);
				virtual inline const char* type() const { return "Unpooling";  }
				//virtual inline int MinBottomBlobs() const { return 1;  }
				//virtual inline int MaxBottomBlobs() const { return 2;  }
				virtual inline int ExactNumBottomBlobs() const { return 1; }
				virtual inline int ExactNumTopBlobs() const { return 1;  }

			protected:
				virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
						const vector<Blob<Dtype>*>& top);
				// virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
				//                          const vector<Blob<Dtype>*>& top);
				virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
						const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
				// virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
				//                                  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

				int kernel_h_, kernel_w_;
				int stride_h_, stride_w_;
				int pad_u_, pad_d_, pad_l_, pad_r_;
				int channels_;
				int height_, width_;
				int unpooled_height_, unpooled_width_;


		};

}  // namespace caffe

#endif
