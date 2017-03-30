#include <vector>

#include "caffe/layers/pixelwise_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

	
__device__ double atomicAddD_pixelwiseloss(double* address, double val)
{
	unsigned long long int* address_as_ull =
		(unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed,
				__double_as_longlong(val +
					__longlong_as_double(assumed)));
	} while (assumed != old);
	return __longlong_as_double(old);
}

template <typename Dtype>
__device__ Dtype dot_stride2(
		const int count,
		const Dtype* const data1,
		const int stride1,
		const Dtype* const data2,
		const int stride2){
	// Perform the dot with stride
	int offset1 = 0;
	int offset2 = 0;

	Dtype accum = 0;
	for (int i = 0; i < count; i++){
		offset1 = i * stride1;
		offset2 = i * stride2;

		accum += data1[offset1] * data2[offset2];
	}

	return accum;
}

template <typename Dtype>
__device__ Dtype calc_distsq(
		const Dtype* const data1,
		const Dtype* const data2,
		const int im_height,
		const int im_width){
	const Dtype h1 = data1[0] / im_height;
	const Dtype w1 = data1[1] / im_width;
	const Dtype h2 = data2[0] / im_height;
	const Dtype w2 = data2[1] / im_width;

	const Dtype dh = h1 - h2;
	const Dtype dw = w1 - w2;

	return dh * dh + dw * dw;
}




template <typename Dtype>
__global__ void Forward_scaleinvariant_gpu_kernel(
		 const int nthreads,
		 const Dtype* const data_label,
		 Dtype* data_diff,
		 Dtype* bad_pixel_data,
		 const int num,
		 const int channels,
		 const int height,
		 const int width,
		 const Dtype max_label,
		 const Dtype min_label){
	CUDA_KERNEL_LOOP(index, nthreads){
		const int n = index / height;
		const int h = index % height;

		const int data_offset = (n*channels*height+h)*width;
		const int bad_pixel_idx = index;
		const int interval = height * width;

		// Iter the width and channels
		for (int w = 0; w < width; w++){
			// Iter the channels
			int err_counter = 0;
			for (int c = 0; c < channels; c++){
				const int idx = data_offset + c * interval + w;
				Dtype dataval = data_label[idx];

				if (dataval > max_label){
					err_counter++;
				}else if(dataval < min_label){
					err_counter++;
				}
			}

			// Only if all channels invalid, the pixel will be considered
			// as invalid
			if(err_counter == channels){
				bad_pixel_data[bad_pixel_idx] += channels;
				for (int c = 0; c < channels; c++){
					const int idx = data_offset + c * interval + w;
					data_diff[idx] = 0;
				}
			}
		}
	}
}

template <typename Dtype>
__global__ void Pixelwise_inference_kernel(
		 const int nthreads,
		 const Dtype* const dep_data,
		 const Dtype* const sp_data,
		 const Dtype* const norm_data,
		 const Dtype* const centroid_data,
		 Dtype* pred_data,
		 const int num,
		 const int height,
		 const int width,
		 const int stride,
		 const Dtype f_,
		 const Dtype z_thd,
		 const bool use_gradient,
		 const bool train_phase){
	CUDA_KERNEL_LOOP(index, nthreads){
		const int n = index / height;
		const int h = index % height;

		// NOTE: The channels must be 1
		// Calc the index of the prediction
		const int pred_idx_nh = (n * height + h) * width;
		// Iter the width
		for (int w = 0; w < width; w++){
			const int sp_id = sp_data[pred_idx_nh + w];
			const int centroid_idx = (n * stride + sp_id) * 2;
			const int norm_idx = n * 3 * stride + sp_id;
			const int dep_idx = n * stride + sp_id;

			const Dtype coord_h = centroid_data[centroid_idx];
			const Dtype coord_w = centroid_data[centroid_idx + 1];

			Dtype dx = 0;
			Dtype dy = 0;
			bool valid_norm = true;

			if (use_gradient){
				dx = norm_data[norm_idx];
				dy = norm_data[norm_idx + stride];
			}else{
				const Dtype x = norm_data[norm_idx];
				const Dtype y = norm_data[norm_idx + stride];
				Dtype z = norm_data[norm_idx + stride * 2];

				if (fabs(z) < z_thd){
					if (train_phase){
						z = z > 0 ? z_thd: -z_thd;
					}else{
						valid_norm = false;
					}
				}
				dx = - x / z;
				dy = - y / z;
				
			}

			const Dtype dep = dep_data[dep_idx];

			Dtype dep_proj = dep;
			if (valid_norm){
				// The current superpixel is valid
				const Dtype coord_diff_h = Dtype(h) - coord_h;
				const Dtype coord_diff_w = Dtype(w) - coord_w;
				dep_proj += dep / f_ * (coord_diff_h * dy + coord_diff_w * dx);
			}

			pred_data[pred_idx_nh + w] = dep_proj;
		}
	}
}

template <typename Dtype>
__global__ void Forward_gpu_kernel(
		 const int nthreads,
		 const Dtype* const data_label,
		 Dtype* data_diff,
		 Dtype* bad_pixel_data,
		 const int num,
		 const int channels,
		 const int height,
		 const int width,
		 const bool has_max_label,
		 const bool has_min_label,
		 const Dtype max_label,
		 const Dtype min_label,
		 const Dtype C){
	CUDA_KERNEL_LOOP(index, nthreads){
		const int n = index / height;
		const int h = index % height;

		const int data_offset = (n*channels*height+h)*width;
		const int bad_pixel_idx = index;
		const int interval = height * width;

		// Iter the width and channels
		for (int w = 0; w < width; w++){
			// Iter the channels
			int err_counter = 0;
			for (int c = 0; c < channels; c++){
				const int idx = data_offset + c * interval + w;
				Dtype dataval = data_label[idx];
				Dtype diffval = data_diff[idx];

				if (has_max_label && dataval > max_label){
					err_counter++;
				}else if(has_min_label && dataval < min_label){
					err_counter++;
				//}else if(has_invalid_label && fabs(dataval - invalid_label) < 0.0001){
				//	err_counter++;
				}
				// alter the diff value
				if (diffval > 0 && diffval < C){
					// L1
					data_diff[idx] = C;
				}else if(diffval < 0 && -diffval < C){
					data_diff[idx] = -C;
				}
				/*
				if (has_h_rate && diffval > H){
					data_diff[idx] = H;
				}else if(has_h_rate && -diffval > H){
					data_diff[idx] = -H;
				}
				*/
			}

			// Only if all channels invalid, the pixel will be considered
			// as invalid
			if(err_counter == channels){
				bad_pixel_data[bad_pixel_idx] += channels;
				for (int c = 0; c < channels; c++){
					const int idx = data_offset + c * interval;
					data_diff[idx] = 0;
				}
			}
		}
	}
}


template<typename Dtype>
__global__ void Backward_step1_gpu_kernel(
		const int nthreads,
		const Dtype* const diff_data,
		const Dtype* const centroid_data,
		const Dtype* const sp_data,
		const Dtype* const norm_data,
		const Dtype* const dep_data,
		const Dtype* const dep_gt_data,
		Dtype* accum_data,
		Dtype* counter_data,
		const int num,
		const int height,
		const int width,
		const int stride,
		const Dtype focal,
		const Dtype z_thd,
		const Dtype lambda,
		const Dtype lr_z,
		const Dtype radius,
		const bool use_gradient);

template<>
__global__ void Backward_step1_gpu_kernel<double>(
		const int nthreads,
		const double* const diff_data,
		const double* const centroid_data,
		const double* const sp_data,
		const double* const norm_data,
		const double* const dep_data,
		const double* const dep_gt_data,
		double* accum_data,
		double* counter_data,
		const int num,
		const int height,
		const int width,
		const int stride,
		const double focal,
		const double z_thd,
		const double lambda,
		const double lr_z,
		const double radius,
		const bool use_gradient){
	CUDA_KERNEL_LOOP(index, nthreads){
		const int n = index / height;
		const int h = index % height;

		// The basic index of the diff
		const int diff_idx_nh = (n * height + h) * width;
		// Iter the width
		for (int w = 0; w < width; w++){
			// The superpixel id
			const int sp_id = sp_data[diff_idx_nh + w];
			// The index for other blobs
			const int norm_idx = n * 3 * stride + sp_id;
			const int dep_idx = n * stride + sp_id;
			const int centroid_idx = (n * stride + sp_id) * 2;

			// The values
			const double dep = dep_data[dep_idx];
			const double coord_h = centroid_data[centroid_idx];
			const double coord_w = centroid_data[centroid_idx + 1];

			const int central_dep_gt_idx = (n * height + int(coord_h)) * width + int(coord_w);
			const double central_dep_gt = dep_gt_data[central_dep_gt_idx];
			const double central_dep_diff = dep - central_dep_gt;
			const double diff = diff_data[diff_idx_nh + w] - central_dep_diff;

			const double coord_diff_h = double(h) - coord_h;
			const double coord_diff_w = double(w) - coord_w;

			if (radius > 0){
				// only use the pixels close to the central pixel
				const double dist = sqrt(coord_diff_h * coord_diff_h + coord_diff_w * coord_diff_w);
				if (dist > radius){
					continue;
				}
			}

			if (use_gradient){
				double diff_x = diff * dep / focal * coord_diff_w;
				double diff_y = diff * dep / focal * coord_diff_h;

				atomicAddD_pixelwiseloss(accum_data+norm_idx, diff_x);
				atomicAddD_pixelwiseloss(accum_data+norm_idx+stride, diff_y);
			}else{
				const double x = norm_data[norm_idx];
				const double y = norm_data[norm_idx + stride];
				double z = norm_data[norm_idx + stride * 2];

				if (fabs(z) < z_thd){
					z = z > 0 ? z_thd : - z_thd;
				}
				const double regula = 2 * lambda * (x * x + y * y + z * z - 1);

				double diff_x = - diff * dep / focal * coord_diff_w / z + regula * x;
				double diff_y = - diff * dep / focal * coord_diff_h / z + regula * y;
				double diff_z = diff * dep / focal * (coord_diff_h * y + coord_diff_w * x) / z / z + regula * z;

				// Since the direction of the surface is ambiguity according to the gradient info
				// We assume the Z must be larger than 0
				// When update: z = z - diff_z
				if (z < z_thd){
					diff_z = -fabs(diff_z);
				}
				// Add the value
				atomicAddD_pixelwiseloss(accum_data+norm_idx, diff_x);
				atomicAddD_pixelwiseloss(accum_data+norm_idx+stride, diff_y);
				atomicAddD_pixelwiseloss(accum_data+norm_idx+stride * 2, lr_z * diff_z);
			}


			// Accumulate
			atomicAddD_pixelwiseloss((double*)(counter_data+norm_idx), double(1));
		}
	}
}


template<>
__global__ void Backward_step1_gpu_kernel<float>(
		const int nthreads,
		const float* const diff_data,
		const float* const centroid_data,
		const float* const sp_data,
		const float* const norm_data,
		const float* const dep_data,
		const float* const dep_gt_data,
		float* accum_data,
		float* counter_data,
		const int num,
		const int height,
		const int width,
		const int stride,
		const float focal,
		const float z_thd,
		const float lambda,
		const float lr_z,
		const float radius,
		const bool use_gradient){
	CUDA_KERNEL_LOOP(index, nthreads){
		const int n = index / height;
		const int h = index % height;

		// The basic index of the diff
		const int diff_idx_nh = (n * height + h) * width;
		// Iter the width
		for (int w = 0; w < width; w++){
			// The superpixel id
			const int sp_id = sp_data[diff_idx_nh + w];
			// The index for other blobs
			const int norm_idx = n * 3 * stride + sp_id;
			const int dep_idx = n * stride + sp_id;
			const int centroid_idx = (n * stride + sp_id) * 2;

			// The values
			const float dep = dep_data[dep_idx];
			const float coord_h = centroid_data[centroid_idx];
			const float coord_w = centroid_data[centroid_idx + 1];
			
			const int central_dep_gt_idx = (n * height + int(coord_h)) * width + int(coord_w);
			const float central_dep_gt = dep_gt_data[central_dep_gt_idx];
			const float central_dep_diff = dep - central_dep_gt;
			const float diff = diff_data[diff_idx_nh + w] - central_dep_diff;

			const float coord_diff_h = float(h) - coord_h;
			const float coord_diff_w = float(w) - coord_w;

			if (radius > 0){
				// only use the pixels close to the central pixel
				const float dist = sqrt(coord_diff_h * coord_diff_h + coord_diff_w * coord_diff_w);
				if (dist > radius){
					continue;
				}
			}

			if (use_gradient){
				float diff_x = diff * dep / focal * coord_diff_w;
				float diff_y = diff * dep / focal * coord_diff_h;

				atomicAdd(accum_data+norm_idx, diff_x);
				atomicAdd(accum_data+norm_idx+stride, diff_y);
			}else{
				const float x = norm_data[norm_idx];
				const float y = norm_data[norm_idx + stride];
				float z = norm_data[norm_idx + stride * 2];

				if (fabs(z) < z_thd){
					z = z > 0 ? z_thd : -z_thd;
				}
				const float regula = 2 * lambda * (x * x + y * y + z * z - 1);

				float diff_x = - diff * dep / focal * coord_diff_w / z + regula * x;
				float diff_y = - diff * dep / focal * coord_diff_h / z + regula * y;
				float diff_z = diff * dep / focal * (coord_diff_h * y + coord_diff_w * x) / z / z + regula * z;
				// Since the direction of the surface is ambiguity according to the gradient info
				// We assume the Z must be larger than 0
				// When update: z = z - diff_z
				if (z < z_thd){
					diff_z = -fabs(diff_z);
				}
				// Add the value
				atomicAdd(accum_data+norm_idx, diff_x);
				atomicAdd(accum_data+norm_idx+stride, diff_y);
				atomicAdd(accum_data+norm_idx+stride * 2, lr_z * diff_z);
			}

			// Accumulate
			atomicAdd((float*)(counter_data+norm_idx), float(1));
		}
	}
}

template<typename Dtype>
__global__ void Backward_step2_gpu_kernel(
			const int nthreads,
			const Dtype* const accum_data,
			const Dtype* const counter_data,
			Dtype* bottom_diff,
			const int num,
			const int stride,
			const Dtype beta,
			const bool use_gradient){
	CUDA_KERNEL_LOOP(index, nthreads){
		const int n = index / stride;
		const int h = index % stride;

		const int idx = n * stride + h;

		bottom_diff[idx] = beta * accum_data[idx] / counter_data[idx];
		bottom_diff[idx + stride] = beta * accum_data[idx + stride] / counter_data[idx];
		if (!use_gradient){
			bottom_diff[idx + stride * 2] = beta * accum_data[idx + stride * 2] / counter_data[idx];
		}
	}
}

template<typename Dtype>
__global__ void Backward_step3_gpu_kernel(
				const int nthreads,
				Dtype* bottom_diff,
				const Dtype H){
	CUDA_KERNEL_LOOP(index, nthreads){

		if (fabs(bottom_diff[index]) > H){
			bottom_diff[index] = bottom_diff[index] > 0 ? H: -H;
		}
	}
}

template <typename Dtype>
void PixelwiseLossLayer<Dtype>::Euclidean_loss_gpu(const Blob<Dtype>* gt, Blob<Dtype>* top){
	const Dtype* gt_data = gt->gpu_data();
	const Dtype* pred_data = Pred_.gpu_data();
	const int count = Pred_.count();
	caffe_gpu_sub(
			count,
			pred_data,
			gt_data,
			Pred_.mutable_gpu_diff());
	Dtype dot;
	caffe_gpu_dot(count, Pred_.gpu_diff(), Pred_.gpu_diff(), &dot);
	Dtype loss = dot / count / Dtype(2);
	top->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void PixelwiseLossLayer<Dtype>::ScaleInvariant_loss_gpu(const Blob<Dtype>* gt, Blob<Dtype>* top){
  CHECK_EQ(gt->channels(), 1);
	
  int count = gt->count();
  caffe_gpu_sub(
      count,
      Pred_.gpu_data(),
      gt->gpu_data(),
      Pred_.mutable_gpu_diff());

  Dtype* data_diff = Pred_.mutable_gpu_diff();
  Dtype* vecSum_data = vecSum_.mutable_cpu_data();
  const Dtype* data_label = gt->gpu_data();
  const int num = gt->num();
  const int channels = gt->channels();
  const int height = gt->height();
  const int width = gt->width();
  // Set the number of the kernel]
  const int num_kernels = num * height;
  // Set the bad_pixel_ buffer to 0
  Dtype* bad_pixel_data = bad_pixel_.mutable_gpu_data();
  caffe_gpu_set(bad_pixel_.count(), Dtype(0), bad_pixel_data);
  
    // Find the bad pixel and alter the diff
  if (has_min_label_ || has_max_label_){
	  Forward_scaleinvariant_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>(
			  num_kernels,
			  data_label,
			  data_diff,
			  bad_pixel_data,
			  num,
			  channels,
			  height,
			  width,
			  max_label_,
			  min_label_);
  }
  // The pixel number per image
  Dtype pixel_num = gt->count(1);

  // Calc the each image's valid pixel number in minibatch
  /*
  for (int n = 0; n < diff_.num(); n++){
	  if(is_adjust_pixel_num_){
		  Dtype val;
		  int offset = bad_pixel_.offset(n);
		  caffe_gpu_asum(height, bad_pixel_data + offset, &val);
		  vecValidPixelNum_data[n] = pixel_num - val;
	  }else{
		  vecValidPixelNum_data[n] = pixel_num;
	  }
  }
  */

  Dtype dot;
  caffe_gpu_dot(count, Pred_.gpu_diff(), Pred_.gpu_diff(), &dot);
  Dtype loss = dot / count / Dtype(2);

  // Calc the second term of the loss
  for (int n = 0; n < gt->num(); n++){
	  const Dtype* cdata_diff = Pred_.cpu_diff() + Pred_.offset(n);
	  Dtype valid_num = pixel_num;
	  Dtype vecSum = caffe_cpu_sum(pixel_num, cdata_diff);
	  vecSum_data[n] = vecSum;
	  loss -= vecSum_data[n] * vecSum_data[n] / valid_num / valid_num / gt->num() * delta_ / Dtype(2);
  }

  top->mutable_cpu_data()[0] = loss;
  // DLOG(INFO) << "valid pixel num:" << valid_pixel_num_ <<" Loss:" << loss;
}

template <typename Dtype>
void PixelwiseLossLayer<Dtype>::Berhu_loss_gpu(const Blob<Dtype>* gt, Blob<Dtype>* top){
  const int count = Pred_.count();
  const Dtype* label_data = gt->gpu_data();
  const Dtype* pred_data = Pred_.gpu_data();
  caffe_gpu_sub(
      count,
	  pred_data,
	  label_data,
      Pred_.mutable_gpu_diff());
  Dtype max_diff = 0;

  switch(c_rate_mode_){
	  case MAX:
		  // Get the abs max diff to determine the C
		  max_diff = caffe_gpu_amax(count, Pred_.gpu_diff(), 1);
		  // Calc the Threshold C
		  break;
	  case AVE:
		  // Calc the mean of the abs diff
		  caffe_gpu_asum(count, Pred_.gpu_diff(), &max_diff);
		  max_diff = max_diff / count;
		  break;
	  default:
		  LOG(FATAL) << "False c_rate_mode";
		  break;
  }
  Dtype C = fabs(max_diff * c_rate_);

  Dtype* data_diff = Pred_.mutable_gpu_diff();
  // const Dtype* data_pred = bottom[0]->cpu_data();
  const int num = Pred_.num();
  const int channels = Pred_.channels();
  const int height = Pred_.height();
  const int width = Pred_.width();
  // The number of kernel is num * height, process a row each time
  const int num_kernels = num * height;
  // Set the bad_pixel_ buffer to zero
  Dtype* bad_pixel_data = bad_pixel_.mutable_gpu_data();
  caffe_gpu_set(bad_pixel_.count(), Dtype(0), bad_pixel_data);
  // Find the bad pixel and alter the diff
  Forward_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>(
		 num_kernels,
		 label_data,
		 data_diff,
		 bad_pixel_data,
		 num,
		 channels,
		 height,
		 width,
		 has_max_label_,
		 has_min_label_,
		 max_label_,
		 min_label_,
		 C);
  CUDA_POST_KERNEL_CHECK;

  Dtype bad_pixel_count;
  caffe_gpu_asum(bad_pixel_.count(), bad_pixel_data, &bad_pixel_count);
  Dtype dot;
  caffe_gpu_dot(count, Pred_.gpu_diff(), Pred_.gpu_diff(), &dot);
  Dtype loss = dot / Dtype(2) / (count-bad_pixel_count);
  top->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void PixelwiseLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top){

  // Inference the pixelwise prediction
  Pixelwise_inference_gpu(bottom);

  // Copy the result if needed
  if (top.size() >= 2){
	  caffe_copy(Pred_.count(), Pred_.gpu_data(), top[1]->mutable_gpu_data());
  }

  // Calc the loss according to the Pred_ and the bottom[0]
  switch (loss_mode_){
	  case L2:
		  Euclidean_loss_gpu(bottom[1], top[0]);
		  break;
	  case Berhu:
		  Berhu_loss_gpu(bottom[1], top[0]);
		  break;
	  case ScaleInvariant:
		  ScaleInvariant_loss_gpu(bottom[1], top[0]);
		  break;
	  default:
		  LOG(FATAL)<<"Unknow loss_mode_ in PixelwiseLossLayer";
		  break;
  }

}


template <typename Dtype>
void PixelwiseLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	// BP the depth
	if (bp_depth_){
		LOG(FATAL)<<"The depth BP is not implemented yet";
	}
	// The BP will performed on the bottom[0] and bottom[2]
	const Dtype loss_weight = top[0]->cpu_diff()[0];
	Dtype beta;
	if (normalize_){
		beta = loss_weight / bottom[0]->count();
	}else{
		beta = loss_weight / bottom[0]->num();
	}

	
	const Blob<Dtype>* centroid_bottom = bottom[3];
	Blob<Dtype>* norm_bottom = bottom[4];
	const Blob<Dtype>* sp_bottom = bottom[2];
	const Blob<Dtype>* depth_bottom = bottom[0];
	const Blob<Dtype>* depth_gt_bottom = bottom[1];

	// BP the surface normal of each superpixel
    const Dtype* diff_data = Pred_.gpu_diff();
	const Dtype* centroid_data = centroid_bottom->gpu_data();
	const Dtype* sp_data = sp_bottom->gpu_data();
	const Dtype* norm_data = norm_bottom->gpu_data();
	const Dtype* dep_data = depth_bottom->gpu_data();
	const Dtype* dep_gt_data = depth_gt_bottom->gpu_data();

	Dtype* accum_data = normAccum_.mutable_gpu_data();
	Dtype* counter_data = normAccum_.mutable_gpu_diff();
	
	Dtype* bottom_diff = norm_bottom->mutable_gpu_diff();

	caffe_gpu_set(normAccum_.count(), Dtype(0), accum_data);
	caffe_gpu_set(normAccum_.count(), Dtype(0), counter_data);

	const int stride = norm_bottom->count(2);
	CHECK_EQ(stride, superpixel_num_);

	const int num = Pred_.num();
	const int height = Pred_.height();
	const int width = Pred_.width();

	// The number of kernels
	const int num_kernels = num * height;

	// Step1, calc the diff for each pixel
	Backward_step1_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>(
			num_kernels,
			diff_data,
			centroid_data,
			sp_data,
			norm_data,
			dep_data,
			dep_gt_data,
			accum_data,
			counter_data,
			num,
			height,
			width,
			stride,
			f_,
			z_thd_,
			lambda_,
			lr_z_,
			radius_,
			use_gradient_);
	CUDA_POST_KERNEL_CHECK;

	// Step2 calc the diff for each superpixel
	const int num_kernels2 = num * stride;
	Backward_step2_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels2), CAFFE_CUDA_NUM_THREADS>>>(
			num_kernels2,
			accum_data,
			counter_data,
			bottom_diff,
			num,
			stride,
			beta,
			use_gradient_);
	CUDA_POST_KERNEL_CHECK;

	// If need to refine the diff value, eliminate the diff which is too large
	if (has_h_rate_){
		Dtype ave_diff = 0;
		caffe_gpu_asum(norm_bottom->count(), bottom_diff, &ave_diff);
	    ave_diff /= Dtype(norm_bottom->count());
		const Dtype H = h_rate_ * ave_diff;

		// Step3 iter the bottom_diff
		const int num_kernels3 = norm_bottom->count();

		Backward_step3_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels3), CAFFE_CUDA_NUM_THREADS>>>(
				num_kernels3,
				bottom_diff,
				H);
		CUDA_POST_KERNEL_CHECK;
	}

}



template <typename Dtype>
void PixelwiseLossLayer<Dtype>::Pixelwise_inference_gpu(const vector<Blob<Dtype>*>& bottom){
	const Blob<Dtype>* depth_bottom = bottom[0];
	const Blob<Dtype>* sp_bottom = bottom[2];
	const Blob<Dtype>* norm_bottom = bottom[4];
	const Blob<Dtype>* centroid_bottom = bottom[3];

	const Dtype* dep_data = depth_bottom->cpu_data();
	const Dtype* sp_data = sp_bottom->cpu_data();
	const Dtype* norm_data = norm_bottom->cpu_data();
	const Dtype* centroid_data = centroid_bottom->cpu_data();

	Dtype* pred_data = Pred_.mutable_gpu_data();

	const int stride = superpixel_num_;
	const int num = Pred_.num();
	const int height = Pred_.height();
	const int width = Pred_.width();

	// The kernel number is N * h
	const int num_kernels = num * height;
	Pixelwise_inference_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>(
		 num_kernels,
		 dep_data,
		 sp_data,
		 norm_data,
		 centroid_data,
		 pred_data,
		 num,
		 height,
		 width,
		 stride,
		 f_,
		 z_thd_,
		 use_gradient_,
		 this->phase_ == TRAIN);
	CUDA_POST_KERNEL_CHECK;

}


template void PixelwiseLossLayer<float>::Euclidean_loss_gpu(const Blob<float>* gt, Blob<float>* top);
template void PixelwiseLossLayer<double>::Euclidean_loss_gpu(const Blob<double>* gt, Blob<double>* top);

template void PixelwiseLossLayer<float>::Berhu_loss_gpu(const Blob<float>* gt, Blob<float>* top);
template void PixelwiseLossLayer<double>::Berhu_loss_gpu(const Blob<double>* gt, Blob<double>* top);

template void PixelwiseLossLayer<float>::ScaleInvariant_loss_gpu(const Blob<float>* gt, Blob<float>* top);
template void PixelwiseLossLayer<double>::ScaleInvariant_loss_gpu(const Blob<double>* gt, Blob<double>* top);

template void PixelwiseLossLayer<float>::Pixelwise_inference_gpu(const vector<Blob<float>*>& bottom);
template void PixelwiseLossLayer<double>::Pixelwise_inference_gpu(const vector<Blob<double>*>& bottom);

INSTANTIATE_LAYER_GPU_FUNCS(PixelwiseLossLayer);

}  // namespace caffe
