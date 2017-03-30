#include <vector>

#include "caffe/layers/crf_norm_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

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

template <typename Dtype>
__global__ void calc_rd_gpu_kernel(
		const int num_kernels, 
		const Dtype* const dep_data,
		const Dtype* const norm_data, 
		const Dtype* const centroid_data,
		const Dtype* const g_data,
		Dtype* r_data, 
		Dtype* d_data,
		const int num, 
		const int channels, 
		const int height, 
		const int width,
		const int im_height,
		const int im_width,
		const Dtype w1,
		const Dtype w2,
		const Dtype w3,
		const Dtype theta,
		const Dtype focal,
		const bool use_gradient){
	CUDA_KERNEL_LOOP(index, num_kernels){
		const int n = index / height / width;
		const int h = (index / width) % height;
		const int w = index % width;

		if (w >= h){
			return;
		}

		const int sp_plan = height;
		const int d_stride = height * width;

		const int feature1_idx = (n*channels+0)*sp_plan+h;
		const int feature2_idx = (n*channels+0)*sp_plan+w;
		const int top_idx1 = ((n*height)+h)*width+w;
		const int top_idx2 = ((n*height)+w)*width+h;
		const int coord_idx1 = ((n + 0) * sp_plan + h) * 2;
		const int coord_idx2 = ((n + 0) * sp_plan + w) * 2;
		const int d_idx1 = (n * 2 * height + h) * width + w;
		const int d_idx2 = (n * 2 * height + w) * width + h;
		const int dep_idx1 = n*height+h;
		const int dep_idx2 = n*height+w;

		const Dtype* coord_data1 = centroid_data + coord_idx1;
		const Dtype* coord_data2 = centroid_data + coord_idx2;

		Dtype dot = 0;
		Dtype feature1_norm = 1;
		Dtype feature2_norm = 1;

		if (use_gradient){
			// Convert the gradient to surface normal
			const Dtype data1_dx = norm_data[feature1_idx];
			const Dtype data1_dy = norm_data[feature1_idx + sp_plan];
			const Dtype data2_dx = norm_data[feature2_idx];
			const Dtype data2_dy = norm_data[feature2_idx + sp_plan];
			const Dtype data1_z = 1;
			const Dtype data1_x = - data1_z * data1_dx;
			const Dtype data1_y = - data1_z * data1_dy;
			const Dtype data2_z = 1;
			const Dtype data2_x = - data2_z * data2_dx;
			const Dtype data2_y = - data2_z * data2_dy;

			dot = data1_x * data2_x + data1_y * data2_y + data1_z + data2_z;
			feature1_norm = sqrt(data1_x * data1_x + data1_y * data1_y + data1_z + data1_z);
			feature2_norm = sqrt(data2_x * data2_x + data2_y * data2_y + data2_z + data2_z);

		}else{

			// Calc the angle between two feature vectors
			// Calc the L2 normal of the given two vectors
			feature1_norm = sqrt(dot_stride2(channels, norm_data+feature1_idx, sp_plan, norm_data+feature1_idx, sp_plan));
			feature2_norm = sqrt(dot_stride2(channels, norm_data+feature2_idx, sp_plan, norm_data+feature2_idx, sp_plan));

			// Calc the dot between the two feature vector
			dot = dot_stride2(channels, norm_data+feature1_idx, sp_plan, norm_data+feature2_idx, sp_plan);

		}


		// Calc the angle between the vector
		Dtype cos_ang;
		if (feature1_norm == 0 || feature2_norm == 0){
			cos_ang = 0;
		}else{
			cos_ang = min(max(dot / feature1_norm / feature2_norm, Dtype(-1)), Dtype(1));
		}

		// Apply the theta regulation
		if (cos_ang < theta){
			cos_ang = 0;
		}

		// Calc the distance between two points
		Dtype distsq = calc_distsq<Dtype>(coord_data1, coord_data2, im_height, im_width);
		// The larger means the less regulation
		cos_ang = Dtype(1) - cos_ang;

		const Dtype height1 = coord_data1[0];
		const Dtype width1 = coord_data1[1];
		const Dtype height2 = coord_data2[0];
		const Dtype width2 = coord_data2[1];

		const Dtype dh = (height2 - height1) / focal;
		const Dtype dw = (width2 - width1) / focal;

		d_data[d_idx1] = dh;
		d_data[d_idx1+d_stride] = dw;
		d_data[d_idx2] = -dh;
		d_data[d_idx2+d_stride] = -dw;


		// Calc the project depth diff
		const Dtype g1y = g_data[(n * 2 + 0)* height + h];
		const Dtype g1x = g_data[(n * 2 + 1)* height + h];
		
		const Dtype g2y = g_data[(n * 2 + 0)* height + w];
		const Dtype g2x = g_data[(n * 2 + 1)* height + w];

		const Dtype dep1 = dep_data[dep_idx1];
		const Dtype dep2 = dep_data[dep_idx2];

		const Dtype dep1_proj = dep2 - dep2 * (dh * g2y + dw * g2x);
		const Dtype dep2_proj = dep1 + dep1 * (dh * g1y + dw * g1x);

		// The distance between two superpixel (focal normalized)
		const Dtype dist = sqrt(dh * dh + dw * dw);
		const Dtype proj_diff = (fabs(dep1_proj - dep1) + fabs(dep2_proj - dep2)) / dist;

		// Set the R data
		r_data[top_idx1] = -cos_ang * w1 - distsq * w2 - proj_diff * w3;
		r_data[top_idx2] = r_data[top_idx1];
	}
}

template<typename Dtype>
__global__ void calc_a_kmn_gpu_kernel(
		const int num_kernels,
	    Dtype* a_data,
	    const Dtype* const r_data,
		const Dtype* const d_data,
	    const Dtype* const g_data,
		const int num,
		const int height,
		const int width){

	CUDA_KERNEL_LOOP(index, num_kernels){
		const int n = index / height;
		const int h = index % height;
		// The channels is 1

		// The basic index
		const int ar_idx = (n * height + h) * width;
		const int d_idx = (n * 2 * height + h) * width;
		const int g_idx = n * 2 * height;

		const int d_stride = height * width;
		const int g_stride = height;

		// A = A - K
		// K_ij = 0.5 * R_ij * (D_ij * G_i + D_ji * G_j)
		// K_ij = 0.5 * R_ij * (D_ij * G_i - D_ij * G_j)

		// A = A + M + N
		// M_ii = sigma_j (R_ij * D_ij * G_i)
		// N_ii = 0.5 * sigma_j (R_ij * (D_ij * G_i)^2 )

		// Iter the width (superpixel num)

		for (int w = 0; w < width; w++){
			Dtype k = 0.5 * r_data[ar_idx+w] * (d_data[d_idx+w] * g_data[g_idx+h] + d_data[d_idx+w+d_stride] * g_data[g_idx+h+g_stride] - d_data[d_idx+w] * g_data[g_idx+w] - d_data[d_idx+w+d_stride] * g_data[g_idx+w+g_stride]);
			a_data[ar_idx+w] = a_data[ar_idx+w] - k;

			Dtype val = d_data[d_idx+w] * g_data[g_idx+h] + d_data[d_idx+w+d_stride] * g_data[g_idx+h+g_stride];

			a_data[ar_idx+h] += r_data[ar_idx+w] * val + 0.5 * r_data[ar_idx+w] * val * val;
		}
	}
}



template <typename Dtype>
__global__ void calc_g_gpu_kernel(
		const int num_kernels, 
		const Dtype* const bottom_data,
		Dtype* g_data,
		const int num,
		const int height){
	CUDA_KERNEL_LOOP(index, num_kernels){
		const int n = index / height;
		const int h = index % height;

		// The basic index of the normal map
		const int norm_idx = n * 3 * height + h;
		// The basic index of the g
		const int g_idx = n * 2 * height + h;

		const Dtype x = bottom_data[norm_idx];
		const Dtype y = bottom_data[norm_idx + height];
		const Dtype z = bottom_data[norm_idx + 2 * height];

		Dtype dx = - x / z;
		Dtype dy = - y / z;

		if (fabs(z) < 0.1){
			dx = 0;
			dy = 0;
		}

		g_data[g_idx] = dy;
		g_data[g_idx + height] = dx;
	}
}


template <typename Dtype>
__global__ void calc_a_gpu_kernel(const int n, Dtype* a_data,
		const int num, const int channels, const int height, const int width,
		const Dtype* r_data){
	CUDA_KERNEL_LOOP(index, n) {
		const int n_idx = index / height / channels;
		const int c = (index / height) % channels;
		const int h = index % height;

		const int idx = ((n_idx*channels+c)*height+h)*width;
		// Calc the sum of the row in r_data
		Dtype sum = 0;
		for (int i = 0; i < width; i++){
			sum += r_data[idx+i];
		}
		a_data[idx+h] = sum + 1;
	}
}


template <typename Dtype>
void CrfNormLossLayer<Dtype>::Calc_A_gpu(void){
	// A = I + D -R
	// Set A to 0
	caffe_gpu_set(A_.count(), Dtype(0), A_.mutable_gpu_data());
	const int num = A_.num();
	const int channels = A_.channels();
	const int height = A_.height();
	const int width = A_.width();
	const Dtype* r_data = R_.gpu_data();
	Dtype* a_data = A_.mutable_gpu_data();
	// A = I + D
	// kernel num: n*c*h
	int num_kernels = num * channels * height;
	calc_a_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                             CAFFE_CUDA_NUM_THREADS>>>(
      num_kernels, a_data, num, channels, height, width, r_data);
	CUDA_POST_KERNEL_CHECK;
	// A = A - R
	caffe_gpu_axpy(A_.count(), Dtype(-1), r_data, a_data);

	// If disable the surface normal guidance, return directly
	if (disable_normal_guidance_) return;

	// A = A - K
	// K_ij = 0.5 * R_ij * (D_ij * G_i + D_ji * G_j)
	// K_ij = 0.5 * R_ij * (D_ij * G_i - D_ij * G_j)

	// A = A + M + N
	// M_ii = sigma_j (R_ij * D_ij * G_i)
	// N_ii = 0.5 * sigma_j (R_ij * (D_ij * G_i)^2 )

	const Dtype* d_data = D_.gpu_data();
	const Dtype* g_data = G_.gpu_data();
	// The kernel number is n * height
	num_kernels = num * height;
	calc_a_kmn_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
		CAFFE_CUDA_NUM_THREADS>>>(num_kernels, a_data, r_data,
				d_data, g_data, num, height, width);
	CUDA_POST_KERNEL_CHECK;

	// If in scale invariant mode
	// A = A - Q
	if (unary_mode_ == ScaleInvariant){
		const Dtype* q_data = Q_.gpu_data();
		caffe_gpu_sub(A_.count(), a_data, q_data, a_data);
	}
}


/* Formulate the R matrix, which can be calculated from
 * bottom[2] normal prediction
 * bottom[3] centroid coordination
 * bottom[4] superpixel appearance [TODO]
 *
 * The R is combination of three parts:
 * 1. The cosine distance of surface normal, which is 1 - cos(angle)
 * 2. The normalized distance between two superpixel
 * 3. The cosine distance between appearance vector [optional]
 *
 * The overall R is:
 * exp(-w1*M1 - w2*M2 - w3*M3)
 *
 * Formulate the D matrix
 * Which is [n, 2, superpixel_num_, superpixel_num_]
 */
template <typename Dtype>
void CrfNormLossLayer<Dtype>::Calc_RD_gpu(const vector<Blob<Dtype>*>& bottom){
	const Blob<Dtype>* norm_bottom = bottom[2];
	const Blob<Dtype>* centroid_bottom = bottom[3];
	const Blob<Dtype>* dep_bottom = bottom[0];

	const Dtype* norm_data = norm_bottom->gpu_data();
	const Dtype* centroid_data = centroid_bottom->gpu_data();
	const Dtype* g_data = G_.gpu_data();
	const Dtype* dep_data = dep_bottom->gpu_data();

	Dtype* r_data = R_.mutable_gpu_data();
	Dtype* d_data = D_.mutable_gpu_data();

	const int num = R_.num();
	const int channels = norm_bottom->channels();
	const int height = R_.height();
	const int width = R_.width();

	CHECK_EQ(height, width);

	// Clear the R
	caffe_gpu_set(R_.count(), Dtype(0), r_data);
	// Clear D
	caffe_gpu_set(D_.count(), Dtype(0), d_data);

	// The kernel number is the num * height * width
	// TODO: Since the top is a symmtic matrix, so half of the calculation is not
	// necessary. There might be a better method to assign the threads to the pixels
	const int num_kernels = num * height * width;
	calc_rd_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>(
			num_kernels, dep_data, norm_data, centroid_data, g_data, r_data, d_data, num, channels,
		   	height, width, height_, width_, w1_, w2_, w3_, theta_, f_, use_gradient_);

	CUDA_POST_KERNEL_CHECK;

	// exp
	caffe_gpu_exp(R_.count(), r_data, r_data);
	// alpha the weight
	caffe_gpu_scal(R_.count(), Dtype(alpha_), r_data);
}


/*
 * Calc the G matrix according to the normal map
 * The bottom shape is [n, 3, superpixel_num_, 1]
 * The shape of G is [n, 2, superpixel_num_, 1]
 */
template <typename Dtype>
void CrfNormLossLayer<Dtype>::Calc_G_gpu(const Blob<Dtype>* bottom){
	Dtype* g_data = G_.mutable_gpu_data();
	const Dtype* bottom_data = bottom->gpu_data();

	if (use_gradient_){
		CHECK_EQ(bottom->count(), G_.count());
		caffe_copy(G_.count(), bottom_data, g_data);
		return;
	}
	// Set the G to zeros
	caffe_gpu_set(G_.count(), Dtype(0), g_data);

	const int num = G_.num();
	const int height = G_.height();

	CHECK_EQ(height, superpixel_num_);

	// The kernel number is num * height
	const int num_kernels = num * height;
	calc_g_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>(
			num_kernels, bottom_data, g_data, num, height);

	CUDA_POST_KERNEL_CHECK;
}



template <typename Dtype>
void CrfNormLossLayer<Dtype>::Euclidean_loss_gpu(const Blob<Dtype>* gt, Blob<Dtype>* top){
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
void CrfNormLossLayer<Dtype>::ScaleInvariant_loss_gpu(const Blob<Dtype>* gt, Blob<Dtype>* top){
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
	  CUDA_POST_KERNEL_CHECK;
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
void CrfNormLossLayer<Dtype>::Berhu_loss_gpu(const Blob<Dtype>* gt, Blob<Dtype>* top){
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
void CrfNormLossLayer<Dtype>::Inference_gpu(const Blob<Dtype>* Z){
	const int dim = A_inv_.height();
	const Dtype* a_data = A_inv_.gpu_data();
	const Dtype* z_data = Z->gpu_data();
	Dtype* pred_data = Pred_.mutable_gpu_data();

	for (int n = 0; n < A_inv_.num(); n++){
		const Dtype* a_data_n = a_data + A_inv_.offset(n);
		for (int c = 0; c < Z->channels(); c++){
			const Dtype* z_data_nc = z_data + Z->offset(n, c);
			Dtype* pred_data_nc = pred_data + Pred_.offset(n, c);
			caffe_gpu_gemv(CblasNoTrans, dim, dim, Dtype(1), a_data_n, z_data_nc, Dtype(0), pred_data_nc);
		}
	}
}


template <typename Dtype>
void CrfNormLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top){

  // Generate the G
  // NOTE: The G must be calc before the R
  Calc_G_gpu(bottom[2]);
  // Generate the R and D
  Calc_RD_gpu(bottom);
  // Init the QP for scale invariant mode
  Init_QP_gpu();
  // Calc the A matrix
  Calc_A_gpu();
  // Calc the A_inv matrix
  // ---- DEBUG -----
  // test the speed
  // time_t start, end;
  //runstart = clock();
  Calc_A_inv_gpu();
  //end = clock();
  // LOG(INFO)<<"TIME: "<<(Dtype)(end-start)/CLOCKS_PER_SEC;
// ---- DEBUG -----
  // ----DEBUG ----
  /*
  Blob<Dtype> tmp;
  int dim = A_.height();
  tmp.Reshape(1,1,A_.height(), A_.width());
  for (int n = 0; n < A_.num(); n++){
	  caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, dim, dim, dim, Dtype(1), A_.cpu_data()+A_.offset(n), A_inv_.cpu_data()+A_inv_.offset(n), Dtype(0), tmp.mutable_cpu_data());

	  LOG(INFO)<<"Det(S): "<<caffe_cpu_det(A_.height(), tmp.cpu_data());
  }
  */

  // Inference the crf
  switch (unary_mode_){
	  case ScaleInvariant:
		  Inference_scaleinvariant_gpu(bottom[0]);
		  break;
	  default:
		  Inference_gpu(bottom[0]);
		  break;
  }
  // Copy the result if needed
  if (top.size() == 2){
	  caffe_copy(Pred_.count(), Pred_.gpu_data(), top[1]->mutable_gpu_data());
  }
  // Calc the loss according to the Pred_ and the bottom[0]
  switch (unary_mode_){
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
		  LOG(FATAL)<<"Unknow unary_mode_ in CrfNormLossLayer";
		  break;
  }

}


template <typename Dtype>
void CrfNormLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	// The BP will performed on the bottom[0] and bottom[2]
	const Dtype loss_weight = top[0]->cpu_diff()[0];
	Dtype beta;
	if (normalize_){
		beta = loss_weight / bottom[0]->count();
	}else{
		beta = loss_weight / bottom[0]->num();
	}

	// BP for bottom[0]
	if (unary_mode_ == ScaleInvariant){
		caffe_gpu_axpby(
				bottom[0]->count(),
				beta,
				Pred_.cpu_diff(),
				Dtype(0),
				buf_.mutable_gpu_data());

		// In scale invariant mode, the BP should be P*A_inv_*P*Z - P*Y
		// diff = A_inv_*P*Z - Y, so the BP should be
		// P * diff
		const Dtype* p_data = P_.gpu_data();
		const Dtype* buf_data = buf_.gpu_data();
		Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

		for (int n = 0; n < bottom[0]->num(); n++){
			const Dtype* p_data_n = p_data + P_.offset(n);
			for (int c = 0; c < bottom[0]->channels(); c++){
				const Dtype* buf_data_nc = buf_data + buf_.offset(n, c);
				Dtype* bottom_diff_nc = bottom_diff + bottom[0]->offset(n, c);
				caffe_gpu_gemv(CblasNoTrans, P_.height(), P_.height(), Dtype(1), p_data_n, buf_data_nc, Dtype(0), bottom_diff_nc);
			}
		}
	}else{
		// Other modes
		caffe_gpu_axpby(
				bottom[0]->count(),
				beta,
				Pred_.gpu_diff(),
				Dtype(0),
				bottom[0]->mutable_gpu_diff());
	}

}


template <typename Dtype>
void CrfNormLossLayer<Dtype>::Calc_A_inv_gpu(void){
	const int num = A_.num();
	const Dtype* a_data = A_.gpu_data();
	Dtype* a_inv_data = A_inv_.mutable_gpu_data();
	const int height = A_.height();

	caffe_gpu_inv(height, num, a_data, a_inv_data);
}

template <typename Dtype>
void CrfNormLossLayer<Dtype>::Init_QP_gpu(void){
	if (unary_mode_ != ScaleInvariant) return;

	const Dtype val = delta_ / superpixel_num_;
	// Set the Q matrix
	Dtype* q_data = Q_.mutable_gpu_data();
	caffe_gpu_set(Q_.count(), val, q_data);

	// Set the P matrix
	// P = I - Q
	Dtype* p_data = P_.mutable_gpu_data();
	caffe_gpu_set(P_.count(), Dtype(0), p_data);
	caffe_gpu_sub(P_.count(), p_data, q_data, p_data);
	Dtype* p_data_cpu = P_.mutable_cpu_data();
	for (int n = 0; n < P_.num(); n++){
		for (int i = 0; i < P_.height(); i++){
			p_data_cpu[P_.offset(n, 0, i, i)] += Dtype(1);
		}
	}
}

template <typename Dtype>
void CrfNormLossLayer<Dtype>::Inference_scaleinvariant_gpu(const Blob<Dtype>* Z){
	// pred = A_inv_ * P * Z
	const int dim = A_inv_.height();
	const Dtype* a_data = A_inv_.gpu_data();
	const Dtype* z_data = Z->gpu_data();
	const Dtype* p_data = P_.gpu_data();

	// Creat a buffer to make sure the blas safe
	Dtype* buf_data = buf_.mutable_gpu_data();

	Dtype* pred_data = Pred_.mutable_gpu_data();

	for (int n = 0; n < A_inv_.num(); n++){
		const Dtype* a_data_n = a_data + A_inv_.offset(n);
		const Dtype* p_data_n = p_data + P_.offset(n);
		for (int c = 0; c < Z->channels(); c++){
			const Dtype* z_data_nc = z_data + Z->offset(n, c);
			Dtype* buf_data_nc = buf_data + buf_.offset(n, c);
			Dtype* pred_data_nc = pred_data + Pred_.offset(n, c);
			// buf = P * Z
			caffe_gpu_gemv(CblasNoTrans, dim, dim, Dtype(1), p_data_n, z_data_nc, Dtype(0), buf_data_nc);
			// pred = A_inv_ * pred
			caffe_gpu_gemv(CblasNoTrans, dim, dim, Dtype(1), a_data_n, buf_data_nc, Dtype(0), pred_data_nc);
		}
	}
}



template void CrfNormLossLayer<float>::Calc_A_gpu(void);
template void CrfNormLossLayer<double>::Calc_A_gpu(void);

template void CrfNormLossLayer<float>::Calc_RD_gpu(const vector<Blob<float>*>& bottom);
template void CrfNormLossLayer<double>::Calc_RD_gpu(const vector<Blob<double>*>& bottom);

template void CrfNormLossLayer<float>::Calc_G_gpu(const Blob<float>* bottom);
template void CrfNormLossLayer<double>::Calc_G_gpu(const Blob<double>* bottom);

template void CrfNormLossLayer<float>::Euclidean_loss_gpu(const Blob<float>* gt, Blob<float>* top);
template void CrfNormLossLayer<double>::Euclidean_loss_gpu(const Blob<double>* gt, Blob<double>* top);

template void CrfNormLossLayer<float>::Berhu_loss_gpu(const Blob<float>* gt, Blob<float>* top);
template void CrfNormLossLayer<double>::Berhu_loss_gpu(const Blob<double>* gt, Blob<double>* top);

template void CrfNormLossLayer<float>::ScaleInvariant_loss_gpu(const Blob<float>* gt, Blob<float>* top);
template void CrfNormLossLayer<double>::ScaleInvariant_loss_gpu(const Blob<double>* gt, Blob<double>* top);

template void CrfNormLossLayer<float>::Inference_gpu(const Blob<float>* Z);
template void CrfNormLossLayer<double>::Inference_gpu(const Blob<double>* Z);

template void CrfNormLossLayer<float>::Calc_A_inv_gpu(void);
template void CrfNormLossLayer<double>::Calc_A_inv_gpu(void);

template void CrfNormLossLayer<float>::Init_QP_gpu(void);
template void CrfNormLossLayer<double>::Init_QP_gpu(void);

template void CrfNormLossLayer<float>::Inference_scaleinvariant_gpu(const Blob<float>* Z);
template void CrfNormLossLayer<double>::Inference_scaleinvariant_gpu(const Blob<double>* Z);

INSTANTIATE_LAYER_GPU_FUNCS(CrfNormLossLayer);

}  // namespace caffe
