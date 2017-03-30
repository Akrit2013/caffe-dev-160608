YanHan's Modification:

1. TransformParamEx: can crop the image into rectangle rather than the square only
2. SyncTransformationParameter: Can sync transform the data image and label image, it is a very useful layer in the image2image models.
   SyncTransformLayer: Transform the input blobs with the same random parameters
3. FocalTransformationParameter: Used in focal length estimation algorithm, can crop the data and alter the facal label.
   FocalTransformLayer: Crop the image and transform the focal length accordingly.
4. Unpooling layer: It can reverse the pooling to enlarge the feature map
5. euclideanfcn_loss_layer: It can ignore the bad pixel caused by NaN in the label image
6. ResizeLayer: This layer use opencv function can resize the feature map (include the data and label image) online.
7. ConvolutionExLayer: This layer support the custom pad value instead of the default 0
8. BaseConvolutionExLayer: Provide base layer for the ConvolutionExLayer
9. im2col_ex: Used by BaseConvolutionExLayer to provide modified pad value
10. Conv1x1Layer: The layer only do 1x1 conv and have some extra regulation on the conv parameters
11. BaseConv1x1Layer: Provide the base for the Conv1x1Layer.
12. UnpackBitsLayer: Split the packed pairwise data into multi channel data
13. CrfLossLayer: The CRF loss layer
14. ConvDropoutLayer: Perform the dropout on the feature map rather than the pixel
15. Crf2LossLayer: The second version of the CrfLossLayer, it embed the Conv1x1Layer param into the loss layer, that means the bottom[2] can directly connected to the output
				   of the UnpackBitsLayer
16. SyncTransFastLayer: A simple and faster version of the SyncTransformLayer.
17. L2NormLayer: Can Perform L2 normalize between the channels of certain pixel position
18. PairwiseMapLayer: Generate the R matrix in CRF loss layer according to the pairwise enery from the superpixel pooling layer
19. Crf3LossLayer: The CRF loss designed for multi channels prediction (e.g. normal map)
20. CrfUnaryLossLayer: modified from the Crf3LossLayer, this loss layer drop the pairwise parameter learning, only learn the unary parameter. It take the R matrix directly
	as the input instead of the pairwise features. NOTE: The R matrix can be calculated by the PairwiseMapLayer.
21. BNLayer: This is another batch normalization layer introducted in PSPnet, contains four parameters as 'slope,bias,mean,variance' while 'batch_norm_layer' contains two parameters as 'mean,variance'.
22. ScaleInvariantLossLayer: A loss layer introducted by Eigen NIPS2014, but this layer DO NOT use the log space as the Eigen did.
23. ScaleInvariantLogLossLayer: The loss layer introducted by Eigne NIPS2014
24. ScaleInvariantBerhuLossLayer: The ScaleInvariantLossLayer combined with Berhu loss layer
25. ScaleInvariantLogBerhuLossLayer: The ScaleInvariantLogLossLayer combined with Berhu loss layer
26. SuperpixelCentroidLayer: This layer locate each superpixel's centroid coordinates, which can be used in other layers
27. CrfNormLossLayer: The normal guided CRF loss layer.
28. DistCorrMatLayer: Generate the superpixel distance correlation matrix according to the central positial of each superpixel
29. NormCorrMatLayer: Generate the superpixel normal correlation matrix
30. PixelwiseLossLayer: Change the superpixel wise prediction into the pixelwise prediction, and BP the surface normal diff
31. GradToNormLayer: This layer convert the gradient input bottom into the surface normal, the input bottom shape is [n, 3, x, x], and the output shape is [n, 2, x, x], it contains both forward and backward functions.

# Caffe

[![Build Status](https://travis-ci.org/BVLC/caffe.svg?branch=master)](https://travis-ci.org/BVLC/caffe)
[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

Caffe is a deep learning framework made with expression, speed, and modularity in mind.
It is developed by the Berkeley Vision and Learning Center ([BVLC](http://bvlc.eecs.berkeley.edu)) and community contributors.

Check out the [project site](http://caffe.berkeleyvision.org) for all the details like

- [DIY Deep Learning for Vision with Caffe](https://docs.google.com/presentation/d/1UeKXVgRvvxg9OUdh_UiC5G71UMscNPlvArsWER41PsU/edit#slide=id.p)
- [Tutorial Documentation](http://caffe.berkeleyvision.org/tutorial/)
- [BVLC reference models](http://caffe.berkeleyvision.org/model_zoo.html) and the [community model zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo)
- [Installation instructions](http://caffe.berkeleyvision.org/installation.html)

and step-by-step examples.

[![Join the chat at https://gitter.im/BVLC/caffe](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/BVLC/caffe?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Please join the [caffe-users group](https://groups.google.com/forum/#!forum/caffe-users) or [gitter chat](https://gitter.im/BVLC/caffe) to ask questions and talk about methods and models.
Framework development discussions and thorough bug reports are collected on [Issues](https://github.com/BVLC/caffe/issues).

Happy brewing!

## License and Citation

Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).
The BVLC reference models are released for unrestricted use.

Please cite Caffe in your publications if it helps your research:

    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year = {2014}
    }
