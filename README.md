# Tensorflow-SegNet
Implement slightly different [SegNet](http://arxiv.org/abs/1511.00561) in tensorflow,
successfully trained segnet-basic in CamVid dataset

for model detail, please go to https://github.com/alexgkendall/caffe-segnet
# Remark
Due to indice unravel still unavailable in tensorflow, the original upsampling
method is temporarily replaced by deconv( or conv-transpose) layer.
this model is still under construction, mainly use for personal research
