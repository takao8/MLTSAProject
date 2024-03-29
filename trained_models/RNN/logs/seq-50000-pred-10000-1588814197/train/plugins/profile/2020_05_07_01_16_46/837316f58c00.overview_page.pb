�	��j��Q@��j��Q@!��j��Q@	���@��?���@��?!���@��?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6��j��Q@�an�@1C���I@AUK:��l�?I����;,@Y�4~�$�?*	9��v��f@2F
Iterator::Model�Vc	k�?!͋�Lz�D@)a���U�?1�-���9@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat �={.S�?!.�?#�4@)<.�ED1�?1�]�=C2@:Preprocessing2U
Iterator::Model::ParallelMapV2��(#. �?!��o�S�.@)��(#. �?1��o�S�.@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate��i��Ѡ?!$�� ��1@)�wg���?1�e��p-@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�9z�ަ�?!3t��_M@)u�i�ȕ?1��+#'@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapJ+��?!���R�:@)8�0C㉐?1 �=2q�!@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice��p���w?!�p��*	@)��p���w?1�p��*	@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor���Qq?!
���-@)���Qq?1
���-@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 6.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�19.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�an�@�an�@!�an�@      ��!       "	C���I@C���I@!C���I@*      ��!       2	UK:��l�?UK:��l�?!UK:��l�?:	����;,@����;,@!����;,@B      ��!       J	�4~�$�?�4~�$�?!�4~�$�?R      ��!       Z	�4~�$�?�4~�$�?!�4~�$�?JGPU�"N
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop�)���?!�)���?"&
CudnnRNNCudnnRNN�@�i�?!s+W�w�?"*
transpose_9	Transpose�Vꆘn?!���:�?"C
$gradients/transpose_9_grad/transpose	Transpose���Im?!_��7�W�?"(

concat_1_0ConcatV2Ӛ���f?!��~�sn�?"*
transpose_0	Transpose�e�&�re?!`�����?"A
"gradients/transpose_grad/transpose	Transpose�.Ċ�c?!x�i2���?"(
gradients/AddNAddN�e��cY?!+ft	b��?";
gradients/split_1_grad/concatConcatV2�w����W?!�X�`��?"9
gradients/split_grad/concatConcatV2Ťt;W?!p�����?2blackQ      Y@Y����?a�����X@"�	
both�Your program is POTENTIALLY input-bound because 6.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�19.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQ2"GPU(: 