from dnn import *
from input import *
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import parsing_ops

placeholder_dict = {}
samples = {}
for key in featuregroup_count_dict:
	placeholder_dict[key + "_indice"] = tf.placeholder(tf.int64, name=key + "_indice")
	placeholder_dict[key + "_value"] = tf.placeholder(tf.int64, name=key + "_value")
	placeholder_dict[key + "_shape"] = tf.placeholder(tf.int64, name=key + "_shape")

	samples[key] = tf.SparseTensor(placeholder_dict[key + "_indice"],
	                               placeholder_dict[key + "_value"],
	                               placeholder_dict[key + "_shape"])

pred = tf.nn.sigmoid(inference(x_batch=samples, ps_num=3))

# define model exporter
saver = tf.train.Saver(write_version=tf.train.SaverDef.V2, max_to_keep=100)

# save service model
graph_options = tf.GraphOptions(enable_bfloat16_sendrecv=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95, allow_growth=True)
config = tf.ConfigProto(graph_options=graph_options, gpu_options=gpu_options, log_device_placement=False,
                        allow_soft_placement=True)

with tf.Session(config=config) as sess:
	model_path = FLAGS.model_path
	serving_path = FLAGS.serving_path + "/date=" + FLAGS.today
	# save graph
	latest_checkpoint_file_path = tf.train.latest_checkpoint(model_path)
	print(latest_checkpoint_file_path)
	saver.restore(sess, latest_checkpoint_file_path)
	print("DNN recall model restored")

	# build graph
	builder = tf.saved_model.builder.SavedModelBuilder(serving_path)

	# build the signature_def_map
	for key in placeholder_dict:
		placeholder_dict[key] = tf.saved_model.utils.build_tensor_info(placeholder_dict[key])

	pred_tensor_info = tf.saved_model.utils.build_tensor_info(pred)

	predict_signature = (
		tf.saved_model.signature_def_utils.build_signature_def(
			inputs=placeholder_dict,
			outputs={'outputs': pred_tensor_info},
			method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

	legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')

	print(predict_signature)
	builder.add_meta_graph_and_variables(sess=sess, tags=[tf.saved_model.tag_constants.SERVING],
	                                     signature_def_map={'predict': predict_signature},
	                                     legacy_init_op=legacy_init_op,
	                                     saver=saver)

	builder.save()
	print('Done exporting!')

