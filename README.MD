A class for build tfrecord file and build dataloader using tf.data which can be used for both tensorflow and pytorch 
requirements:

	tensorflow==2.0
	pyyaml
	numpy
	tqdm
usage: as is shown in the builder.py, you just need build a list of example using numpy and config the yaml file.

init: operator = RecordOperator(example_list(can be empty for load and build loader), config_file_path, tfrecord_file_path)

build tfrecord: operator.build_tfrecord_file()

build dataloader: operator.build_data_loader(#args#)

dtype:data type.in [float32 and int64]

shape_type: feature shape in ['var', 1, 2], var means the shape variable , 1 and 2 means the shape is fixed and have shape num 1 or 2. 

    examples1 = {
        'feature': np.array([[1, 2, 3], [2, 3, 4]], dtype='float32'),
        'feature_shape': np.array(list(np.array([[1, 2, 3], [2, 3, 4]], dtype='int64').shape), dtype='int64'),
        'target': np.array([1, 2, 3, 4], dtype='int64'),
        'target_shape': np.array(list(np.array([1, 2, 3, 4], dtype='int32').shape), dtype='int64')
    }
    examples2 = {
        'feature': np.array([[1, 2, 3, 4], [2, 3, 4, 5]], dtype='float32'),
        'feature_shape': np.array(list(np.array([[1, 2, 3, 4], [2, 3, 4, 5]], dtype='int64').shape), dtype='int64'),
        'target': np.array([1, 2, 3, 4, 6], dtype='int64'),
        'target_shape': np.array(list(np.array([1, 2, 3, 4, 6], dtype='int32').shape), dtype='int64')
    }
    examples = [examples1, examples2]
    config_file = 'tfrecord_config.yaml'
    record_file = 'record_test.tfrecord'
    operator = RecordOperator(examples, config_file, record_file)
    operator.build_tfrecord_file()
    dataloader = operator.build_data_loader(num_parallel_calls=2, num_epoch=3, batch_size=3)
    for i in dataloader:
        break
