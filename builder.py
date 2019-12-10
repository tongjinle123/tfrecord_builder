import tensorflow as tf
import yaml
import numpy as np
from tqdm import tqdm


def load_yaml(file):
    config = yaml.safe_load(open(file))
    return config


class RecordOperator:
    """
    dtype in int64 or float32
    """

    def __init__(self, examples, config_file, record_file):
        self.config = load_yaml(config_file)
        self.examples = examples
        self.record_file = record_file

    def encode_example(self, example):
        encoded_example = {}
        for feature in self.config.keys():
            print(feature)
            assert feature in example
            assert isinstance(example[feature], np.ndarray)
            assert example[feature].dtype == self.config[feature]['dtype']
            if not self.config[feature]['shape_type'] == 'var':
                assert self.config[feature]['shape_type'] == len(example[feature])
            encoded_example[feature] = self.encode_feature(
                example[feature], self.config[feature]['dtype'], self.config[feature]['shape_type']
            )
        serialized_example = tf.train.Example(features=tf.train.Features(feature=encoded_example)).SerializeToString()
        return serialized_example

    def encode_feature(self, feature, dtype, shape_type):
        if shape_type == 'var' and dtype in ['float32', 'int64']:
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[feature.tostring()]))
        elif shape_type in [1,2] and dtype == 'int64':
            return tf.train.Feature(int64_list=tf.train.Int64List(value=feature.tolist()))

    def build_tfrecord_file(self):
        with tf.io.TFRecordWriter(self.record_file) as tfwriter:
            for example in tqdm(self.examples, desc=f'writing examples to {self.record_file}'):
                serialized = self.encode_example(example)
                tfwriter.write(serialized)
        print(f'\ntfrecord file {self.record_file} built. \n')

    def parse_func(self, dtype, shape_type):
        if shape_type == 'var':
            return tf.io.FixedLenFeature(shape=[], dtype=tf.string)
        elif shape_type != 'var':

            if dtype == 'int64':
                _dt = tf.int64
            elif dtype == 'float32':
                _dt = tf.float32
            else:
                raise ValueError
            return tf.io.FixedLenFeature(shape=[shape_type], dtype=_dt)

    def decode_func(self, feature, dtype, shape_type):
        if dtype == 'float32':
            _dt = tf.float32
        elif dtype == 'int64':
            _dt = tf.int64
        else:
            raise ValueError

        if shape_type != 'var':
            return tf.cast(feature, dtype=_dt)
        else:
            return tf.io.decode_raw(feature, out_type=_dt)

    def build_parser(self):
        def parse_func(sample):
            parse_dict = {}
            for feature_name, feature_sub_config in self.config.items():
                parse_dict[feature_name] = self.parse_func(feature_sub_config['dtype'], feature_sub_config['shape_type'])
            parsed_sample = tf.io.parse_single_example(sample, parse_dict)

            result = {}
            for feature_name, feature_sub_config in self.config.items():
                result[feature_name] = self.decode_func(parsed_sample[feature_name], feature_sub_config['dtype'], feature_sub_config['shape_type'])

            for feature_name, feature_sub_config in self.config.items():
                if feature_sub_config['shape_type'] == 'var':
                    shape_for_this = feature_name + '_shape'
                    assert shape_for_this in self.config
                    result[feature_name] = tf.reshape(result[feature_name], result[shape_for_this])
            return result
        return parse_func

    def build_data_loader(self, num_parallel_calls, num_epoch, batch_size):
        parser = self.build_parser()
        data_loader = tf.data.TFRecordDataset(self.record_file)
        data_loader = data_loader.map(parser, num_parallel_calls=num_parallel_calls)
        data_loader = data_loader.repeat(num_epoch)
        padded_shapes = {}
        for feature_name, feature_sub_config in self.config.items():
            if feature_sub_config['shape_type'] == 'var':
                shape_for_this = feature_name + '_shape'
                assert shape_for_this in self.config
                if self.config[shape_for_this]['shape_type'] != 'var':
                    padded_shapes[feature_name] = [None for i in range(self.config[shape_for_this]['shape_type'])]
            else:
                assert feature_name[:-6] in self.config
                padded_shapes[feature_name] = [None]
        print(padded_shapes)
        #padded_shapes = {'feature': [None, None], 'feature_shape': [None], 'target': [None], 'target_shape': [None]}
        data_loader = data_loader.padded_batch(batch_size, padded_shapes=padded_shapes)
        return data_loader


if __name__ == '__main__':

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
    operator.encode_example(examples[0])
    operator.build_tfrecord_file()
    dataloader = operator.build_data_loader(num_parallel_calls=2, num_epoch=3, batch_size=3)
    for i in dataloader:
        break

