import h5py
from collections import namedtuple

class HDF5Generator:
    """
    Generator for yielding HDF5 data entries.
    """
    
    def __init__(self, key=None, fields=None):
        """
        Args:
            key (str): HDF5 key
            fields (namedtuple): Named tuple of columns elements names to extract from HDF5 file
        """
        
        if key is not None:
            if not isinstance(key, str):
                raise ValueError('key <{}> must be str'.format(key))
            self.key = key
        if fields is not None:
            if not isinstance(fields, type):
                raise ValueError('fields<{}> must be namedtuple'.format(fields))
            self.fields = fields
        
    def __call__(self, file):        
        print('Processing file <{}>...\n'.format(file))
            
        with h5py.File(file, 'r') as hf:
            for entry in hf[self.key]:
                # Generates tuple field in dict comprehension and yields fields data
                yield self.fields(**{key: entry[key] for key in self.fields._fields})


class PreProcess(object):
    """
    Class for defining and preprocessing tf.Dataset stream.
    """
    
    # Data format parameters; modify as needed to fit data pipeline
    key = 'Waveforms'
    props = ('feature', 'label')
    fields = namedtuple('Raw', 'samples class_num')
    output = namedtuple('Processed', props)
    generator_datatypes = (tf.float64, tf.int32)
    generator_shapes = (tf.TensorShape([None]), tf.TensorShape([]))
    
    def __init__(self, cycle_length=2, read_option='hdf5', start_index=0, stop_index=90000,
                 num_parallel_calls=2, buffer_size=100, repeat=1, batch=1, prefetch=1, **kwargs):
        
        """
        Args:
            cycle_length (int): Number of input files to process.
            read_option (str): Type of input file.
            start_index (int): Waveform starting point.
            stop_index (int): Waveform ending point.
            num_parallel_calls (int): Number of processes to run in parallel.
            buffer_size (int): Buffer size.
            repeat (int): Number of times dataset is repeated.
            batch (int): Batch size.
            prefetch (int): Number of datapoints to prefetch to memory.
            kwargs: Other arguments.
        """
        
        self.cycle_length = cycle_length
        self.read_option = read_option
        self.start_index = start_index
        self.stop_index = stop_index
        self.num_parallel_calls = num_parallel_calls
        self.buffer_size = buffer_size
        self.repeat = repeat
        self.batch = batch
        self.prefetch = prefetch
    
    def _preprocess(self, *args):
        """
        Applies preprocessing steps on input data.
        """
        
        data_stream = {key: args[self.props.index(key)] for key in self.props}
        
        # Use only segment of waveform (default full waveform)
        data_stream['feature'] = data_stream['feature'][self.start_index:self.stop_index]
            
        length = self.stop_index - self.start_index
        data_stream['feature'] = tf.reshape(tensor=data_stream['feature'], shape=[length,1], name='feature')
        
        # Normalize vector by largest value (gives values in range [0,1])
        data_stream['feature'] /= tf.reduce_max(data_stream['feature'], axis=0)

        data_stream['feature'] = tf.math.abs(data_stream['feature'])
        data_stream['feature'] = tf.where(tf.is_nan(data_stream['feature']), tf.zeros_like(data_stream['feature']), data_stream['feature'])

        data_stream['label'] = tf.one_hot(data_stream['label'], 2, name='label')
        return self.output(**{key: data_stream[key] for key in self.props})
    
    def _generate(self, files, read_option='hdf5'):
        """
        Creates dataset from input files.
        """
        if self.read_option == 'hdf5':   
            dataset = files.interleave(lambda filename: tf.data.Dataset.from_generator(HDF5Generator(key=self.key, fields=self.fields), 
                                                                                       self.generator_datatypes, self.generator_shapes, 
                                                                                       args=(filename,)), cycle_length=self.cycle_length)
        elif self.read_option =='tfrecord':
            raise NotImplementedError
        else:
            raise NotImplementedError
        return dataset
    
    def prep(self, files):
        """
        Main method for generating and delivering preprocessed dataset.
        
        Args:
            files (list): List of data files to use.
        Returns:
            tf.Dataset: Preprocessed tf.Dataset object.
        """
        
        files = tf.data.Dataset.from_tensor_slices(files)
        dataset = self._generate(files)
        dataset = dataset.map(lambda *data: self._preprocess(*data), num_parallel_calls=self.num_parallel_calls)
        dataset = dataset.shuffle(buffer_size=self.buffer_size)
        dataset = dataset.repeat(self.repeat)
        dataset = dataset.batch(self.batch)
        dataset = dataset.prefetch(self.prefetch)
        return dataset