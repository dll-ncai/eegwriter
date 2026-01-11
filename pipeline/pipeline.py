''' Defines all the preprocessing functions and applies them
    to the data in the pipeline
'''


import mne
import numpy as np
from .channels import NMT_CHANNELS, NMT_PAIRS, LOCAL_CHANNELS

class Preprocess:
    def func(self, data):
        # Do something to the data
        # This is to be overloaded always
        return data
    def apply(self, data):
        ''' Applies the preprocessing pipeline to the data
            INPUT:
                data - EEG - data to be preprocessed
            OUTPUT:
                data - EEG - preprocessed data
        '''
        return self.func(data)
    def get_id(self):
        ''' Returns the ID of the preprocessing function
        '''
        return self.__class__.__name__

class ReduceChannels(Preprocess):
    ''' Reducing the number of channels to the 21 channels in use
        Takes in raw data in mne format
        Returns raw data in mne format with 21 channels only
    '''
    def __init__(self, channels):
        self.channels = channels
    def func(self, data):
        return data.pick(self.channels).reorder_channels(self.channels)

class ClipAbsData(Preprocess):
    ''' Responsible for Clipping the data inside a fixed voltage range
        Inputs: raw EEG data in MNE format
        Outputs: raw EEG data clipped between -absclipx10^-6 and absclipx10^-6
    '''
    def __init__(self, absclip):
        self.absclip = absclip
    def func(self, data):
        return data.apply_function(lambda data: np.clip(data, -0.000001*self.absclip, 0.000001*self.absclip))
    def get_id(self):
        return f'{self.__class__.__name__}_{self.absclip}'

class ResampleData(Preprocess):
    ''' Responsible for resampling the data to 100 Hz
        Inputs: raw EEG data in MNE format
        Outputs: raw EEG data resampled to 100 Hz
    '''
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
    def func(self, data):
        sfreq = data.info['sfreq']
        if (sfreq == self.sample_rate):
            return data
        return data.resample(self.sample_rate)
    def get_id(self):
        return f'{self.__class__.__name__}_{self.sample_rate}'

class PaddedCropData(CropData):
    ''' Responsible for cropping the data to the specified time range.
        If duration < tmax, appends data to the end
        Inputs: raw EEG data in MNE format
        Outputs: raw EEG data cropped to the specified time range
    '''
    def __init__(self, tmin, tmax):
        self.tmin = tmin
        self.tmax = tmax
        self.time_span = tmax - tmin
    def func(self, data):
        data.crop(tmin=self.tmin)
        if data.n_times / data.info["sfreq"] >= self.tmax:
            return data.crop(tmin=0, tmax=self.tmax - self.tmin, include_tmax=False)
        else:
            while data.n_times / data.info["sfreq"] < self.tmax:
                data_only, _ = data[:]
                reversed = np.flip(data_only, axis = 1)
                info = data.info
                data_only = np.concatenate([data_only, data_only], axis=1)
                data = mne.io.RawArray(data_only, info)
            return data.crop(tmin=0, tmax=self.tmax - self.tmin, include_tmax=False)
    def get_id(self):
        return f'{self.__class__.__name__}_{self.time_span}'

class Scale(Preprocess):
    ''' Responsible for scaling the data by a fixed numer
        Inputs: Raw EEG in MNE format
        Outputs: Raw EED Data that is scaled
    '''
    def __init__(self, scale):
        self.scale = scale

    def func(self, data):
        data._data *= self.scale
        return data

    def get_id(self):
        return f'{self.__class__.__name__}_{self.scale}'

class BipolarRef(Preprocess):
    ''' Responsible for applying a bipolar reference to the data
        Inputs: raw EEG data in MNE format
        Outputs: raw EEG data with a bipolar reference applied
    '''
    def __init__(self, pairs, channels):
        self.pairs = pairs
        self.channels=channels
    def func(self, data):
        for anode, cathode in self.pairs:
            data = mne.set_bipolar_reference(data.load_data(), anode=[anode], cathode=[cathode], ch_name=f'{anode}-{cathode}', drop_refs=False, copy=True, verbose=False)
        data.drop_channels(ch_names=self.channels)
        return data

class WindowData(Preprocess):
    ''' Slices the continuous data into windows (epochs) of a specified
        duration and overlap.
    '''
    def __init__(self, window_duration=2.0, overlap_ratio=0):
        '''
        Args:
            window_duration (float): The length of each window in seconds.
            overlap_ratio (float): The fraction of overlap between consecutive windows (0.0 to 1.0).
        '''
        self.window_duration = window_duration
        self.overlap_ratio = overlap_ratio
        self.overlap_duration = window_duration * overlap_ratio

    def func(self, data):
        ''' Applies the windowing to the MNE Raw data object.

            Note: This function returns an MNE Epochs object, not a Raw object.
                  It should be the last step in a processing pipeline.
        '''
        epochs = mne.make_fixed_length_epochs(
            data,
            duration=self.window_duration,
            overlap=self.overlap_duration,
            preload=True, # Load data into memory
            verbose=False
        )
        return epochs

    def get_id(self):
        ''' Returns a unique ID for this preprocessing step. '''
        return f"{self.__class__.__name__}_{self.window_duration}s_{self.overlap_ratio}o"

class Pipeline(Preprocess):
    ''' Pipeline class defines the preprocessing pipeline for the EEG data.
        Keeps the pipeline for preprocessing the data
    '''
    def __init__(self):
        ''' Constructor Function
            INPUT:
                pipeline - list - list of functions to be applied to the data
        '''

        self.pipeline = []
        self.sampling_rate = -1
        self.time_span = -1
        self.channels = -1

    def __iter__(self):
        ''' Returns the iterator for the pipeline
        '''
        return iter(self.pipeline)

    def add(self, func):
        ''' Adds a function to the pipeline
            INPUT:
                func - function - function to be added to the pipeline
        '''
        if (func.__class__.__name__ == 'ResampleData'):
            self.sampling_rate = func.sample_rate
        if (func.__class__.__name__ in ['CropData', 'PaddedCropData']):
            self.time_span = func.time_span
        if (func.__class__.__name__ == 'ReduceChannels'):
            self.channels = len(func.channels)
        if (func.__class__.__name__ == 'BipolarRef'):
            self.channels = len(func.pairs)
        self.pipeline.append(func)

    def __add__(self, pipeline):
        ''' Adds a function to the pipeline
            INPUT:
                pipeline - function - function to be added to the pipeline
                pipeline - another list to be added to the pipeline
        '''
        new_pipeline = Pipeline()
        new_pipeline.pipeline = self.pipeline + pipeline.pipeline
        if (pipeline.sampling_rate != -1):
            new_pipeline.sampling_rate = pipeline.sampling_rate
        if (pipeline.time_span != -1):
            new_pipeline.time_span = pipeline.time_span
        if (pipeline.channels != -1):
            new_pipeline.channels = pipeline.channels
        return new_pipeline

    def func(self, data):
        ''' Applies the pipeline to the data
            INPUT:
                data - EEG - data to be preprocessed
            OUTPUT:
                data - EEG - preprocessed data
        '''
        for func in self.pipeline:
            data = func.func(data)
        return data

    def get_id(self):
        return super().get_id() + '_' + '_'.join([func.get_id() for func in self.pipeline])

def get_scnet_pipeline():
    '''Returns the preprocessing pipeline for SCNet Model
    '''
    pipeline = Pipeline()
    pipeline.add(PaddedCropData(60, 60 + 7 * 60))
    pipeline.add(ReduceChannels(channels= NMT_CHANNELS))
    pipeline.add(BipolarRef(pairs=NMT_PAIRS, channels= NMT_CHANNELS))
    pipeline.add(ResampleData(100))
    pipeline.add(ClipAbsData(100))
    pipeline.add(Scale(1e6))
    return pipeline

def local_pipeline(dataset='NMT'):
    pipeline = Pipeline()
    pipeline.add(ReduceChannels(channels=LOCAL_CHANNELS))
    pipeline.add(Scale(1e6))
    pipeline.add(ResampleDataKaiser(200))
    pipeline.add(WindowData())
    return pipeline

def resample():
    pipeline = Pipeline()
    pipeline.add(ReduceChannels(channels=LOCAL_CHANNELS))
    pipeline.add(ResampleData(200))
    return pipeline
