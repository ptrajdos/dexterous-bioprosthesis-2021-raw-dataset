import abc

from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals

class RawSignalsAugumenter(abc.ABC):
    
    @abc.abstractmethod
    def fit(self, raw_signals:RawSignals):
        """
        Fits the data augumenter

        Arguments:
        ---------
        raw_signals: RawSignals -- the dataset to be augumented

        """

    def transform(self, raw_signals:RawSignals)->RawSignals:
        """
        Transforms the dataset. 
        New data contains changed version of each RawSignal in RawSignals

        Arguments:
        ---------
        raw_signals: RawSignals -- the dataset to be augumented

        Returns:
        --------
        Transformed RawSignals

        """

    def fit_transform(self, raw_signals:RawSignals)->RawSignals:
        """
        Fits and then transforms the dataset. 
        New data contains changed version of each RawSignal in RawSignals

        Arguments:
        ---------
        raw_signals: RawSignals -- the dataset to be augumented

        Returns:
        --------
        Transformed RawSignals

        """