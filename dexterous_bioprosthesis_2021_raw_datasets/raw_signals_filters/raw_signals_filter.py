import abc

from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals

class RawSignalsFilter(abc.ABC):
    """
    Class that represents an interface for filters
    """
    @abc.abstractmethod
    def fit(self, raw_signals: RawSignals)->None:
        """
        Fits the filter.
        Arguments:
        ---------
        raw_signals --- An object of RawSignals class to be fitted with
        """
        
    @abc.abstractmethod
    def transform(self,raw_signals: RawSignals)->RawSignals:
        """
        Transforms the RawSignals object
        Arguments:
        ---------
        raw_signals --- RawSignals object to be filtered

        Returns:
        --------
        RawSignals object
        """

    def fit_transform(self,raw_signals:RawSignals)->RawSignals:
        """
        Fits and then transforms the given RawSignals object

        Arguments:
        ---------
        raw_signals --- RawSignals object to be filtered

        Returns:
        --------
        RawSignals object
        """
        self.fit(raw_signals)
        return self.transform(raw_signals)