"""Module providing a Pipeline subclass that exposes SetCreator-specific methods.

Extends :class:`sklearn.pipeline.Pipeline` to delegate
:meth:`get_channel_attribs_indices` to the underlying :class:`SetCreator` step.
"""
from sklearn.pipeline import Pipeline

from dexterous_bioprosthesis_2021_raw_datasets.set_creators.set_creator import SetCreator


class SetCreatorPipeline(Pipeline):
    """Pipeline subclass exposing SetCreator-specific methods.

    Automatically locates the :class:`SetCreator` step in the pipeline
    and delegates calls such as :meth:`get_channel_attribs_indices` to it.
    """

    def _find_set_creator(self) -> SetCreator:
        """Return the first SetCreator step found in the pipeline.

        Raises
        ------
        ValueError
            If no SetCreator step is found.
        """
        for name, step in self.steps:
            if isinstance(step, SetCreator):
                return step
        raise ValueError("No SetCreator step found in the pipeline.")

    def get_channel_attribs_indices(self):
        """Delegate to the SetCreator step's get_channel_attribs_indices."""
        return self._find_set_creator().get_channel_attribs_indices()

    def get_channel_names(self):
        """Delegate to the SetCreator step's get_channel_names."""
        return self._find_set_creator().get_channel_names()
