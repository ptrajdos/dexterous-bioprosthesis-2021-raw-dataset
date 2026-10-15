"""Module providing scikit-learn compatibility helpers.

scikit-learn >= 1.6 reads estimator tags (``__sklearn_tags__``), e.g. when a
Pipeline checks whether its last step is fitted. Older versions never call
``__sklearn_tags__``, so sklearn tag classes are imported lazily.
"""


def sklearn_transformer_tags(requires_fit=True, target_required=False):
    """Return scikit-learn tags of a transformer.

    Arguments:
    ---------
    requires_fit -- whether the transformer has to be fitted before transform
    target_required -- whether fit requires y

    Returns:
    --------
    sklearn.utils.Tags
    """
    from sklearn.utils import Tags, TargetTags, TransformerTags

    return Tags(
        estimator_type=None,
        target_tags=TargetTags(required=target_required),
        transformer_tags=TransformerTags(),
        requires_fit=requires_fit,
    )
