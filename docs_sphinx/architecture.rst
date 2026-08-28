Architecture & Pipeline
=======================

This page provides interactive diagrams illustrating the library's data processing pipeline and module architecture.

Data Processing Pipeline
------------------------

.. mermaid::

   graph LR
       A[Raw EMG Recordings] --> B[Signal Extraction]
       B --> C[Data Augmentation]
       C --> D[Dataset Creation]
       D --> E[NumPy / ARFF Output]
       D --> F[Hyperparameter Optimisation]
       F -->|feedback| B
       F -->|feedback| C

       style A fill:#e1f5fe
       style E fill:#c8e6c9
       style F fill:#fff9c4

Signal Extraction Flow
----------------------

.. mermaid::

   graph TD
       RS[RawSignals] --> |iterate| RSig[RawSignal]
       RSig --> SE[SignalExtractor]
       SE --> |extract features| NP[NumPy Array]
       SE --> DRMS[DeltaRMS Extractor]
       SE --> Custom[Custom Extractors]

       style RS fill:#e1f5fe
       style NP fill:#c8e6c9

Dataset Creation
----------------

.. mermaid::

   graph LR
       SC[SetCreator] --> SCD[SetCreatorDefault]
       SC --> SCM[SetCreatorMDS]
       SC --> SCA[SetCreatorAugmented]
       SCD --> DS[Dataset]
       SCM --> DS
       SCA --> DS
       DS --> ARFF[ARFF Format]
       DS --> NPY[NumPy Format]

       style SC fill:#e1f5fe
       style ARFF fill:#c8e6c9
       style NPY fill:#c8e6c9

Augmentation Pipeline
---------------------

.. mermaid::

   graph TD
       RAW[Raw Signal] --> AUG[Augmenter Base]
       AUG --> TW[Time Warping - DTW]
       AUG --> NI[Noise Injection]
       AUG --> AA[Audio Augmentations]
       TW --> OUT[Augmented Signal]
       NI --> OUT
       AA --> OUT

       style RAW fill:#e1f5fe
       style OUT fill:#c8e6c9

Module Dependency Overview
--------------------------

.. mermaid::

   graph LR
       raw_signals --> set_creators
       raw_signals --> data_augmentation
       data_augmentation --> set_creators
       set_creators --> np_signal_extractors
       np_signal_extractors -.-> scikit-learn
       data_augmentation -.-> librosa
       data_augmentation -.-> audiomentations
       set_creators -.-> hyperopt
       set_creators -.-> pygad

       style raw_signals fill:#e1f5fe
       style set_creators fill:#fff9c4
       style data_augmentation fill:#f3e5f5

Class Inheritance
-----------------

Each API page includes an auto-generated inheritance diagram (via ``sphinx.ext.inheritance_diagram``).
Browse the :doc:`api/modules` section to see interactive class hierarchy graphs for every module.
