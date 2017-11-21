Configuration File
==================

This is the core of the algorithm, so this file has to be filled properly based on your data. Even if all key parameters of the algorithm are listed in the file, only few are likely to be modified by a non-advanced user. The configuration file is divided in several sections. For all those sections, we will review the parameters, and tell you what are the most important ones

Data
----

The data section is::

    file_format    =                       # Can be raw_binary, openephys, hdf5, ... See >> spyking-circus help -i for more info
    stream_mode    = None                  # None by default. Can be multi-files, or anything depending to the file format
    mapping        = ~/probes/mea_252.prb  # Mapping of the electrode (see http://spyking-circus.rtfd.ord)
    suffix         =                       # Suffix to add to generated files
    global_tmp     = True                  # should be False if local /tmp/ has enough space (better for clusters)
    overwrite      = True                  # If you want to filter or remove artefacts on site. Data are duplicated otherwise

.. warning::

    This is the most important section, that will allow the code to properly load your data. If not properly filled, then results will be wrong. Note that depending on your file_format, you may need to add here several parameters, such as ``sampling_rate``, ``data_dtype``, ... They will be requested if they can not be infered from the header of your data structure. To check if data are properly loaded, consider using :doc:`the preview mode <../GUI/python>` before launching the whole algorithm

Parameters that are most likely to be changed:
    * ``file_format`` You must select a supported file format (see :doc:`What are the supported formats <../code/fileformat>`) or write your own wrapper (see :doc:`Write your own data format  <../advanced/datafile>`)
    * ``mapping`` This is the path to your probe mapping (see :doc:`How to design a probe file <../code/probe>`)
    * ``global_temp`` If you are using a cluster with NFS, this should be False (local /tmp/ will be used by every nodes)
    * ``stream_mode`` If streams in you data (could be multi-files, or even in the same file) should be processed together (see :doc:`Using multi files <../code/multifiles>`)
    * ``overwrite`` If True, data are overwritten during filtering, assuming the file format has write access. Otherwise, an external raw_binary file will be created during the filtering step, if any.

Detection
---------

The detection section is::

    radius         = auto       # Radius [in um] (if auto, read from the prb file)
    N_t            = 5          # Width of the templates [in ms]
    spike_thresh   = 6          # Threshold for spike detection
    peaks          = negative   # Can be negative (default), positive or both
    matched-filter = False      # If True, we perform spike detection with matched filters
    matched_thresh = 5          # Threshold for detection if matched filter is True
    alignment      = True       # Realign the waveforms by oversampling

Parameters that are most likely to be changed:
    * ``N_t`` The temporal width of the templates. For *in vitro* data, 5ms seems a good value. For *in vivo* data, you should rather use 3 or even 2ms
    * ``radius`` The spatial width of the templates. By default, this value is read from the probe file. However, if you want to specify a larger or a smaller value [in um], you can do it here
    * ``spike_thresh`` The threshold for spike detection. 6-7 are good values
    * ``peaks`` By default, the code detects only negative peaks, but you can search for positive peaks, or both
    * ``matched-filter`` If activated, the code will detect smaller spikes by using matched filtering
    * ``matched_thresh`` During matched filtering, the detection threshold
    * ``alignment`` By default, during clustering, the waveforms are realigned by oversampling at 5 times the sampling rate and using bicubic spline interpolation
    
Filtering
---------

The filtering section is::

    cut_off        = 300, auto # Min and Max (auto=nyquist) cut off frequencies for the band pass butterworth filter [Hz]
    filter         = True      # If True, then a low-pass filtering is performed
    remove_median  = False     # If True, median over all channels is substracted to each channels (movement artefacts)

.. warning::

    The code performs the filtering of your data writing on the file itself. Therefore, you ``must`` have a copy of your raw data elsewhere. Note that as long as your keeping the parameter files, you can relaunch the code safely: the program will not filter multiple times the data, because of the flag ``filter_done`` at the end of the configuration file.

Parameters that are most likely to be changed:
    * ``cut_off`` The default value of 500Hz has been used in various recordings, but you can change it if needed. You can also specify the upper bound of the Butterworth filter
    * ``filter`` If your data are already filtered by a third program, turn that flag to False
    * ``remove_median`` If you have some movement artefacts in your *in vivo* recording, and want to substract the median activity over all analysed channels from each channel individually

Triggers
--------

The triggers section is::

    trig_file      =            # External stimuli to be considered as putative artefacts [in trig units] (see documentation)
    trig_windows   =            # The time windows of those external stimuli [in trig units]
    trig_unit      = ms         # The unit in which times are expressed: can be ms or timestep
    clean_artefact = False      # If True, external artefacts induced by triggers will be suppressed from data
    dead_file      =            # Portion of the signals that should be excluded from the analysis [in dead units]
    dead_unit      = ms         # The unit in which times for dead regions are expressed: can be ms or timestep
    ignore_times   = False      # If True, any spike in the dead regions will be ignored by the analysis
    make_plots     =            # Generate sanity plots of the averaged artefacts [Nothing or None if no plots]

Parameters that are most likely to be changed:
    * ``trig_file`` The path to the file where your artefact times and labels. See :doc:`how to deal with stimulation artefacts <../code/artefacts>`
    * ``trig_windows`` The path to file where your artefact temporal windows. See :doc:`how to deal with stimulation artefacts <../code/artefacts>`
    * ``clean_artefact`` If you want to remove any stimulation artefacts, defined in the previous files. See :doc:`how to deal with stimulation artefacts <../code/artefacts>`
    * ``make_plots`` The default format to save the plots of the artefacts, one per artefact, showing all channels. You can set it to None if you do not want any
    * ``trig_unit`` If you want times/duration in the ``trig_file`` and ``trig_windows`` to be in timestep or ms
    * ``dead_file`` The path to the file where the dead portions of the recording, that should be excluded from the analysis, are specified. . See :doc:`how to deal with stimulation artefacts <../code/artefacts>`
    * ``dead_unit`` If you want times/duration in the ``dead_file`` to be in timestep or ms
    * ``ignore_times`` If you want to remove any dead portions of the recording, defined in ``dead_file``. See :doc:`how to deal with stimulation artefacts <../code/artefacts>`

Whitening
---------

The whitening section is::

    chunk_size     = 60        # Size of the data chunks [in s]
    safety_time    = 1         # Temporal zone around which templates are isolated [in ms]
    temporal       = False     # Perform temporal whitening
    spatial        = True      # Perform spatial whitening
    max_elts       = 10000     # Max number of events per electrode (should be compatible with nb_elts)
    nb_elts        = 0.8       # Fraction of max_elts that should be obtained per electrode [0-1]
    output_dim     = 5         # Can be in percent of variance explain, or num of dimensions for PCA on waveforms

Parameters that are most likely to be changed:
    * ``output_dim`` If you want to save some memory usage, you can reduce the number of features kept to describe a waveform.
    * ``chunk_size`` If you have a very large number of electrode, and not enough memory, you can reduce it


Clustering
----------

The clustering section is::

    extraction     = median-raw # Can be either median-raw (default), median-pca, mean-pca, mean-raw, or quadratic
    safety_space   = True       # If True, we exclude spikes in the vicinity of a selected spikes
    safety_time    = 1          # Temporal zone around which templates are isolated [in ms]
    max_elts       = 10000      # Max number of events per electrode (should be compatible with nb_elts)
    nb_elts        = 0.8        # Fraction of max_elts that should be obtained per electrode [0-1]
    nclus_min      = 0.002      # Min number of elements in a cluster (given in percentage)
    max_clusters   = 10         # Maximal number of clusters for every electrodes
    nb_repeats     = 3          # Number of passes used for the clustering
    make_plots     =            # Generate sanity plots of the clustering
    sim_same_elec  = 3          # Distance within clusters under which they are re-merged
    cc_merge       = 0.975      # If CC between two templates is higher, they are merged
    dispersion     = (5, 5)     # Min and Max dispersion allowed for amplitudes [in MAD]
    smart_search   = True       # Parameter to activate the smart search mode
    smart_select   = False      # Experimental: activate the smart selection of centroids (max_clusters is ignored)
    noise_thr      = 0.8        # Minimal amplitudes are such than amp*min(templates) < noise_thr*threshold
    remove_mixture = True       # At the end of the clustering, we remove mixtures of templates

.. note::

    This is the a key section, as bad clustering will implies bad results. However, the code is very robust to parameters changes.

Parameters that are most likely to be changed:
    * ``extraction`` The method to estimate the templates. ``Raw`` methods are slower, but more accurate, as data are read from the files. ``PCA`` methods are faster, but less accurate, and may lead to some distorted templates. ``Quadratic`` is slower, and should not be used.
    * ``max_elts`` The number of elements that every electrode will try to collect, in order to perform the clustering
    * ``nclus_min`` If you have too many clusters with few elements, you can increase this value. This is expressed in percentage of collected spike per electrode. So one electrode collecting *max_elts* spikes will keep clusters with more than *nclus_min.max_elts*. Otherwise, they are discarded
    * ``max_clusters`` This is the maximal number of cluster that you expect to see on a given electrode. For *in vitro* data, 10 seems to be a reasonable value. For *in vivo* data and dense probes, you should set it to 10-15. Increase it only if the code tells you so.
    * ``nb_repeats`` The number of passes performed by the algorithm to refine the density landscape
    * ``smart_search`` By default, the code will collect only a subset of spikes, randomly, on all electrodes. However, for long recordings, or if you have low thresholds, you may want to select them in a smarter manner, in order to avoid missing the large ones, under represented. If the smart search is activated, the code will first sample the distribution of amplitudes, on all channels, and then implement a rejection algorithm such that it will try to select spikes in order to make the distribution of amplitudes more uniform. This can be very efficient, and may become True by default in future releases.
    * ``smart_select`` This option (experimental) should boost the quality of the clustering, by selecting the centroids in a automatic manner. If activated the ``max_clusters`` parameter is ignored
    * ``cc_merge`` After local merging per electrode, this step will make sure that you do not have duplicates in your templates, that may have been spread on several electrodes. All templates with a correlation coefficient higher than that parameter are merged. Remember that the more you merge, the faster is the fit
    * ``dispersion`` The spread of the amplitudes allowed, for every templates, around the centroid.
    * ``remove_mixture`` By default, any template that can be explained as sum of two others is deleted. 
    * ``make_plots`` By default, the code generates sanity plots of the clustering, one per electrode.

Fitting
-------

The fitting section is::

    chunk          = 1         # Size of chunks used during fitting [in second]
    gpu_only       = True      # Use GPU for computation of b's AND fitting
    amp_limits     = (0.3, 30) # Amplitudes for the templates during spike detection
    amp_auto       = True      # True if amplitudes are adjusted automatically for every templates
    max_chunk      = inf       # Fit only up to max_chunk   
    collect_all    = False      # If True, one garbage template per electrode is created, to store unfitted spikes


Parameters that are most likely to be changed:
    * ``chunk`` again, to reduce memory usage, you can reduce the size of the temporal chunks during fitting. Note that it has to be one order of magnitude higher than the template width ``N_t``
    * ``gpu_only`` By default, all operations will take place on the GPU. However, if not enough memory is available on the GPU, then you can turn this flag to False. 
    * ``max_chunk`` If you just want to fit the first *N* chunks, otherwise, the whole file is processed
    * ``collect_all`` If you want to also collect all the spike times at which no templates were fitted. This is particularly useful to debug the algorithm, and understand if something is wrong on a given channel

Merging
-------

The merging section is::

    cc_overlap     = 0.5       # Only templates with CC higher than cc_overlap may be merged
    cc_bin         = 2         # Bin size for computing CC [in ms]
    correct_lag    = False     # If spikes are aligned when merging. May be better for phy usage
    auto_mode      = 0         # If >0, merging will be automatic (see doc, 0.15 is a good value) [0-1]

To know more about how those merges are performed and how to use this option, see :doc:`Automatic Merging <../code/merging>`. Parameters that are most likely to be changed:
    * ``correct_lag`` By default, in the meta-merging GUI, when two templates are merged, the spike times of the one removed are simply added to the one kept, without modification. However, it is more accurate to shift those spike, in times, by the temporal shift that may exist between those two templates. This will lead to a better visualization in phy, with more aligned spikes
    * ``auto_mode`` If your recording is stationary, you can try to perform a fully automated merging. By setting a value between 0 and 1, you control the level of merging performed by the software. Values such as 0.15 should be a good start, but see see :doc:`Automatic Merging <../code/merging>` for more details. 

Converting
----------

The converting section is::

    erase_all      = True      # If False, a prompt will ask you to export if export has already been done
    sparse_export  = False     # If True, data for phy are exported in a sparse format. Need recent version of phy
    export_pcs     = prompt    # Can be prompt [default] or in none, all, some
    export_all     = False     # If True, unfitted spikes will be exported as the last Ne templates


Parameters that are most likely to be changed:
    * ``erase_all`` If you want to always erase former export, and skip the prompt
    * ``sparse_export`` If you have a large number of templates or a very high density probe, you should use the sparse format for phy
    * ``export_pcs`` If you already know that you want to have all, some, or no PC and skip the prompt
    * ``export_all`` If you used the ``collect_all`` mode in the ``[fitting]`` section, you can export unfitted spike times to phy. In this case, the last `N` templates, if `N` is the number of electrodes, are the garbage collectors.

Extracting
----------

The extracting section is::

    safety_time    = 1         # Temporal zone around which spikes are isolated [in ms]
    max_elts       = 10000     # Max number of events per templates (should be compatible with nb_elts)
    nb_elts        = 0.8       # Fraction of max_elts that should be obtained per electrode [0-1]
    output_dim     = 5         # Percentage of variance explained while performing PCA
    cc_merge       = 0.975     # If CC between two templates is higher, they are merged
    noise_thr      = 0.8       # Minimal amplitudes are such than amp*min(templates) < noise_thr*threshold


This is an experimental section, not used by default in the algorithm, so nothing to be changed here

Validating
----------

The validating section is::

    nearest_elec   = auto      # Validation channel (e.g. electrode closest to the ground truth cell)
    max_iter       = 200       # Maximum number of iterations of the stochastic gradient descent (SGD)
    learning_rate  = 1.0e-3    # Initial learning rate which controls the step-size of the SGD
    roc_sampling   = 10        # Number of points to estimate the ROC curve of the BEER estimate
    test_size      = 0.3       # Portion of the dataset to include in the test split
    radius_factor  = 0.5       # Radius factor to modulate physical radius during validation
    juxta_dtype    = uint16    # Type of the juxtacellular data
    juxta_thresh   = 6         # Threshold for juxtacellular detection
    juxta_valley   = False     # True if juxta-cellular spikes are negative peaks
    juxta_spikes   =           # If none, spikes are automatically detected based on juxta_thresh
    filter         = True      # If the juxta channel need to be filtered or not
    make_plots     = png       # Generate sanity plots of the validation [Nothing or None if no plots]

Please get in touch with us if you want to use this section, only for validation purposes. This is an implementation of the :doc:`BEER metric <../advanced/beer>`
