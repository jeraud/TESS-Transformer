# TESS Transformer

To train or perform inference, edit the appriate config.yaml file, then run train.py or inference.py. Consult the comments in the config files for further explanation.

Training:

Data can be either be the paths to .pt files containing time, flux, and label tensors if the user already has tensors saved(note that doing so assumes data is already preprocessed), or data can be a directory containing .fits files and a targets file. The targets file should be a csv or ecsv with the following columns: TICID, Class, Method, Path_to_lightcurve_file. More information on how to input paths to data can be found in the train_config.yaml file under Data.


Inference:

As with training, data can be the paths to .pt files containing time, flux, and ticid tensors if the user already has tensors saved. Otherwise, data should be a directory containing .fits files. More information can be found in the inference_config.yaml file under Data. 