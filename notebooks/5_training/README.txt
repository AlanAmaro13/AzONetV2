Folder Description: 

    This folder contains all the U-NET models developed. The relevant model corresponds to F145F2. This notebook has this advantages: 

    * This notebook uses the simulated data F145F. 
    * There's NO use of EarlyStopping nor RLonPlateu. 
    * There's use of Checkpoint over the validation MAPE (minimum flow)
    * The model has been CORRECTED: Batch Normalization and Activation has been added and UpSampling is not added when is not required. 
    * The loss function is MAE, and the metrics are: MAPE, RMSE. 
    * The model is trained for 100 epochs with a BatchSize of 512 samples. 
    * The Checkpoint model is .keras, and the complete trained model is .h5. 