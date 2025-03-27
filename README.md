Group Members:  
24034383 LSLL8  
24046706 LQQB5  
24055542 MNLS5  

This is a short introduction of our SNS project files.

We built single-step and multi-step prediction models:  
Single-step prediction model with LSTM, GRU, CNN-LSTM: Single-step model with plots.py      
Multi-step prediction model with LSTM, GRU, CNN-LSTM: Multi-step model with plots.py

The best two models and feature scalers are saved as well:  
Best single-step prediction:  
model:single-step_model.h5  
scalers: feature_scaler_single-step.pkl and target_scaler_single-step.pkl

Best multi-step prediction:  
model:multi-step_model.h5  
scalers: feature_scaler_multi-step.pkl and target_scaler_multi-step.pkl

We use two data sets.   
The first one is separated into 7:2:1 for training, validation and initial testing: london_weather.csv  
The second one is used to evaluate the generalization ability of the two best models: 2021-2023.csv

The prediction results for the second data set are shown in the following 2 .py files:  
single-step generalization: Testset2_2020-23_singlestep.py  
multi-step generalization: Testset2_2020-23_multistep.py

The two interaction models are built based on the best single-step prediction model and multi-step prediction model:  
Client-Server for single-step prediction:(test file: 5days.txt)    
Oracle_server_single-step.py  
Oracle_client_single-step.py  

Client-Sever for multi-step prediction:(test file: 30days.csv)  
Oracle_server_multi-step.py  
Oracle_client_multi-step.py  


