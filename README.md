# POS_MEMM
This project was written by Carmel Rabinovitz and Amir Livne as part of NLP Technion course
This is a part of speech tagger (using max entropy).
We reached 93.7% accuracy over test set
 
## How to use:
To run the POS MEMM on comp data use:
```
python main.py <your results directory name> -comp_path <path to comp> -data_path <path to train data>
```

if -comp_path is no provided it will defualtly use:
current script directory + '\\data\\comp.words'

if -data_path is no provided it will defualtly use:
current script directory + '\\data\\data.words'

if you already trained the model and just want to predict use in adition:
	-trained_model <your results directory name>

you can also run it test mode if you have labled test data, in this case run:
```
python main.py <your results directory name> -test_path <path to test data> -data_path <path to train data> -oparation_mode test
```
	
## Results:
The script will create a directory named "results directory name" containing the results files:
  1. confusion_matrix.csv
  2. max_failure_matrix.csv - 10 most failed to tag
  3. model.pkl - the POS MEMM model
  4. predictions.txt - predictions of the test data provided
  5. predictions_logs.pkl
  6. logs.txt - information about the training congifurations

additional flags:
1. -regularization <#>, 0.0005 is default value
2. -word_threshold <#>, use words that apeared less then # times in training data for unknown word feature
3. -spelling_threshold <#>, use suffix and prefix that apeared more then # times in training data, 5 is default value
4. -verbosity <0,1>, 1-print logs in each epoch and each predicted sentence, 0-without printings, o is default value
5. -mode <base, complex>, complex is default value
6. -parallel, in prder to use multiproccesing prediction	
