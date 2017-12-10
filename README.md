# POS_MEMM
This project was written by Carmel Rabinovitz and Amir Livne as part of NLP technion course

## How to use:
run: python main.py "results directory name"

this will create a directory named "results directory name" containing these files:
  1. confusion_matrix.csv
  2. max_failure_matrix.csv - 10 most failed to tag
  3. model.pkl - the POS MEMM model
  4. predictions.txt - predictions of the test data provided
  5. predictions_logs.pkl
  6. logs.txt - information about the training congifurations

additional flags:
1. -regularization <#>, 0.02 is default value
2. -toy, to run toy examples
3. -threshold <#>, use suffix and prefix that apeared more then # times in training data, 5 is default value
4. -verbosity <0,1>, 1-print logs in each epoch and each predicted sentence, 0-without printings, o is default value
5. -mode <base, complex>, complex is default value

