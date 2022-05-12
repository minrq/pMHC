To learn how to train the MHCflurry models, please check https://github.com/openvax/mhcflurry/blob/master/downloads-generation/models_class1_pan/GENERATE.sh for more details



## Get Training Data

```
run the file below under the directory "./downloads-generation/models_class1_pan/"


python reassign_mass_spec_training_data.py \
    "$(mhcflurry-downloads path data_curated)/curated_training_data.csv.bz2" \
    --set-measurement-value 100 \
    --out-csv "$(pwd)/train_data.csv"
bzip2 -f "$(pwd)/train_data.csv"
TRAINING_DATA="$(pwd)/train_data.csv.bz2"
```





## Preparing Pseudo Data

1. use RL model to generate a lot of positive peptides with respect to some alleles.

2. append the generated pseudo data to the training data file.

   ```
   python ./append_pseudo_data.py --data "$(pwd)/train_data.csv.bz2" --pseudo <generated pseudo sequences from RL> --num <number of appended pseudo data for each allele> --out <output data file path and name>
   ```

   

## Training commands

1. generate the hyperparameters for the training of mhcflurry

   ```
   python generate_hyperparameters.py > hyperparameters.yaml
   ```

   you can decrease the below sets of hyper-parameters to one set by changing the file "generate_hyperparameters.py", or train the same model with different hyperparameters for multiple times, and calculate the results of each ensemble with multiple models.

   ```
   for layer_sizes in [[512, 256], [512, 512], [1024, 512], [1024, 1024]]:
       l1_base = 0.0000001
       for l1 in [l1_base, l1_base / 10, l1_base / 100, l1_base / 1000, 0.0]:
   
   for layer_sizes in [[256, 512], [256, 256, 512], [256, 512, 512]]:
       l1_base = 0.0000001
       for l1 in [l1_base, l1_base / 10, l1_base / 100, l1_base / 1000, 0.0]:
   ```

    

2. Train the mhcflurry model

```
mhcflurry-class1-train-pan-allele-models --data <data appended with pseudo data> --allele-sequences $(mhcflurry-downloads path allele_sequences)/allele_sequences.csv --pretrain-data  $(mhcflurry-downloads path random_peptide_predictions)/predictions.csv.bz2 --held-out-measurements-per-allele-fraction-and-max 0.25 100 --num-folds 4 --hyperparameters ./hyperparameters.yaml --out-models-dir <your path> --worker-log-dir <text log file path and name> --continue-incomplete
```

To accelerate the training, take a look at the generate.sh "$PARALLELISM_ARGS" .





## Test commands

1. select the data for the test alleles and append decoy to the data set

   ```
   python ./append_decoy_to_data.py --test_alleles <a file with all the test alleles> --data <your out-models-dir path>/train_data.csv.bz2
   ```

   MHCflurry splits the dataset into 4 folds automatically within their code during the running, and saves the splits of dataset "train_data.csv.bz2" into the \<your out-models-dir path\> in a format as below. (I don't think they provide an option for the dataset that has been split into several folds.)

   To make sure that the MHCflurry models trained with 3 folds are tested on the held-out fold. It is necessary to append decoys to the dataset that is split by MHCflurry.

   I followed the way that MHCflurry used to generate decoys. That is, for each sequence in the dataset, we sample 4 sequences from proteasome with length 8, 9, 10, 11. So the task is to distinguish the positive peptides from known negative peptides and random sequences sampled from proteasome.

   ![image-20220511210839102](C:\Users\50303\AppData\Roaming\Typora\typora-user-images\image-20220511210839102.png)

2. evaluate the unselected model through evaluate_allele_models.py

(I think it would be better if the model is selected, but I didn't do it)

    python ./evaluate_allele_models.py --data <your appended data file path and name> --out_models_dir <your model path> --test_alleles <file with all your test alleles> --out <your output file with all the predictions>
3. calculate auc values for the predictions.