import pandas
import argparse
import pdb
from mhcflurry.regression_target import from_ic50
import sklearn 

def auc(
        predictions,
        actual,
        inequalities=None,
        threshold_nm=500,
        affinities_are_already_01_transformed=False):
    predictions = from_ic50(predictions)
    #actual = from_ic50(actual)
    #pdb.set_trace()
    auc = sklearn.metrics.roc_auc_score(
            actual <= threshold_nm,
            predictions)
    
    return auc


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str)
args = parser.parse_args()

data = pandas.read_csv(args.data)
data.dropna(inplace=True)
for allele in data.allele.unique():
    allele_df = data.loc[data['allele'] == allele]
    if len(allele_df.measurement_value.unique()) < 2:
        continue
    else:
        auc_score = auc( allele_df[str(0)], allele_df['measurement_value'] )
        print("%s %.4f" % (allele, auc_score)) 
