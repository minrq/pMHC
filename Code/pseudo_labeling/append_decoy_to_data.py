import pandas
import mhcflurry
import numpy 
import pdb
import argparse

random_num = 2022

path = "/users/PES0781/ziqichen/.local/share/mhcflurry/4/2.0.0/data_references/uniprot_proteins.csv.bz2"
DECOY_UNIVERSE = pandas.read_csv(path, usecols=["seq"])
DECOY_UNIVERSE = pandas.Series(DECOY_UNIVERSE.seq.unique())
DECOY_UNIVERSE = DECOY_UNIVERSE.loc[
        DECOY_UNIVERSE.str.match("^[%s]+$" % "".join(
            mhcflurry.amino_acid.COMMON_AMINO_ACIDS)) & (
            DECOY_UNIVERSE.str.len() >= 50)
]
print("Read decoy universe from", path)

def make_decoys(num, length, seed=None):
    return DECOY_UNIVERSE.sample(num, replace=True, random_state=seed).map(
        lambda s: s[numpy.random.randint(0, len(s) - length):][:length]).values

def get_decoy(size, seed=None, decoy_per_hit=4):
    lengths = [8, 9, 10, 11]
    result_df = []
    
    for length in lengths:
        decoys = make_decoys(size, length, seed=seed)
        result_df.append(decoys)
        
    return result_df


def append_decoy(allele, data, seed=None):
    allele_df = data.loc[ (data['allele'] == allele) & ~(data['measurement_source'] == 'pseudo_label')].copy()

    decoys = numpy.concatenate(get_decoy(allele_df.shape[0], seed=seed))
    numpy.random.shuffle(decoys)
    new_df = pandas.DataFrame({'allele': allele_df.allele.repeat(4), 'peptide': decoys, 'fold_0': allele_df.fold_0.repeat(4),\
                               'fold_1': allele_df.fold_1.repeat(4), 'fold_2': allele_df.fold_2.repeat(4), 'fold_3': allele_df.fold_3.repeat(4)})
    
    new_df['measurement_inequality'] = '>'
    new_df['measurement_value'] = 20000
    
    allele_df = allele_df.append(new_df).reset_index(drop=True)
    
    return allele_df


parser = argparse.ArgumentParser()
parser.add_argument("--test_alleles", type=str)
parser.add_argument("--data", type=str)
parser.add_argument("--out", type=str)
args = parser.parse_args()

test_alleles = [line.strip() for line in open(args.test_alleles, 'r').readlines()]

data = pandas.read_csv(args.data)

new_data = []
for i, allele in enumerate(test_alleles):
    tmp = append_decoy(allele, data, seed=random_num+i)
    new_data.append(tmp)

new_data = pandas.concat(new_data).reset_index(drop=True)
new_data.to_csv(args.out)
