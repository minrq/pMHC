import argparse
from mhcflurry import Class1PresentationPredictor

def generate_random(allele, rollout):
    peptide_len = np.random.choice(np.arange(8, 15), 1)
    peptides = ["".join(list(np.random.choice(AMINO_ACIDS, plen))) for plen in peptide_len]

    predictions = self.predictor.predict(peptides=peptides, alleles=[allele], verbose=0) 
    return peptides, predictions['presentation_score']

def generate_motif(allele, rollout, vocab):
    s

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--alleles", type=str)
    parser.add_argument("--out", type=str)
    parser.add_argument("--rollout", type=int)
    parser.add_argument("--type", type=str)
    
    args = parser.parse_args()

    alleles = [allele.strip() for allele in open(args.alleles, 'r')]
    results = {}
    result_scores = {}
    
    if args.type == "random":
        for allele in alleles:
            peptides, scores = generate_random(allele, args.rollout)
            results[allele] = peptides
            result_scores[allele] = scores

        f = open(args.out, 'r')
        for allele in results:
            for peptide, score in zip(results[allele], result_scores[allele]):
                f.write("%s %s %.4f\n" % (allele, peptide, score))

        f.close()

    elif args.type == "motif":
        for allele in alleles:
            if 
