import csv
import itertools
import sys

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """
    
    def prob_of_passing(name):
        """
        Returns the probability that the named parent will pass an abnormal gene depending
        on the number of copies of abnormal gene they have (0, 1, or 2).
        """
        if name in zero_genes: # Name has no abnormal genes.
            return PROBS['mutation']
        elif name in one_gene: # Name has one abnormal gene.
            return (1 - PROBS['mutation'])/2
        else: # Name has two abnormal genes.
            return 1 - PROBS['mutation']
    
    def get_p(name, mother, father, genes):
        """Calculates the probability that a person will receive 0, 1, or 2 of the
        abnormal genes from their parents. 
        
        Keyword arguments:
        name -- name of the person.
        mother -- name of the person's mother.
        father -- name of the person's father.
        genes -- number of genes received by the person.
        Return: probability that 'name' receives 'genes' number of the abnormal gene.
        """
        # Check for parents (if persons have one parent, then they have both parents.)
        if mother:
            mpp = prob_of_passing(mother) # mpp - mother's probability of passing.
            fpp = prob_of_passing(father) # fpp - father's probability of passing.
            p = mpp * (1 - fpp) + fpp * (1 - mpp)
        else: # No parents provided in the data file. Use the unconditional probability.
            p = PROBS['gene'][genes]
        # Include have_trait status in the calculation of p.
        if name in have_trait:
            p *= PROBS['trait'][genes][True]
        else:
            p *= PROBS['trait'][genes][False]
        return p
        
    p_0, p_1, p_2 = 0, 0, 0 # Probabilities of 0, 1, and 2 genes respectively.
    # Get set of people with zero genes.
    zero_genes = set(list(p for p in people if p not in one_gene and p not in two_genes))
    # Get the probabilities for receiving 0, 1, and 2 abnormal genes.
    for person in people:
        if people[person]['name'] in zero_genes:
            p_0 = get_p(people[person]['name'], people[person]['mother'], people[person]['father'], 0)

        if people[person]['name'] in one_gene:
            p_1 = get_p(people[person]['name'], people[person]['mother'], people[person]['father'], 1)

        if people[person]['name'] in two_genes:
            p_2 = get_p(people[person]['name'], people[person]['mother'], people[person]['father'], 2)

    return p_0 * p_1 * p_2


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    for person in probabilities:
        if person in one_gene:
            probabilities[person]['gene'][1] += p
        elif person in two_genes:
            probabilities[person]['gene'][2] += p
        else:
            probabilities[person]['gene'][0] += p
        if person in have_trait:
            probabilities[person]['trait'][True] += p
        else:
            probabilities[person]['trait'][False] += p


def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    for person in probabilities:
        gene_values = list(probabilities[person]['gene'].values())
        alpha = 1/sum(list(gene_values))
        probabilities[person]['gene'][0] *= alpha
        probabilities[person]['gene'][1] *= alpha
        probabilities[person]['gene'][2] *= alpha
        
        trait_values = list(probabilities[person]['trait'].values())
        print(trait_values)
        alpha = 1/sum(list(trait_values))
        probabilities[person]['trait'][True] *= alpha
        probabilities[person]['trait'][False] *= alpha


if __name__ == "__main__":
    main()
