"""
Utilities for experimentation
"""
from numpy.random import choice, randint
from numpy import prod, sum, array

def generate_polynomial_map(domain_dims, codomain_dims):
    """ Generates a function where each dimension of the codomain
        is a polynomial over the domain dimensions 
        
        Used here for assessing the practical implications of 
        universal approximation theory """
    output_polys = list()
    domain_indices = list(range(domain_dims))

    """ Construct a polynomial for each output dimension """
    for _ in range(codomain_dims):
        """ Determine the number of summands in the polynomial """
        n_terms = randint(1, min(domain_dims, 5))
        """ Determine which variables to multiply together in each summand """
        terms = [
            choice(domain_indices, randint(1, min(domain_dims, 5)))
            for __ in range(n_terms)
        ]
        output_polys.append(terms)

    """ Return function that will map from domain to codomain
        Also returns its definition (i.e. output_polys) """
    return (
        lambda X : array([
            sum([
                prod([
                    X[i] for i in summand_terms
                ])
                for summand_terms in poly
            ])
            for poly in output_polys
        ]), 
        output_polys 
    )

