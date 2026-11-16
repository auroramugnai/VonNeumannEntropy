import qutip as qt
from tqdm import tqdm
import ncpol2sdpa as nc
import picos as pic
import sympy as sp
import sys
from ncpol2sdpa.nc_utils import flatten, apply_substitutions
from mosek.fusion import Model, Domain, Expr, ObjectiveSense

def generate_operators(dim, label, adj=False):
    """
    Generate a dim × dim table of symbolic operators for NPA-like moment matrices.

    Parameters
    ----------
    dim : int
        Hilbert space dimension and the size of the operator table.

    label : str
        Prefix for the symbolic operator names. For example, label="A" produces
        operators named A00, A01, etc.

    Returns
        dim × dim table of lists of SymPy noncommuting symbols.
    """

    # Initialize a dim × dim table
    operators = [[None for _ in range(dim)] for _ in range(dim)]
    for i in range(dim):
        for j in range(dim):
            # Define Sympy operators label, inverting indices if 'adj==True'
            name = f"{label}{j}{i}" if adj else f"{label}{i}{j}"

            if i==j:
                operators[i][j] = nc.generate_operators(name, hermitian=True)[0]
            if i!=j:
                operators[i][j] = nc.generate_operators(name, hermitian=False)[0]

    return operators
    

def generating_monomials(M, level, subs, adj=False):
    # Flatten the operators table to get all individual operators
    alphabet = flatten(M)

    # Start generating set with the identity and all basic operators

    Id = sp.S.One #identity operator
    generating_set =  [Id] + alphabet 
     
    # Helper function to handle product ordering
    def prod(x, y, z=None):
        """
        Multiply two or three operators, inverting ordering if 'adj==True'.
        """
        if z is None:
            return y*x if adj else x*y
        else:
            return z*y*x if adj else x*y*z

    # ------------------------
    # Level-specific monomials
    # ------------------------
    if level == 1.5:
        # Explicit per-block 2-products
        generating_set += [prod(x, y) for x in flatten(M[0]) for y in flatten(M[0])]
        generating_set += [prod(x, y) for x in flatten(M[1]) for y in flatten(M[1])]

    elif level == 2:
        # All 2-products of the full alphabet
        generating_set += [prod(x, y) for x in alphabet for y in alphabet]
        generating_set = [apply_substitutions(s, subs) for s in generating_set]

    elif level == 2.5:
        # 2-products of the full alphabet
        generating_set += [prod(x, y) for x in alphabet for y in alphabet]

        # 3-products within each block
        generating_set += [prod(x, y, z) for x in flatten(M[0]) for y in flatten(M[0]) for z in flatten(M[0])]
        generating_set += [prod(x, y, z) for x in flatten(M[1]) for y in flatten(M[1]) for z in flatten(M[1])]

        # Cross-block 3-products
        generating_set += [prod(x, y, z) for x in flatten(M[1]) for y in flatten(M[1]) for z in flatten(M[0])]

    elif level == 3:
        # All 3-products of the full alphabet
        generating_set += [prod(x, y, z) for x in alphabet for y in alphabet for z in alphabet]

    # Remove duplicates while preserving order
    generating_set = list(dict.fromkeys(generating_set).keys())
    
    return generating_set

                
def calculate_moment_matrix(measS, gen_set, gen_set_adj, dim, subs):
    """
    Construct a dictionary mapping each monomial to its positions in the moment matrix.

    Only the upper-triangular part of the moment matrix is constructed.

    Parameters
    ----------
    measS : list[list]
        dim × dim table of measurement operators (or symbols)
    gen_set : list
        Generating set of monomials
    gen_set_adj : list
        Adjoint generating set
    dim : int
        Hilbert space dimension

    Returns
    -------
    dict
        Keys are monomials (sympy expressions), values are lists of tuples
        indicating the position in the moment matrix.
    """
    Msym_positions = {}
    Id = sp.S.One  # Identity operator

    for j in tqdm(range(len(gen_set_adj)), desc="Computing moment matrix:"):  # rows
        for i in range(len(gen_set)):  # columns
            for l in range(dim):
                for k in range(dim):
                    # Start with measurement operator
                    monomial = measS[k][l]

                    # Multiply by generators if they are not the identity
                    if gen_set[i] != Id:
                        monomial *= gen_set[i]
                    if gen_set_adj[j] != Id:
                        monomial *= gen_set_adj[j]
                    monomial = apply_substitutions(monomial, subs)

                    # Store position in the dictionary
                    pos = (l * len(gen_set) + j, k * len(gen_set) + i)
                    if monomial in Msym_positions:
                        Msym_positions[monomial].append(pos)
                    else:
                        Msym_positions[monomial] = [pos]

    return Msym_positions




def coarse_grain_binary_POVM(meas_full):
    """
    Coarse-grain a d-dimensional POVM into a binary POVM by summing
    the first half of the effects for the first outcome.

    Parameters
    ----------
    meas_full : list of qutip.Qobj
        Original POVM effects, length d.

    Returns
    -------
    list of qutip.Qobj
        Binary POVM [E0, E1], where
        E0 = sum of first d/2 effects
        E1 = sum of remaining effects
    """
    d = len(meas_full)
    half = d // 2

    E0 = sum(meas_full[:half])
    E1 = sum(meas_full[half:])

    return [E0, E1]


def generate_noisy_projective_POVM(dim, pM):
    """
    Generate a noisy projective POVM:
        M_i = pM |i><i| + (1-pM)/dim * I
    for i = 0,...,dim-2 
    and the last element M_last = I - sum_i M_i.
    
    Returns a list of QuTiP Qobj operators.
    """

    # Identity
    id_qt = qt.qeye(dim)

    measS = []

    # Generate the first dim-1 POVM elements
    for i in range(dim-1):
        ket = qt.basis(dim, i)
        proj = ket * ket.dag()                      # |i><i|
        M_i = pM * proj + (1 - pM) * id_qt / dim    # noisy projector
        measS.append(M_i)

    # Compute the completeness leftover: M_last = I − Σ_i M_i
    M_sum = sum(measS)             # QuTiP handles this correctly
    M_last = id_qt - M_sum

    measS.append(M_last)           # Final POVM element

    return measS
