import numpy as np
import sympy as sp
import qutip as qt
from mosek.fusion import Model, Domain, Expr, ObjectiveSense
import sys
import chaospy

from utils_VN import *
# Re / Im of the moments are read as BLOCK AVERAGES of the unstructured 2n x 2n matrix
# (see utils_mosek_simple.py). Do NOT also star-import utils_mosek: its sympy2mosek_im
# reads a single block and would silently replace the averaged one.
from utils_mosek_simple import (setNumericalPrecisionForMosek, add_mosek,
                                re_entry, im_entry, sympy2mosek_re, sympy2mosek_im)

"""
Same SDP as Hierarchy_VN_entropy_fixed.py (lower bound on H(A|E), device-dependent scenario,
Gauss-Radau variational formula of Brown, Fawzi, Fawzi), but lighter for MOSEK.

Only difference with Hierarchy_VN_entropy_fixed.py
--------------------------------------------------
The complex Hermitian moment matrix Gamma = R + iI is represented by a real 2n x 2n PSD
matrix G. Hierarchy_VN_entropy_fixed.py forces G = [[R, -I], [I, R]] with 2 n^2 element-wise
equalities (G11 = G22, G21 = -G12). Here these equalities are REMOVED and R, I are defined as

        R := (G11 + G22) / 2          I := (G21 - G12) / 2

(re_entry / im_entry, used through sympy2mosek_re / sympy2mosek_im).

This is exact: R is symmetric, I antisymmetric, and with J = [[0, -1], [1, 0]]
(G + J^T G J)/2 = [[R, -I], [I, R]] >= 0, so Gamma >= 0. Every constraint and the objective
depend on G only through R and I, so any feasible G can be replaced by (G + J^T G J)/2,
which has the block structure, the same R, I and the same objective value. The optimal
value is the same as in Hierarchy_VN_entropy_fixed.py.

IMPORTANT: every constraint on gamma must go through sympy2mosek_re / sympy2mosek_im
(or re_entry / im_entry). Indexing gamma directly breaks the equivalence.

Everything else (generating set, tails, Z bound, objective, quadrature) is unchanged; see the
docstring of Hierarchy_VN_entropy_fixed.py for the differences with Hierarchy_VN_entropy.py.
"""


def monomials_exist(Msym_positions, x):
    """Return True if every monomial of the expanded expression x is an entry of the moment matrix."""
    return all(mon in Msym_positions for mon in x.expand().as_coefficients_dict())


def tail_is_supported(Msym_positions, measS, meas, dim, n_out, tail):
    """Return True if all the monomials needed by the projectivity constraints with this tail exist."""
    for k in range(dim):
        for l in range(dim):
            if not monomials_exist(Msym_positions, measS[k][l] * tail):
                return False
            for a1 in range(n_out-1):
                for i in range(dim):
                    for j in range(dim):
                        if not monomials_exist(Msym_positions, measS[k][l] * meas[a1][i][j] * tail):
                            return False
                        for a2 in range(n_out-1):
                            for l1 in range(dim):
                                if not monomials_exist(Msym_positions, measS[k][l] * meas[a1][i][l1] * meas[a2][l1][j] * tail):
                                    return False
    return True


def  add_projectivity_constraints(M, gamma, Msym_positions, measS, meas, dim, n_out, len_mon, tail):
    """
    Constraint \Tr[(\Pi_{SM}^a)^2 w \sigma] = \Tr[\Pi_{SM}^a w \sigma] for every explicit outcome a and for the
    last outcome \Pi^{n_out-1} = 1 - \sum_a \Pi^a, where w (tail) is a word in Eve's operators commuting with \Pi.
    With tail = 1 this is exactly the projectivity constraint of Hierarchy_VN_entropy.py.
    """
    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                for l in range(dim):
                    # Reset accumulators
                    projSMd, projSMd_2 = (None,) * 2
                    projSMd_im, projSMd_2im = (None,) * 2
                    for a1 in range(n_out-1):
                        projSM, projSM_im = (None,) * 2
                        projSM_2, projSM_2im = (
                            [None for _ in range(n_out - 1)],
                            [None for _ in range(n_out - 1)],
                        )

                        # Level 0 monomial   S_k,l * M_i,j * w
                        projSM = add_mosek(projSM,
                                        sympy2mosek_re(gamma, Msym_positions, measS[k][l] * meas[a1][i][j] * tail, len_mon))
                        projSM_im = add_mosek(projSM_im,
                                        sympy2mosek_im(gamma, Msym_positions, measS[k][l] * meas[a1][i][j] * tail, len_mon))

                        # Level 1 monomials:  S_k,l * M_i,l1 * M_l1,j * w
                        for a2 in range(n_out-1):
                            for l1 in range(dim):
                                projSM_2[a2] = add_mosek(projSM_2[a2],
                                                    sympy2mosek_re(gamma, Msym_positions,
                                                                   measS[k][l] *
                                                                   meas[a1][i][l1] *
                                                                   meas[a2][l1][j] * tail, len_mon))
                                projSM_2im[a2] = add_mosek(projSM_2im[a2],
                                                    sympy2mosek_im(gamma, Msym_positions,
                                                                   measS[k][l] *
                                                                   meas[a1][i][l1] *
                                                                   meas[a2][l1][j] * tail, len_mon))
                            projSMd_2 = add_mosek(projSMd_2, projSM_2[a2])
                            projSMd_2im = add_mosek(projSMd_2im, projSM_2im[a2])

                        projSMd_2 = add_mosek(projSMd_2,
                                                Expr.mul(-2, sympy2mosek_re(gamma, Msym_positions,
                                                            measS[k][l] *
                                                            meas[a1][i][j] * tail, len_mon)))
                        projSMd_2im = add_mosek(projSMd_2im,
                                                Expr.mul(-2, sympy2mosek_im(gamma, Msym_positions,
                                                            measS[k][l] *
                                                            meas[a1][i][j] * tail, len_mon)))

                        # (\Pi^a)^2 = \Pi^a for the explicit outcomes
                        M.constraint(Expr.sub(projSM, projSM_2[a1]), Domain.equalsTo(0))
                        M.constraint(Expr.sub(projSM_im, projSM_2im[a1]), Domain.equalsTo(0))

                        projSMd = add_mosek(projSMd, projSM)
                        projSMd_im = add_mosek(projSMd_im, projSM_im)

                    # Last outcome: (1 - \sum_a \Pi^a)^2 = 1 - \sum_a \Pi^a
                    if i == j:
                        projSMd_2 = Expr.add(projSMd_2, sympy2mosek_re(gamma, Msym_positions, measS[k][l] * tail, len_mon))
                        projSMd_2im = Expr.add(projSMd_2im, sympy2mosek_im(gamma, Msym_positions, measS[k][l] * tail, len_mon))

                    if i == j:
                        projSMd = Expr.sub(sympy2mosek_re(gamma, Msym_positions, measS[k][l] * tail, len_mon), projSMd)
                        projSMd_im = Expr.sub(sympy2mosek_im(gamma, Msym_positions, measS[k][l] * tail, len_mon), projSMd_im)
                    if i != j:
                        projSMd = Expr.mul(-1, projSMd)
                        projSMd_im = Expr.mul(-1, projSMd_im)

                    M.constraint(Expr.sub(projSMd, projSMd_2), Domain.equalsTo(0))
                    M.constraint(Expr.sub(projSMd_im, projSMd_2im), Domain.equalsTo(0))


def build_optimization_mosek_complex_VN(dim, n_out, t, rhoS, measS, meas, measS_fix, len_gen, Msym_positions, subs,
                                        opZ, opZ_adj, tails, Z_bound='trace', verbose=True):
    with Model('sdo2_simple') as M:

        len_mon = dim*len_gen

        # Free 2*len_mon x 2*len_mon PSD matrix: NO block-structure constraints.
        # Re and Im of the moments are read as block averages (re_entry / im_entry).
        gamma = M.variable(Domain.inPSDCone(2*len_mon))

        def Re(x):
            return sympy2mosek_re(gamma, Msym_positions, x, len_mon)

        def Im(x):
            return sympy2mosek_im(gamma, Msym_positions, x, len_mon)


        # Constraints relating different entries of the moment matrix that correspond to the same monomial
        # (same Re and same Im)
        for x, y in Msym_positions.items():
            if len(y) > 1:
                p0, q0 = y[0]
                for (p, q) in y[1:]:
                    # If (p, q) is the transpose of (p0, q0), Re is the same scalar by symmetry (0 = 0): skip it
                    if (p, q) != (q0, p0):
                        M.constraint(Expr.sub(re_entry(gamma, p, q, len_mon), re_entry(gamma, p0, q0, len_mon)),
                                     Domain.equalsTo(0))
                    M.constraint(Expr.sub(im_entry(gamma, p, q, len_mon), im_entry(gamma, p0, q0, len_mon)),
                                 Domain.equalsTo(0))

        # (The 2 len_mon^2 block-structure equalities of Hierarchy_VN_entropy_fixed.py are not imposed.)

        # Trace of the state
        trace_state, trace_state_im = (None,) * 2
        for l in range(dim):
            trace_state = add_mosek(trace_state, Re(measS[l][l]))
            trace_state_im = add_mosek(trace_state_im, Im(measS[l][l]))
        M.constraint(trace_state_im, Domain.equalsTo(0))


        # Projectivity of \Pi_{SM}^a, also multiplied by Eve's words w (they commute with \Pi_{SM}^a)
        n_tails = 0
        for tail in tails:
            if tail_is_supported(Msym_positions, measS, meas, dim, n_out, tail):
                add_projectivity_constraints(M, gamma, Msym_positions, measS, meas, dim, n_out, len_mon, tail)
                n_tails += 1
        print(f"Projectivity constraints imposed for {n_tails}/{len(tails)} tails")


        # Constraint \bra{i}\rho_{S}\ket{j}
        for i in range(dim):
            for j in range(dim):
                rho_ij = rhoS.full()[i, j]
                M.constraint(Expr.sub(Re(measS[j][i]), rho_ij.real), Domain.equalsTo(0))
                M.constraint(Expr.sub(Im(measS[j][i]), rho_ij.imag), Domain.equalsTo(0))

        # Constraint \bra{i}M_{S}^a\ket{j}
        for i in range(dim):
            for j in range(dim):
                measS_gamma_compl_r, measS_gamma_compl_im = (None,) * 2
                for a in range(n_out-1):
                    measS_gamma_r, measS_gamma_im = (None,) * 2
                    for l in range(dim):
                        measS_gamma_r = add_mosek(measS_gamma_r, Re(measS[l][l]*meas[a][i][j]))
                        measS_gamma_im = add_mosek(measS_gamma_im, Im(measS[l][l]*meas[a][i][j]))
                    mfix_ij = measS_fix[a].full()[i, j]
                    M.constraint(Expr.sub(measS_gamma_r, mfix_ij.real), Domain.equalsTo(0))
                    M.constraint(Expr.sub(measS_gamma_im, mfix_ij.imag), Domain.equalsTo(0))
                    measS_gamma_compl_r = add_mosek(measS_gamma_compl_r, measS_gamma_r)
                    measS_gamma_compl_im = add_mosek(measS_gamma_compl_im, measS_gamma_im)
                mfix_last_ij = measS_fix[n_out-1].full()[i, j]
                if i==j:
                    M.constraint(Expr.sub(Expr.sub(1,measS_gamma_compl_r), mfix_last_ij.real), Domain.equalsTo(0))
                if i!=j:
                    M.constraint(Expr.sub(Expr.mul(-1,measS_gamma_compl_r), mfix_last_ij.real), Domain.equalsTo(0))
                M.constraint(Expr.sub(Expr.mul(-1,measS_gamma_compl_im), mfix_last_ij.imag), Domain.equalsTo(0))


        # positivity of the (M_M^a)_ii
        for i in range(dim):
            for a in range(n_out-1):
                positive_meas_r, positive_meas_im = (None,) * 2
                for l in range(dim):
                    positive_meas_r = add_mosek(positive_meas_r, Re(measS[l][l]*meas[a][i][i]))
                    positive_meas_im = add_mosek(positive_meas_im, Im(measS[l][l]*meas[a][i][i]))
                M.constraint(positive_meas_r, Domain.greaterThan(0))
                M.constraint(positive_meas_im, Domain.equalsTo(0))


        # Constraint prob. distribution \Tr[\Pi_{SM}^a\sigma_{SM}]=\Tr[M_S^a\rho_{S}]
        prob_gamma_sum_r = None
        for a in range(n_out-1):
            prob_gamma_r, prob_gamma_im = (None,) * 2
            for i in range(dim):
                for j in range(dim):
                    prob_gamma_r = add_mosek(prob_gamma_r, Re(measS[i][j]*meas[a][i][j]))
                    prob_gamma_im = add_mosek(prob_gamma_im, Im(measS[i][j]*meas[a][i][j]))
            prob_gamma_sum_r = add_mosek(prob_gamma_sum_r, prob_gamma_r)

            p_a = complex((measS_fix[a] * rhoS).tr())
            M.constraint(Expr.sub(prob_gamma_r, p_a.real), Domain.equalsTo(0))
            M.constraint(Expr.sub(prob_gamma_im, p_a.imag), Domain.equalsTo(0))  # Actually this is zero
        p_last = complex((measS_fix[n_out-1] * rhoS).tr())
        M.constraint(Expr.sub(Expr.sub(1,prob_gamma_sum_r), p_last.real), Domain.equalsTo(0))


        ### Constraints over Z elements: ||Z|| <= alpha, imposed as the d x d PSD matrix
        ### \alpha^2 \rho_S - Tr_ME[(1 \otimes Z^* Z) \sigma] >= 0 (and the same with Z Z^*)   (Z_bound='psd')
        ### or only its trace \alpha^2 - Tr[(1 \otimes Z^* Z) \sigma] >= 0                     (Z_bound='trace')
        ### For a pure rho_S the 'psd' version has no strictly feasible point and MOSEK can stall.
        alpha = 1.5 * max(1.0 / t, 1.0 / (1.0 - t))

        if Z_bound == 'trace':
            for z, z_adj in zip(opZ, opZ_adj):
                loc_tr = None
                for i in range(dim):
                    loc_tr = add_mosek(loc_tr, Expr.sub(Expr.mul(alpha**2, Re(measS[i][i])),
                                                        Re(measS[i][i] * z_adj * z)))
                M.constraint(loc_tr, Domain.greaterThan(0))

        elif Z_bound:
            for z, z_adj in zip(opZ, opZ_adj):      # z runs over Z_a and Z_a^*, so both Z^*Z and Z Z^* are bounded
                # Small 2d x 2d matrix: its block structure is fixed entry by entry (cheap), as before
                gamma_loc = M.variable(Domain.inPSDCone(2*dim))
                for i in range(dim):
                    for j in range(dim):
                        loc_r = Expr.sub(Expr.mul(alpha**2, Re(measS[j][i])), Re(measS[j][i] * z_adj * z))
                        loc_im = Expr.sub(Expr.mul(alpha**2, Im(measS[j][i])), Im(measS[j][i] * z_adj * z))
                        # same real representation [[Re, -Im], [Im, Re]]
                        M.constraint(Expr.sub(gamma_loc.index(i, j), loc_r), Domain.equalsTo(0))
                        M.constraint(Expr.sub(gamma_loc.index(i + dim, j + dim), loc_r), Domain.equalsTo(0))
                        M.constraint(Expr.sub(gamma_loc.index(i + dim, j), loc_im), Domain.equalsTo(0))
                        M.constraint(Expr.add(gamma_loc.index(i, j + dim), loc_im), Domain.equalsTo(0))


        # Objective: \sum_a Tr[\sigma (\Pi^a \otimes (Z_a + Z_a^* + (1-t) Z_a^* Z_a) + t (1 \otimes Z_a Z_a^*))]
        # with \Pi^{n_out-1} = 1 - \sum_{a<n_out-1} \Pi^a. Only real parts: the imaginary parts cancel.
        obj = None

        for i in range(dim):
            for a in range(n_out-1):
                for j in range(dim):
                    obj = add_mosek(obj, Re(measS[i][j]*meas[a][i][j]*opZ[a]))
                    obj = Expr.sub(obj, Re(measS[i][j]*meas[a][i][j]*opZ[n_out-1]))
                    obj = Expr.add(obj, Re(measS[i][j]*meas[a][i][j]*opZ_adj[a]))
                    obj = Expr.sub(obj, Re(measS[i][j]*meas[a][i][j]*opZ_adj[n_out-1]))
                    obj = Expr.add(obj, Expr.mul(1-t, Re(measS[i][j]*meas[a][i][j]*opZ_adj[a]*opZ[a])))
                    obj = Expr.sub(obj, Expr.mul(1-t, Re(measS[i][j]*meas[a][i][j]*opZ_adj[n_out-1]*opZ[n_out-1])))
                obj = Expr.add(obj, Expr.mul(t, Re(measS[i][i]*opZ[a]*opZ_adj[a])))
            obj = Expr.add(obj, Re(measS[i][i]*opZ[n_out-1]))
            obj = Expr.add(obj, Re(measS[i][i]*opZ_adj[n_out-1]))
            obj = Expr.add(obj, Expr.mul(1-t, Re(measS[i][i]*opZ_adj[n_out-1]*opZ[n_out-1])))
            obj = Expr.add(obj, Expr.mul(t, Re(measS[i][i]*opZ[n_out-1]*opZ_adj[n_out-1])))

        # Solve
        M.objective(ObjectiveSense.Minimize,obj)
        if verbose:
            M.setLogHandler(sys.stdout) # print details of the optimization
        setNumericalPrecisionForMosek(M)
        M.solve()

        return M.primalObjValue(), gamma


def random_real_pure_state(dim):
    psi_vec = np.random.randn(2).astype(np.complex128)  # make array complex
    psi_vec[1] = psi_vec[1]+psi_vec[1]*1j/10
    psi_vec /= np.linalg.norm(psi_vec)

    psi = qt.Qobj(psi_vec.reshape((dim,1)))
    rho = psi * psi.dag()

    return rho


def generate_quadrature(m):
    """
    Generates the Gauss-Radau quadrature nodes t and weights w on [0,1] with the
    fixed node t = 1 (last element). Due to the way the package works it generates
    2*M nodes and weights.

        m    --    number of nodes in quadrature / 2
    """
    t, w = chaospy.quad_gauss_radau(m, chaospy.Uniform(0, 1), 1)
    t = t[0]
    return t, w


def generate_operators_Eve(dim, label):
    """
    Generate a single symbolic (non-Hermitian) operator of Eve with name label.
    """
    name = f"{label}"
    operators = nc.generate_operators(name, hermitian=False)[0]
    return operators


def extract_elements(op):
    """Flatten 2D lists / SymPy matrices into a 1D list of scalar symbols."""
    if isinstance(op, list):
        elements = []
        for item in op:
            if isinstance(item, list):
                elements.extend(item) # Handle 2D list
            else:
                elements.append(item) # Handle 1D list
        return elements
    elif hasattr(op, 'is_Matrix') and op.is_Matrix:
        return list(op) # Flattens a SymPy Matrix
    else:
        return [op] # It's already a single scalar operator


def compute_VN_entropy_lower_bound(dim, n_out, rhoS, measS_fix, m=4, Z_level=1, Z_bound='trace', verbose=True):
    """
    Lower bound on H(A|E) = c_m + \sum_{i<m} w_i/(t_i ln 2) * min_{Gamma}(...) (Lemma 2.3 of Brown-Fawzi-Fawzi).

    Z_level = 1: generating set {1, K, Z, Z^*, K Z, K Z^*}           ("local level 1")
    Z_level = 2: as above plus all products of two of Eve's operators (as in Hierarchy_VN_entropy.py)
    Z_bound = 'trace', 'psd' (True) or None/False: how the norm bound ||Z|| <= alpha is imposed
    verbose : print the MOSEK log of each SDP
    """

    # --- Step 1: Generate the single-site operators MS[i][j] ----------
    measS = generate_operators(dim, 'MS')

    # --- Step 2: Generate measurement operators for each POVM ----------
    meas = []
    meas_adj = []
    for a in range(n_out-1):
        label = f"M{a}_"                  # M0_, M1_, M2_, ...
        meas.append(generate_operators(dim, label))
        meas_adj.append(generate_operators(dim, label, adj=True))

    # --- Step 3: Eve's operators Z_a and their adjoints Z_a^* ----------
    opZ = []
    opZ_adj = []
    for a in range(n_out):
        opZ.append(generate_operators_Eve(dim, f"Z{a}"))
        opZ_adj.append(generate_operators_Eve(dim, f"Z{a}*"))
    for a in range(n_out):
        opZ.append(generate_operators_Eve(dim, f"Z{a}*"))
        opZ_adj.append(generate_operators_Eve(dim, f"Z{a}"))

    # Commutation rules: Z_elem * M_elem -> M_elem * Z_elem
    mon_subs = {}
    flat_meas = []
    for mm in (meas + meas_adj):
        flat_meas.extend(extract_elements(mm))
    flat_opZ = []
    for z in (opZ + opZ_adj):
        flat_opZ.extend(extract_elements(z))
    for z_sym in flat_opZ:
        for m_sym in flat_meas:
            mon_subs[z_sym * m_sym] = m_sym * z_sym

    # --- Step 4: Generating monomials ----------
    # Measurement operators K^{ij|a}
    alphabet = flatten(meas)
    alphabet_adj = flatten(meas_adj)
    generating_set_1      = generating_monomials(alphabet,      1, mon_subs)
    generating_set_adj_1  = generating_monomials(alphabet_adj,  1, mon_subs, adj=True)

    # Eve's operators Z_a, Z_a^*
    alphabet_Z = flatten([opZ, opZ_adj])
    alphabet_Z_adj = flatten([opZ_adj, opZ])
    generating_set_2      = generating_monomials(alphabet_Z,      Z_level, mon_subs)
    generating_set_adj_2  = generating_monomials(alphabet_Z_adj,  Z_level, mon_subs, adj=True)

    # Mixed monomials K^{ij|a} Z, whose adjoint is (K^{ij|a})^\dagger Z^* (K and Z commute)
    generating_set_3 = []
    generating_set_adj_3 = []
    for a in range(n_out-1):
        for i in range(dim):
            for j in range(dim):
                for z, z_adj in zip(opZ, opZ_adj):
                    generating_set_3.append(meas[a][i][j] * z)
                    generating_set_adj_3.append(meas_adj[a][i][j] * z_adj)

    # Remove duplicates (e.g. the identity appears in both sets) keeping generators and adjoints aligned
    pairs = list(dict.fromkeys(zip(flatten([generating_set_1, generating_set_2, generating_set_3]),
                                   flatten([generating_set_adj_1, generating_set_adj_2, generating_set_adj_3]))))
    generating_set = [g for g, _ in pairs]
    generating_set_adj = [g_adj for _, g_adj in pairs]

    print("Generating set:")
    print(generating_set)

    print("Generating set adjoint:")
    print(generating_set_adj)

    # --- Step 5: Build the moment matrix positions ----------
    Msym_positions = calculate_moment_matrix(
        measS,
        generating_set,
        generating_set_adj,
        dim,
        mon_subs
    )

    # Eve's words used as tails in the projectivity constraints: 1, Z, Z Z'
    Z_unique = list(dict.fromkeys(opZ))
    tails = [sp.S.One] + Z_unique + [z1 * z2 for z1 in Z_unique for z2 in Z_unique]

    # --- Step 6: Gauss-Radau quadrature and SDPs ----------
    t,w = generate_quadrature(m)
    print('t', t)
    print('w', w)

    c_m = sum(w_l / (t_l * np.log(2)) for t_l, w_l in zip(t[:-1], w[:-1]))

    entropy = 0
    for i in range(0,len(t)-1):
        c = w[i]/(t[i]*np.log(2))
        obj_val, _ = build_optimization_mosek_complex_VN(dim, n_out, t[i], rhoS, measS, meas, measS_fix,
                                                         len(generating_set), Msym_positions, mon_subs,
                                                         opZ, opZ_adj, tails, Z_bound, verbose)
        print(f"t = {t[i]:.6f}: SDP value = {obj_val:.8f}")
        entropy += c*obj_val
    entropy += c_m

    return entropy

'''
if __name__ == "__main__":

    # Problem setup
    dim = 2  # Hilbert space dimension
    n_out = 2

    # Define |0> and |1>
    basis = [qt.basis(dim, i) for i in range(dim)]

    # Define the parameter
    pM = 0.75 # You can set this to any value between 0 and 1
    pS = 0.8

    # Define |+> state and projector
    plus = sum(basis).unit()              # |+> = (|0>+|1>+...)/sqrt(dim)
    proj_plus = plus * plus.dag()
    id_qt = qt.qeye(dim)

    rhoS = pS * proj_plus + (1 - pS) * id_qt / dim

    # Measurement M0, M1 (generalisation of |0><0| + noise)
    measS_fix = generate_noisy_projective_POVM(dim, pM)
    print(measS_fix)

    H = compute_VN_entropy_lower_bound(dim, n_out, rhoS, measS_fix, m=4, Z_level=3, Z_bound='trace')
    print("Conditional von Neumann entropy H(A|E) is lower bounded by:", H)
'''

if __name__ == "__main__":

    # Problem setup
    dim = 2  # Hilbert space dimension
    n_out = 2

    # Define |0> and |1>
    basis = [qt.basis(dim, i) for i in range(dim)]

    # Define the parameter
    m1 = 0.75 # You can set this to any value between 0 and 1
    m2 = 1-m1
    pS = 1.0

    # Define |+> state and projector
    plus = sum(basis).unit()              # |+> = (|0>+|1>+...)/sqrt(dim)
    proj_plus = plus * plus.dag()
    id_qt = qt.qeye(dim)

    rhoS = pS * proj_plus + (1 - pS) * id_qt / dim

    # Measurement M0, M1 (generalisation of |0><0| + noise)
    measS_fix = generate_two_outcome_POVM(m1, m2, dim)
    print(measS_fix)

    H = compute_VN_entropy_lower_bound(dim, n_out, rhoS, measS_fix, m=4, Z_level=3, Z_bound='trace')
    print("Conditional von Neumann entropy H(A|E) is lower bounded by:", H)