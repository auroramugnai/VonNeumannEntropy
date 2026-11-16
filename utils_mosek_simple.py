"""
utils_mosek_simple.py

Lighter version of build_optimization_mosek_complex (utils_mosek.py) for the
complex-valued SDP that upper-bounds the guessing probability.

Difference with utils_mosek.py
------------------------------
utils_mosek.py represents each complex Hermitian moment matrix Gamma = R + iI by a
real 2n x 2n PSD variable G and FORCES it to have the block structure

        G = [ R  -I ]
            [ I   R ]

with 2 n^2 element-wise equalities per matrix (G11 = G22 and G21 = -G12).
For dim = 4, n_out = 4 these are ~77,000 equalities per matrix, which make the
interior-point Schur complement ~25 GB per matrix.

Here those equalities are DROPPED. G is a free real symmetric PSD matrix and the
real and imaginary parts of Gamma are DEFINED as block averages:

        R := (G11 + G22) / 2          I := (G21 - G12) / 2

Why this is exact
-----------------
* R is symmetric and I antisymmetric by construction, so Gamma is Hermitian.
* With J = [[0, -1], [1, 0]]:   (G + J^T G J) / 2 = [[R, -I], [I, R]].
  G >= 0 implies J^T G J >= 0, hence [[R, -I], [I, R]] >= 0, i.e. Gamma >= 0.
* Every constraint and the objective below depend on G ONLY through R and I.
  So any feasible G can be replaced by (G + J^T G J)/2, which has the block
  structure, the same R and I, and the same objective value. Conversely every
  structured G is feasible here. The optimal value is therefore identical to
  the one of utils_mosek.build_optimization_mosek_complex.

IMPORTANT: if you add a constraint, write it through re_entry / im_entry
(or sympy2mosek_re / sympy2mosek_im), never by indexing G directly, otherwise
the equivalence above is lost.

Usage
-----
Drop-in replacement: same function name and arguments as in utils_mosek.py.

    from utils_mosek_simple import build_optimization_mosek_complex
"""

import sys
from mosek.fusion import Model, Domain, Expr, ObjectiveSense


def setNumericalPrecisionForMosek(M):
    """Tighten MOSEK interior-point tolerances beyond the default."""
    M.setSolverParam("intpntTolRelGap", 1e-10)
    M.setSolverParam("intpntTolPfeas",  1e-10)
    M.setSolverParam("intpntTolDfeas",  1e-10)
    M.setSolverParam("intpntTolMuRed",  1e-14)


def add_mosek(acc, term):
    """
    Accumulate a MOSEK Fusion expression without an integer-0 seed.
    Returns term if acc is None, else acc + term.
    """
    return term if acc is None else Expr.add(acc, term)


# =============================================================================
# READING Re AND Im OF Gamma FROM THE UNSTRUCTURED 2n x 2n VARIABLE G
# =============================================================================

def re_entry(gamma, p, q, n):
    """Re Gamma[p, q] = ( G[p, q] + G[p+n, q+n] ) / 2."""
    return Expr.mul(0.5, Expr.add(gamma.index(p, q), gamma.index(p + n, q + n)))


def im_entry(gamma, p, q, n):
    """Im Gamma[p, q] = ( G[p+n, q] - G[p, q+n] ) / 2."""
    return Expr.mul(0.5, Expr.sub(gamma.index(p + n, q), gamma.index(p, q + n)))


def sympy2mosek_re(gamma, Msym_positions, x, n):
    """
    Real part of the moment of the SymPy expression x.

    Expands x into coeff * monomial terms, looks up the first position (p, q)
    of each monomial in Msym_positions and returns  sum coeff * Re Gamma[p, q].
    """
    y = None
    for mon, coeff in x.expand().as_coefficients_dict().items():
        p, q = Msym_positions[mon][0]
        y = add_mosek(y, Expr.mul(float(coeff), re_entry(gamma, p, q, n)))
    return y


def sympy2mosek_im(gamma, Msym_positions, x, n):
    """Imaginary part of the moment of x:  sum coeff * Im Gamma[p, q]."""
    y = None
    for mon, coeff in x.expand().as_coefficients_dict().items():
        p, q = Msym_positions[mon][0]
        y = add_mosek(y, Expr.mul(float(coeff), im_entry(gamma, p, q, n)))
    return y


# =============================================================================
# COMPLEX-VALUED SDP WITHOUT BLOCK-STRUCTURE CONSTRAINTS
# =============================================================================

def build_optimization_mosek_complex(dim, n_out, rhoS, measS, meas, measS_fix,
                                     len_gen, Msym_positions, PPT=False,
                                     verbose=True):
    """
    Solve the complex-valued SDP and return an upper bound on the guessing probability.

    Same constraints and objective as utils_mosek.build_optimization_mosek_complex,
    without the 2 n^2 block-structure equalities per moment matrix (see module docstring).

    Parameters
    ----------
    dim : int
        Hilbert space dimension.
    n_out : int
        Number of measurement outcomes.
    rhoS : qutip.Qobj
        System state.
    measS : list[list]
        Symbolic operator table for M_S.
    meas : list[list[list]]
        Symbolic operator tables for each non-trivial POVM element.
    measS_fix : list of qutip.Qobj
        Numerical reference POVM.
    len_gen : int
        Number of generating monomials.
    Msym_positions : dict
        Monomial -> position lookup.
    PPT : bool
        If True, add a positive partial transpose constraint on Gamma = sum_e Gamma^e.
    verbose : bool
        If True, print the MOSEK log.

    Returns
    -------
    (float, list)
        Guessing probability upper bound and list of gamma MOSEK variables.
    """
    with Model('sdo2_simple') as M:

        n = dim * len_gen   # size of the complex moment matrix Gamma^e

        # G^e: free 2n x 2n real symmetric PSD matrix. Gamma^e = R^e + i I^e is read
        # from it through re_entry / im_entry (block averages).
        gamma = [M.variable(Domain.inPSDCone(2 * n)) for e in range(n_out)]

        # Shorthands
        def Re(e, x):
            return sympy2mosek_re(gamma[e], Msym_positions, x, n)

        def Im(e, x):
            return sympy2mosek_im(gamma[e], Msym_positions, x, n)

        # ------------------------------------------------------------------
        # Moment matrix consistency: all positions sharing the same monomial
        # have the same Re and the same Im.
        # ------------------------------------------------------------------
        for e in range(n_out):
            for _, positions in Msym_positions.items():
                if len(positions) > 1:
                    p0, q0 = positions[0]
                    for (p, q) in positions[1:]:
                        # If (p, q) is the transpose of (p0, q0), Re is the same
                        # scalar by symmetry: the real constraint is 0 = 0, skip it.
                        if (p, q) != (q0, p0):
                            M.constraint(
                                Expr.sub(re_entry(gamma[e], p, q, n),
                                         re_entry(gamma[e], p0, q0, n)),
                                Domain.equalsTo(0)
                            )
                        M.constraint(
                            Expr.sub(im_entry(gamma[e], p, q, n),
                                     im_entry(gamma[e], p0, q0, n)),
                            Domain.equalsTo(0)
                        )

        # ------------------------------------------------------------------
        # Optional PPT constraint on Gamma = sum_e Gamma^e.
        #
        # Partial transpose on the S index (block indices k <-> l):
        #     Gamma^{T_S}[(l,i), (k,j)] = Gamma[(k,i), (l,j)]
        # imposed on Re and Im. Only the upper triangle is imposed: R is symmetric
        # and I antisymmetric on both sides, so the lower triangle is the same
        # equation, and the diagonal of Im is identically zero.
        # (utils_mosek imposes the complex conjugate of this matrix; it has the
        # same spectrum, so the PPT condition is the same.)
        # ------------------------------------------------------------------
        if PPT:
            gamma_pt = M.variable(Domain.inPSDCone(2 * n))
            for k in range(dim):
                for l in range(dim):
                    for i in range(len_gen):
                        for j in range(len_gen):
                            p_pt, q_pt = l * len_gen + i, k * len_gen + j
                            if p_pt > q_pt:
                                continue
                            p, q = k * len_gen + i, l * len_gen + j

                            re_sum, im_sum = None, None
                            for e in range(n_out):
                                re_sum = add_mosek(re_sum, re_entry(gamma[e], p, q, n))
                                im_sum = add_mosek(im_sum, im_entry(gamma[e], p, q, n))

                            M.constraint(
                                Expr.sub(re_entry(gamma_pt, p_pt, q_pt, n), re_sum),
                                Domain.equalsTo(0)
                            )
                            if p_pt != q_pt:
                                M.constraint(
                                    Expr.sub(im_entry(gamma_pt, p_pt, q_pt, n), im_sum),
                                    Domain.equalsTo(0)
                                )

        # ------------------------------------------------------------------
        # Trace of unnormalised adversary states. Im part must vanish.
        # ------------------------------------------------------------------
        trace_states    = [None for _ in range(n_out)]
        trace_states_im = [None for _ in range(n_out)]
        for e in range(n_out):
            for l in range(dim):
                trace_states[e]    = add_mosek(trace_states[e],    Re(e, measS[l][l]))
                trace_states_im[e] = add_mosek(trace_states_im[e], Im(e, measS[l][l]))
            M.constraint(trace_states_im[e], Domain.equalsTo(0))

        # ------------------------------------------------------------------
        # Projector / idempotency constraints (Re and Im).
        # For each outcome a1: (M^a1)^2 = M^a1 element-wise.
        # Also for M^last = I - sum_a M^a.
        # ------------------------------------------------------------------
        for e in range(n_out):
            for i in range(dim):
                for j in range(dim):
                    for k in range(dim):
                        for l in range(dim):
                            projSMd,    projSMd_2    = None, None
                            projSMd_im, projSMd_2im  = None, None

                            for a1 in range(n_out - 1):
                                projSM_2   = [None for _ in range(n_out - 1)]
                                projSM_2im = [None for _ in range(n_out - 1)]

                                # Degree-1: Tr[|k><l| M^a1_{ij} sigma^e]
                                mon1 = measS[k][l] * meas[a1][i][j]
                                projSM    = Re(e, mon1)
                                projSM_im = Im(e, mon1)

                                # Degree-2: Tr[|k><l| M^a1_{i,l1} M^a2_{l1,j} sigma^e]
                                for a2 in range(n_out - 1):
                                    for l1 in range(dim):
                                        mon2 = measS[k][l] * meas[a1][i][l1] * meas[a2][l1][j]
                                        projSM_2[a2]   = add_mosek(projSM_2[a2],   Re(e, mon2))
                                        projSM_2im[a2] = add_mosek(projSM_2im[a2], Im(e, mon2))
                                    projSMd_2   = add_mosek(projSMd_2,   projSM_2[a2])
                                    projSMd_2im = add_mosek(projSMd_2im, projSM_2im[a2])

                                # Subtract 2 * degree-1 to form sum M^a1 M^a2 - 2 sum M^a1
                                projSMd_2   = add_mosek(projSMd_2,   Expr.mul(-2, Re(e, mon1)))
                                projSMd_2im = add_mosek(projSMd_2im, Expr.mul(-2, Im(e, mon1)))

                                # Idempotency for outcome a1: M^a1 = (M^a1)^2
                                M.constraint(Expr.sub(projSM,    projSM_2[a1]),   Domain.equalsTo(0))
                                M.constraint(Expr.sub(projSM_im, projSM_2im[a1]), Domain.equalsTo(0))

                                projSMd    = add_mosek(projSMd,    projSM)
                                projSMd_im = add_mosek(projSMd_im, projSM_im)

                            if i == j:
                                # Identity term completing (M^last)^2
                                projSMd_2   = Expr.add(projSMd_2,   Re(e, measS[k][l]))
                                projSMd_2im = Expr.add(projSMd_2im, Im(e, measS[k][l]))
                                # projSMd -> Tr[|k><l| M^last sigma^e]
                                projSMd    = Expr.sub(Re(e, measS[k][l]), projSMd)
                                projSMd_im = Expr.sub(Im(e, measS[k][l]), projSMd_im)
                            else:
                                projSMd    = Expr.mul(-1, projSMd)
                                projSMd_im = Expr.mul(-1, projSMd_im)

                            # Idempotency for the last outcome: M^last = (M^last)^2
                            M.constraint(Expr.sub(projSMd,    projSMd_2),   Domain.equalsTo(0))
                            M.constraint(Expr.sub(projSMd_im, projSMd_2im), Domain.equalsTo(0))

        # ------------------------------------------------------------------
        # State constraint (Re and Im):  <i|rhoS|j> = sum_e Tr[|i><j| sigma^e]
        # ------------------------------------------------------------------
        for i in range(dim):
            for j in range(dim):
                rhoS_gamma_r, rhoS_gamma_im = None, None
                for e in range(n_out):
                    rhoS_gamma_r  = add_mosek(rhoS_gamma_r,  Re(e, measS[j][i]))
                    rhoS_gamma_im = add_mosek(rhoS_gamma_im, Im(e, measS[j][i]))
                rho_ij = rhoS.full()[i, j]
                M.constraint(Expr.sub(rhoS_gamma_r,  rho_ij.real), Domain.equalsTo(0))
                M.constraint(Expr.sub(rhoS_gamma_im, rho_ij.imag), Domain.equalsTo(0))

        # ------------------------------------------------------------------
        # Measurement constraint (Re and Im):
        #     <i|M_S^a|j> = sum_e Tr[M^a_{ij} sigma^e]
        # Last POVM element via completeness: M^last = I - sum_a M^a.
        # ------------------------------------------------------------------
        for i in range(dim):
            for j in range(dim):
                measS_gamma_compl_r, measS_gamma_compl_im = None, None
                for a in range(n_out - 1):
                    measS_gamma_r, measS_gamma_im = None, None
                    for e in range(n_out):
                        for l in range(dim):
                            mon = measS[l][l] * meas[a][i][j]
                            measS_gamma_r  = add_mosek(measS_gamma_r,  Re(e, mon))
                            measS_gamma_im = add_mosek(measS_gamma_im, Im(e, mon))
                    mfix_ij = measS_fix[a].full()[i, j]
                    M.constraint(Expr.sub(measS_gamma_r,  mfix_ij.real), Domain.equalsTo(0))
                    M.constraint(Expr.sub(measS_gamma_im, mfix_ij.imag), Domain.equalsTo(0))
                    measS_gamma_compl_r  = add_mosek(measS_gamma_compl_r,  measS_gamma_r)
                    measS_gamma_compl_im = add_mosek(measS_gamma_compl_im, measS_gamma_im)

                mfix_last_ij = measS_fix[n_out - 1].full()[i, j]
                if i == j:
                    M.constraint(
                        Expr.sub(Expr.sub(1, measS_gamma_compl_r), mfix_last_ij.real),
                        Domain.equalsTo(0)
                    )
                else:
                    M.constraint(
                        Expr.sub(Expr.mul(-1, measS_gamma_compl_r), mfix_last_ij.real),
                        Domain.equalsTo(0)
                    )
                M.constraint(
                    Expr.sub(Expr.mul(-1, measS_gamma_compl_im), mfix_last_ij.imag),
                    Domain.equalsTo(0)
                )

        # ------------------------------------------------------------------
        # Positivity of diagonal POVM elements: (M^a)_{ii} >= 0
        # NOTE: kept identical to utils_mosek.py, including Im >= 0. Since
        # (M^a)_{ii} is real, Im = 0 (or no constraint) would be the natural choice.
        # ------------------------------------------------------------------
        for e in range(n_out):
            for i in range(dim):
                for a in range(n_out - 1):
                    positive_meas_r, positive_meas_im = None, None
                    for l in range(dim):
                        mon = measS[l][l] * meas[a][i][i]
                        positive_meas_r  = add_mosek(positive_meas_r,  Re(e, mon))
                        positive_meas_im = add_mosek(positive_meas_im, Im(e, mon))
                    M.constraint(positive_meas_r,  Domain.greaterThan(0))
                    M.constraint(positive_meas_im, Domain.greaterThan(0))

        # ------------------------------------------------------------------
        # Probability distribution constraint:
        #     Tr[M_S^a rhoS] = Tr[Pi^a sum_e sigma^e]
        # ------------------------------------------------------------------
        prob_gamma_sum_r = None
        for a in range(n_out - 1):
            prob_gamma_r, prob_gamma_im = None, None
            for e in range(n_out):
                for i in range(dim):
                    for j in range(dim):
                        mon = measS[i][j] * meas[a][i][j]
                        prob_gamma_r  = add_mosek(prob_gamma_r,  Re(e, mon))
                        prob_gamma_im = add_mosek(prob_gamma_im, Im(e, mon))
            prob_gamma_sum_r = add_mosek(prob_gamma_sum_r, prob_gamma_r)
            p_a = (measS_fix[a] * rhoS).tr()
            M.constraint(Expr.sub(prob_gamma_r,  complex(p_a).real), Domain.equalsTo(0))
            M.constraint(Expr.sub(prob_gamma_im, complex(p_a).imag), Domain.equalsTo(0))
        p_last = (measS_fix[n_out - 1] * rhoS).tr()
        M.constraint(
            Expr.sub(Expr.sub(1, prob_gamma_sum_r), complex(p_last).real),
            Domain.equalsTo(0)
        )

        # ------------------------------------------------------------------
        # Objective: guessing probability
        #     sum_{a < last} Tr[Pi^a sigma^a] + Tr[Pi^last sigma^last]
        # NOTE: kept identical to utils_mosek.py, which also adds the Im parts.
        # ------------------------------------------------------------------
        obj = None
        for a in range(n_out - 1):
            for i in range(dim):
                for j in range(dim):
                    mon = measS[i][j] * meas[a][i][j]
                    obj = add_mosek(obj, Expr.sub(Re(a, mon), Re(n_out - 1, mon)))
                    obj = add_mosek(obj, Expr.sub(Im(a, mon), Im(n_out - 1, mon)))
        obj = Expr.add(obj, trace_states[n_out - 1])

        M.objective(ObjectiveSense.Maximize, obj)
        if verbose:
            M.setLogHandler(sys.stdout)   # print the MOSEK log
        setNumericalPrecisionForMosek(M)
        M.solve()

        return M.primalObjValue(), gamma
