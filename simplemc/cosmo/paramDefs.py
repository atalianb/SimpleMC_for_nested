##
# This file has parameter definitions for all
# parameters used in this code.
##
# Change here for bounds, or import and rewrite.
##
##
# The parameter class is defined as
# Parameter(name, value, err=0.0, bounds=None, Ltxname=None)


from simplemc.cosmo.Parameter import Parameter
import numpy as np

# Parameters are value, variation, bounds.
Om_par   = Parameter("Om",   0.3038,  0.05,    (0.1, 0.5),   "\Omega_m")
Obh2_par = Parameter("Obh2", 0.02234, 0.001, (0.02, 0.025), "\Omega_{b}h^2")
h_par    = Parameter("h",    0.6821,  0.05,   (0.4, 0.9),    "h")

mnu_par  = Parameter("mnu",  0.06,    0.1,    (0, 1.0),      "\Sigma m_{\\nu}")
Nnu_par  = Parameter("Nnu",  3.046,   0.5,    (3.046, 5.046),"N_{\\rm eff}")

Ok_par = Parameter("Ok", 0.0, 0.01, (-0.02, 0.02), "\Omega_k")
w_par  = Parameter("w", -1., 0.1, (-2.0, 0.0), "w_0")
wa_par = Parameter("wa", 0.0, 0.1, (-2.0, 2.0), "w_a")
wb_par = Parameter("wb", 0.7, 0.2,  (-2., 3.0), "w_b")
wc_par = Parameter("wc", 0.7, 0.2,   (-3., 5.0), "w_c")

s8_par    = Parameter("s8", 0.8, 0.01, (0.5, 1.0), "s8")

# This is the prefactor parameter c/rdH0.
Pr_par = Parameter("Pr", 28.6, 4, (5, 70), "c/(H_0r_d)")

# Poly Cosmology Parameters.
Om1_par = Parameter("Om1", 0.0, 0.7, (-3, 3), "\Omega_1")
Om2_par = Parameter("Om2", 0.0, 0.7, (-3, 3), "\Omega_2")

# JordiCDM Cosmology Parameters.
q_par  = Parameter("q",  0.0, 0.2, (0, 1),      "q")
za_par = Parameter("za", 3,   1.0, (2, 10),     "z_a")
zb_par = Parameter("zb", 1,   0.5, (0, 2),      "z_b")
wd_par = Parameter("wd", -1,  0.5, (-2.0, 0.0), "w_d")
Od_par = Parameter("Od", 0.0, 0.2, (0.0, 1.0),  "O_d")

# Weird model.
mu_par  = Parameter("mu",  2.13,    0.1,  (0.01, 5), "\mu")
Amp_par = Parameter("Amp", -0.1839, 0.01, (-2, 2),   "Amp")
sig_par = Parameter("sig", 0.1,     0.2,  (0, 3.0),  "\sig")

# Tired Light.
beta_par = Parameter("beta", 1, 0.1, (-1, 1), "\\beta")

# Spline reconstruction
Sp1_par = Parameter("Sp1", 1, 0.01, (-1.2, 1.2), "S1")
Sp2_par = Parameter("Sp2", 1, 0.01, (-1.2, 1.2), "S2")
Sp3_par = Parameter("Sp3", 1, 0.01, (-1.2, 1.2), "S3")
Sp4_par = Parameter("Sp4", 1, 0.01, (-1.2, 1.2), "S4")


# Step-wise dark energy density (edit: JG)
# number of redshift boundaries (fixed).
step_nz_par = Parameter("StepNZ", 3, 1, (0., 10.), "nz")
# redshift of bin boundaries (fixed)
step_z0_par = Parameter("StepZ0", 0.5, 0.01, (0., 10.), "z0")
step_z1_par = Parameter("StepZ1", 1.0, 0.01, (0., 10.), "z1")
step_z2_par = Parameter("StepZ2", 1.6, 0.01, (0., 10.), "z2")
step_z3_par = None
step_z4_par = None

# rho_DE/rho_c in redshift bins (nz+1, free), 0.2 is working.
step_rho0_par = Parameter("StepR0", 0.7, 0.5, (-20., 20.), "\\rho_0")
step_rho1_par = Parameter("StepR1", 0.7, 0.5, (-20., 20.), "\\rho_1")
step_rho2_par = Parameter("StepR2", 0.7, 0.5, (-20., 20.), "\\rho_2")
step_rho3_par = Parameter("StepR3", 0.7, 0.5, (-20., 20.), "\\rho_3")
step_rho4_par = None
step_rho5_par = None

# Decaying Dark Matter.
lambda_par = Parameter("lambda", 1.0, 1.0, (0., 20.0), "\lambda")
xfrac_par  = Parameter("xfrac",  0.1, 0.1, (0.0, 1.0), "f_x")

# Early Dark Energy.
Ode_par = Parameter("Ode", 0.05, 0.01, (0, 0.9), "\Omega^e_{\\rm de}")

# Slow-Roll DE.
dw_par = Parameter("dw", 0, 0.1, (-1.0, 1.0), "\delta w_0")

# EarlyDE model.
lam_par = Parameter("lam", 1,  0.5, (0, 5),      "\lambda")
V0_par  = Parameter("V0",  1,  1,   (0, 30),     "V_0")
A_par   = Parameter("A",   10,  1,  (-20, 20),   "A")
B_par   = Parameter("B",   -20, 2,  (-100, 100), "B")

# wDark Matter.
wDM_par = Parameter("wDM", 0.0, 0.1, (-1.0, 1.0), "w_{DM}")

# Generic models.
a_par = Parameter("a", 0., 0.5, (-10., 10.), "a" )
b_par = Parameter("b", 0., 0.5, (-10., 10.), "b")


# Compress data
# where the first bin that is fix.
Nbins = 15
step  = (13-1.3)/(Nbins-1)
zbin_par = [Parameter("zbin%d"%i, 1.3+step*i, 0.3, (step*i, 3+step*i), "zbin%d"%i) for i in range(Nbins)]


# Quintess Cosmology.
mphi_par   = Parameter("mphi", 0.2, 0.1, (-5, 1), "m_{\phi}")

# PhiRatra scalar field model.
ralpha_par  = Parameter("alpha", 0.01, 0.005, (0.001, 0.05), "\\alpha")

# Logarithmized Energy momentum-tensor.
alpha_par  = Parameter("alpha", 0., 0.01, (-1, 1), "alpha")

# Quintom Cosology.
mquin_par = Parameter("mquin", 1.7, 0.2, (0, 4.0), "m_{\phi}")
mphan_par = Parameter("mphan", 0.8, 0.2, (0, 3.0), "m_{\psi}")
iniphi_par = Parameter("iniphi", 0.5, 0.1, (0, 1.5), "\phi_0")
coupling_par = Parameter("beta",  1.0, 0.5, (-10, 10), "\\beta")

# Fourier series expansion for w(z).
a0_par = Parameter("a0", -2., 0.1, (-3.0, -1.0), "a_0")
a1_par = Parameter("a1", 0.0, 0.1, (-1.5, 1.5), "a_1")
b1_par = Parameter("b1", 0.0, 0.1, (-1.5, 1.5), "b_1")
a2_par = Parameter("a2", 0.0, 0.1, (-1.5, 1.5), "a_2")
b2_par = Parameter("b2", 0.0, 0.1, (-1.5, 1.5), "b_2")

# Anisotropic dark energy.
bd_par     = Parameter("bd", 2.0, 0.5, (0, 4), "\omega")
Osig_par   = Parameter("Osig", -9., 0.5, (-12, 0), "\Omega_{\sigma}")

# Cosine Parameterisation for deceleration parameter.
nk_par = Parameter("nk", 0.0, 0.1, (-1, 1), "n")
kk_par = Parameter("kk", 1.5, 0.1, (0.01, 3.14), "k")

# Graduated DE.
ggama_par     = Parameter("ggama", -1.0, 0.1, (-1.3, -0.7), "\gamma")
glambda_par   = Parameter("glambda", 0, 0.2, (-10, 0), "\lambda")

# Phi Cosmology, e-Scalar field.
phialp_par  = Parameter("phialp",   1.0, 0.1,  (-3, 3.), "\\alpha")
philam_par  = Parameter("philam",   0.5, 0.01, (-2.0, 2.0), "\\lambda_i")
phibeta_par = Parameter("phibeta",  0.1, 0.05, (-3.0, 3.0), "\\beta")
phimu_par   = Parameter("phimu",    1.0, 0.05, (-4.0, 4.0), "\\mu")

# IBEG Cosmology Parameters, bose-einstein.
Oi0_par= Parameter("Oi0", -1.0, 0.2, (-5.0, 2.0), "\Omega_{io}")
xx_par= Parameter("xx", 0.99, 0.1, (0.5,1.5), "x")

# QDGP, Quintessence+ DGP model.
Oq_par = Parameter("Oq",  0.7, 0.05, (0.5, 1.0), "\Omega_q")
wq_par = Parameter("wq",  -0.9, 0.05, (-1.0, -0.5), "w_q")

# Rotation curves, for astrophysical purposes.
#Anfw_par = Parameter("Anfw",  0.1, 0.01, (0.0, 5.5), "A_s")
#rs_par =   Parameter("rs",  250., 10.0, (0.0, 350.0), "r_s")
#Anfw->m_a,rs->eps0
Anfw_par = Parameter("Anfw",  np.log10(6.81e-24), 0.01, (np.log10(1.0e-26), np.log10(1.0e-20)), "A_s")
rs_par =   Parameter("rs",  np.log10(3.69e-4), 0.01, (np.log10(1.0e-6), np.log10(1.0e-2)), "r_s")
phi0_par =   Parameter("phi0",  np.log10(1.0), 0.01, (np.log10(0.00001), np.log10(2.0)), "phi0")
phi1_par =   Parameter("phi1",  np.log10(2.02e-4), 0.01, (np.log10(0.000001), np.log10(2.0)), "phi1")
phi2_par =   Parameter("phi2",  np.log10(1.25e-1), 0.01, (np.log10(0.000001), np.log10(2.0)), "phi2")
# Restrained DE.
weff_par      = Parameter("weff", -1.0, 0.02, (-1., 0.), "w_{eff}")
wcpl_par      = Parameter("wcpl", 0.0, 0.03, (0, 0.5), "w_{cpl}")


LMBD_par      = Parameter("LMBD", 0.65, 0.02, (0, 1), "w_{cpl}")


