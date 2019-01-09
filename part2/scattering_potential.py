# PART 2: SCATTERING


#This program is based on scattering_potential.py

# SCAT1

# ---------------------------------------------------------------
# Modification of the program scattering_solution.py from
# the exercise sessions, used to calculate the scattering cross section
# for two particals colliding by means of the central Morse potential.
# ---------------------------------------------------------------

import math as m
import numpy as np
import scipy.special
import matplotlib.pyplot as pl


# GLOBAL PARAMETERS AND UNITS
#
# This program works with rescaled energies, E' and V'
# which relate to physical energies (E, V) as
# E' = (2 * m / hbar ** 2) * E
# V' = (2 * m / hbar ** 2) * V,
# where m is the reduced mass of the two-particle system
# We choose the parameter Re (the location of the maximum in the Morse potential)
# as the unit of length.
# This means that energies are expressed in units 1 / fm **2
#
# This version of the program works in these rescaled units, it can however be
# easily modified to work in SI units.
#
# To apply this program to a physical problem, i.e. work with
# actual physical energies (expressed in eV) one simply has to choose the relevant values of
# m an fm, and express the quantity 2 * m / hbar ** 2 in terms of eV and fm as
# 2 * m / hbar ** 2 = Scaling * 1 / (eV * fm ** 2)
# Physical energies can then be used as input by assigning this value to the variable Scaling
# 
#
#
# parameters of the Morse potential: These are the parameters which are used in the paper attached to the assignment.
# 

V0 = 6.
alpha = 0.3
Re = 4.

# parameter to convert to physical units
Scaling = 1.

# maximum number of gridpoints for solving the radial SE
MaxSol = 6000

# number of degree bins to calculate the diff. cross section in
nDegrees=181

def InputParams():
    # define and create the input parameters for the program
    
    # probed energy interval in EnerNum amount of steps
    MinEner, MaxEner = 0.08, 10
    EnerNum = 300
    
    # Define a maximum distance for the potential, where it approaches 0 sufficiently 
    MaxDist = 5 / alpha
    
    # Define a maximum L, as LMax \approx k * MaxDist, where k = sqrt(E) (has to be integer!)
    # Here we implement a maximum energy of 16
    LMax = int(4*MaxDist)
    
    # Define a range of R values for integration
    Start = Re / 10000 # avoid divergence at 0 or divide by 0
    HStep = 0.005
    MaxI = int(m.ceil((MaxDist-Start)/HStep))
    
    # Use the Morse function as potential (effective potential)
    potential = 'Morse'
    
    return MaxEner, MinEner, EnerNum, LMax, Start, HStep, MaxDist, MaxI, potential

def CalcScatterMorse(MaxEner, MinEner, EnerNum, LMax, Start, HStep, MaxDist, MaxI):
    # Modified the function CalcScatterpotential to specificly use the Morse potential
    #  As only finite amounts of angular momentum can be transferred to the target, we cut all sums at LMax = k * MaxDist
    # We have improved on this by taking everywhere LMax = 4*MaxDist (k runs only from 1 to 10**0.5)
    #      Calculates total and angular cross sections for a sequence of
    #      energies
    #      Cross sections are computed in two different ways:
    #         i)  exactly
    #         ii) in the Born approximation
    #
    #   Results for total cross section are written to a file "sigmadat"
    #   Results for differential cross section are written to a "sigmadiff"
    #   The radial wavefunctions are written to "wavefunc" (only the first 6, or first LMax if LMax < 6)
    #
    #   wavef[l,i] is de golffunctie u_l(R_i), met R_i = start + i * Hstep
 
    # initialise all data and datafiles
    wavefunc = np.zeros([6,MaxSol])
    fwavefunc = open("wavefunc", 'w')

    # phase shift delta_L for every energy and L
    fdeltaL = open("deltaL", 'w')
    fdeltaL.write( "#1:Energy,2:L,3:delta_L\n")

    # total cross section, decomposed per L and cumulative up to L
    fsigmaL = open("sigmaL", 'w')
    fsigmaL.write( "#1:Energy,2:L,3:S_L,4:S_LBorn,5:S_Tot,6:S_TotBorn\n")

    # total cross section
    fsigmadat = open("sigmadat", 'w')
    fsigmadat.write("#1:Energy,2:SigmaTot,3:SigmaTotBorn\n")

    # angular dependence of the differential cross section
    fsigmadiff = open("sigmadiff",'w')
    fsigmadiff.write("#1:theta,2:Energy,3:dSigma/dOmega(theta)\n")

    #We calculate the cross section only for the Morse potential
    #which is non-singular
    Singular = False
    Function = Morse

    # Start loop over the energies
    DeltaE = (MaxEner-MinEner)/EnerNum
    Ener = MinEner
    for i in range(0,EnerNum): # loop energies
        # calculate k for this Energy
        k = m.sqrt(Scaling*Ener)
        
        # initialize cross-sections = 0
        SigmaTot = 0.0
        SigmaTotBorn = 0.0
        DSigma1, DSigma2, SigmaOmega = np.zeros(nDegrees), np.zeros(nDegrees), np.zeros(nDegrees)
        
        # for each energy, sum over the partial waves.
        # compute the phase shift exactly with function CalcDelta
        for L in range(0, LMax+1):
            Delta = CalcDelta(Function, Singular, L, Ener, k, Start, MaxDist, MaxI, HStep, wavefunc, fdeltaL)
            SigmaL = 4.*m.pi/(k*k) *(2*L+1)*m.sin(Delta)*m.sin(Delta);
            SigmaTot += SigmaL

            # Calculate the phase shift and total cross section in the Born Approximation
            DeltaBorn = CalcDeltaBorn(Function, L, k, Start, MaxI, HStep)
            SigmaLBorn = 4.* m.pi/k/k*(2*L+1)*m.sin(DeltaBorn)*m.sin(DeltaBorn)
            SigmaTotBorn += SigmaLBorn

            # Write out the computed results for each L:
            #    SigmaL = contribution of this L to the total cross section
            #    SigmaTot = total cross section resulting from partial waves up to L

            txt = str(Ener) + "\t" + str(L) + "\t" + str(SigmaL) + "\t" +  str(SigmaLBorn) + "\t" + str(SigmaTot) + "\t" + str(SigmaTotBorn) + "\n"
            fsigmaL.write(txt)

            # Compute the differential cross section, using the Legendre polynomials.
            # j in degrees, theta in radians. (j is stored.)
            for j in range(0, nDegrees):
                theta = m.pi*j/180.0
                DSigma1.itemset(j, DSigma1[j] + ((2.*L+1.)*m.sin(Delta) * m.cos(Delta)*Legendre(L,m.cos(theta))))
                DSigma2.itemset(j, DSigma2[j] + ((2.*L+1.)*m.sin(Delta) * m.sin(Delta)*Legendre(L,m.cos(theta))))
            # End of the loop over the angles

        # End of the loop over partial waves

        for j in range(0, nDegrees):
            SigmaOmega.itemset(j,1./(k*k) * (DSigma1[j]*DSigma1[j] + DSigma2[j]*DSigma2[j]))

            txt = str(j) + '\t' + str(Ener) + '\t' + str(SigmaOmega[j]) + '\n'
            fsigmadiff.write(txt)
        fsigmadiff.write("\n\n")
        fsigmaL.write("\n")
        fdeltaL.write("\n\n")

        # write out all sums over L:
        # print("{:8d}   Ener={:8.3f}   SigmaTot={:8.3e}   TanDelta={:8.3e}".format(i+1,Ener,SigmaTot,m.tan(Delta)))

        txt = str(Ener) + '\t' + str(SigmaTot) + '\t' + str(SigmaTotBorn) + '\n'
        fsigmadat.write(txt)

        # Write the first 6 (or LMax if LMax < 6) wave functions in a file
        for jj in range(0, MaxI+1, 3):
            txt = str(Ener)+"\t"+str(Start + jj*HStep)
            ll = 0
            while ll < 6 and ll < LMax:
                txt += "\t" + str(wavefunc[ll][jj])
                ll+=1
            txt += "\n"
            fwavefunc.write(txt)
        fwavefunc.write("\n\n")

        #increment energy
        Ener += DeltaE

    fsigmadat.close()
    fsigmaL.close()
    fsigmadiff.close()
    fwavefunc.close()
    fdeltaL.close()
    print("Output written to files sigmadat, sigmadiff, sigmaL, wavefunc, deltaL.\n")
# End of function CalcScatter

def CalcDelta(F, Singular, L, Ener, k, Start, MaxDist, MaxI, HStep, wavefunc, fdeltaL):

# Calculate the phase shift at certain angular momentum and energy.
# GQuotient: defined in Eq. (2.9b)
# L: angular momentum quantum number
# MaxDist: First radius, r_1 in Eqs. (2.9a) and (2.9b)
# SecR: Second radius,r_2 in Eqs. (2.9a) and (2.9b)

    SecR = MaxDist + 0.5*m.pi/k # MaxDist + 1/4 wavelength = Second radius
    SecMaxI = int((SecR-Start)/HStep)
    if SecMaxI >= MaxSol:
        raise Exception("Error, second point in R = {:f}, for calculation of delta is out of range. Increase max radius or energy!".format(SecR))
    SecR = Start+SecMaxI*HStep
    GQuotient = FindGQuotient(F, Singular, L, Ener, Start, MaxDist, MaxI, SecR, SecMaxI, HStep, wavefunc)
    Delta = PhaseSft(L, k, GQuotient, MaxDist, SecR)
    #Correc = PhaseSftCor(L, k)
    txt = str(Ener) + "\t" + str(L) + "\t" + str(Delta) + "\n"
    fdeltaL.write(txt)
    return Delta

def CalcDeltaBorn( F, L, k, Start, MaxI, HStep):

# EXERCISE: Write a simple integration routine to calculate the Born approximation.
# Extra: an example of trapezium integration can be found in isotropho-loop-FT.f  line 61 - 73
# L: angular momentum quantum number
# k = momentum in units \hbar
#
#  Jan Ryckebusch, November 2007
# -------------------------------------------------------------
#  Note with respect to the Born approximation:
#  -----------------------------------------------
#  the integral over
#
#     (r**2) * (j_l (kr))**2 * V(r)
#
#   is divergent for V(r) = Lennard-Jones and r small
#
#      As a matter of fact: reasonable results require some
#      "renormalization" - here this reflects itself in cutting the
#      the hard short-range part of the Lennard-Jones potential -
#      reasonable results can be obtained for Start values of the order:
#       Start \approx 0.9
#
    DeltaBorn = 0.0
    R = Start
    for i in range( 1, MaxI+1, 1):
        if i == 1 or i== MaxI:
            weight = 0.5
        else:
            weight = 1.0
        bess = SphBesJ(L, k*R)
        DeltaBorn += weight * F(R,0,0)*bess*bess*R*R

        R += HStep
    return -DeltaBorn * HStep * k


def PhaseSft( L, k, GQuotient, MaxDist, SecR):

# Calculates the phase shift Delta_l, using Eq. (2.9a)

    Help = (GQuotient*SphBesJ(L,k*MaxDist)-SphBesJ(L,k*SecR))
    Help /= GQuotient*SphBesN(L,k*MaxDist)-SphBesN(L,k*SecR)
    return m.atan(Help)


def FindGQuotient(F, Singular, L, Ener, Start, MaxDist, MaxI, SecR, SecMaxI, HStep, wavefunc):

# Find GQuotient defined in Eq. (2.9b), from which the phase shift can be found
# L: Angular momentum quantum number
# Ener: Energy
# MaxR: Maximum integration radius for first integration
# SecR: MaxR+1/4 of a wavelength

#
    Phi1 = Start**(L+1)
    Phi2 = (Start+HStep)**(L+1)

    FArr = FillFArr(F, 0, SecMaxI, L, Ener, Start, HStep)
    Solution = Numerov(HStep, 0, SecMaxI, FArr, Singular, Phi1, Phi2)

    # store the the (first 6) wave functions in the variable wavefunc

    if L < 6:
        for j in range(0, SecMaxI+1):
            wavefunc[L].itemset(j, Solution[j])

    PhiEnd1 = Solution[MaxI]
    PhiEnd2 = Solution[SecMaxI]
    return PhiEnd2*MaxDist/(PhiEnd1*SecR)


def FillFArr( F, First, Last, L, Ener, Start, HStep):

# Fill the array FArr = F(R, L, E) (see below)
# for use in the Numerov routine

    FArr = np.empty(MaxSol)
    for i in range(First, Last+1):
        R = Start+i*HStep
        Value = F(R, L, Ener)
        FArr.itemset(i, Value)
    return FArr


# Implement the Morse potential
  
def Morse(R, L, Ener):
# Morse potential, for e.g. the energy of a diatomic molecule
# (effective potential)
# R: radial distance
# L: Angular momentum
# Ener: Energy
    R2 = R*R
    fac = np.exp(-(R - Re) * alpha)
    return L*(L+1)/R2 - Scaling * V0 * fac * (2 - fac) - Scaling*Ener



def SphBesJ(L, X):
    return scipy.special.spherical_jn(L,X)


def SphBesN( L, X):

# Returns the spherical bessel function n_l(x) as a function of l and x
# Upward recursion is used.

    if L==0:
        return -m.cos(X)/X
    elif L==1:
        return -m.cos(X)/X/X-m.sin(X)/X
    else:
        HelpCos =  m.cos(X)
        NLMin1 = -HelpCos/X
        NL = NLMin1/X-m.sin(X)/X
        for HelpL in range(2, L+1):
            NLMin2 = NLMin1
            NLMin1 = NL
            NL = (2*HelpL-1)/X*NLMin1 - NLMin2
        return NL

def Legendre( L, X):

# Returns the Legendre Polynomial of order L with argument X.
# Upward recursion is used.

    PN=[]
    PN.insert(0, 1.0)
    PN.insert(1, X)
    for J in range(2, L+1):
        PN.insert(J, (2.0*J-1.0)/J * X * PN[J-1]-(J-1.0)/J * PN[J-2])
    return PN[L]

def Numerov( Delta, StartI, EndI, FArr, Singular, Phistart, Phinext):

#  Integrates the Schrodinger equation. The initial values
#  of the Solution are Phistart and Phinext resectively. The integration
#  step is Delta, and the integration steps run from StartI to EndI. StartI may be larger
#  than EndI; in that case, integration is performed backward.
#  The output values is the Solution, stored in the array "Solution".
#  Sing determines whether the potential contains a singularity at
#  r=0.
#  If there is a singularity at r=0, the value of the Numerov
#  function w at r=0 is taken equal to Phistart, and not
#  equal to Phistart/(1-h^2 FArr/12).
#
#  This array is declared with linear size MaxSol.
#  Delta is the integration step.
#  The equation solved is
#  Psi''(R_I) = FArr(I) Psi(R_I)
#  FArr must therefore be filled with the appropriate values before
#  calling the present routine. In the case of the radial
#  Schrodinger equation, FArr would contain the values
#  FArr(I) = 2*(V(R)-E)+L*(L+1)/R**2 for R=R_I.

    Solution = [None]*MaxSol
    if Delta > 0:
        IStep = 1
    else:
        IStep = -1

    Deltasq = Delta*Delta
    Fac = Deltasq/12.

    if( Singular):
        Wprev = Phistart
    else:
        Wprev=(1-Fac*FArr[StartI+IStep])*Phistart
        Solution[StartI] = Phistart

    Phi = Phinext
    Solution[StartI+IStep]= Phinext
    W = ( 1 - Fac * FArr[StartI+IStep] )*Phi

    for i in range(StartI+IStep, EndI-IStep+1, IStep):
        Wnext = W*2. - Wprev + Deltasq*Phi*FArr[i]
        Wprev = W
        W     = Wnext
        Phi   = W/(1-Fac*FArr[i+IStep])
        Solution[i+IStep] = Phi

    return Solution


# Begin Program
if __name__=="__main__":
    # initialise parameters
    MaxEner, MinEner, EnerNum, LMax, Start, HStep, MaxDist, MaxI, potential= InputParams()
    
    # calculate the cross-sections/wavefunctions for the chosen parameters
    CalcScatterMorse(MaxEner, MinEner, EnerNum, LMax, Start, HStep, MaxDist, MaxI)
# End Program
