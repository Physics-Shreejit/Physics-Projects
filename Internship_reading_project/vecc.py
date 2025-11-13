import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import scipy.stats as stats

mu = 782.00    # Average value of the mass of the mother particle (in Mev).
sigma = 8.5

m_D_1 = m_D_2 = m_D = 105.66    

nbin = 200
r = 100000

#======================================================

Mother_energy_Gauss = stats.norm # In the mother rest frame.
en_M_COM = Mother_energy_Gauss.rvs(loc =782, scale = 8.5, size = r) #data = (energy of the mother in lab frame) is a list itself.   
plt.figure(1)
plt.hist(en_M_COM, bins = nbin)
plt.title('Rest mass energy of the mother following Gauss distribution in COM frame.')
plt.xlabel('Rest mass energy')
#==================================================================================

uniform_cos_theta_M_COM = stats.uniform
cos_theta_M_COM = uniform_cos_theta_M_COM.rvs(loc =-1, scale = 2, size = r)
theta_M_COM = np.arccos(cos_theta_M_COM)
plt.figure(2)
plt.hist(theta_M_COM, bins = nbin)
plt.xlabel('Theta')
plt.title('Distribution of Theta in COM frame')
plt.figure(3)
plt.hist(cos_theta_M_COM, bins = nbin)
plt.xlabel('Cos theta')
plt.title('Uniform distribution of cos theta in COM frame')

#==================================================================================

uniform_phi_M_COM = stats.uniform
phi_M_COM = uniform_phi_M_COM.rvs(loc =0, scale = 2*np.pi, size = r)
plt.figure(4)
plt.hist(phi_M_COM, bins = nbin)
plt.title('Uniform distribution of Phi in COM frame.')
plt.title('Phi')


#==================================================================================

Mother_energy_maxwell = stats.maxwell # In the lab frame.
en_M_lab = Mother_energy_maxwell.rvs(loc =782, scale = 20, size = r) #data = (energy of the mother in lab frame) is a list itself.   
plt.figure(5)
plt.hist(en_M_lab, bins = nbin)
plt.title('Energy of the mother in the lab following MB distribution.')
plt.xlabel('Energy')


#==================================================================================

mom_D_1 = mom_D_2 = mom_D = (np.sqrt((en_M_COM)**2 - (4*m_D**2)))/(2) #COM frame.
en_D_1 = en_D_2 = en_D = (en_M_COM)/2                               #COM frame.

px_1 = mom_D * np.sin(theta_M_COM) * np.cos(phi_M_COM)             #COM frame.
py_1 = mom_D * np.sin(theta_M_COM) * np.sin(phi_M_COM)             #COM frame.
pz_1 = mom_D * np.cos(theta_M_COM)                                 #COM frame. 
px_2 = -mom_D * np.sin(theta_M_COM) * np.cos(phi_M_COM)            #COM frame.
py_2 = -mom_D * np.sin(theta_M_COM) * np.sin(phi_M_COM)            #COM frame.
pz_2 = -mom_D * np.cos(theta_M_COM)                                #COM frame.

mass_M =(np.sqrt(2 * (m_D)**2 + 2*((en_D)**2 - (px_1*px_2 + py_1*py_2 + pz_1*pz_2)))) #COM frame.

#===============================================================================================

p_M_lab = np.sqrt((en_M_lab)**2 - (mu)**2)  # In the lab frame.


vel_M_lab =  (p_M_lab/en_M_lab)             # vel_M = vel of the mother rest frame(boost velocity) #np.sqrt((beta_x_M)**2 +(beta_y_M)**2 + (beta_z_M)**2)
gamma_M_lab = 1/np.sqrt(1-(vel_M_lab)**2)  #(np.sqrt((data)**2 - M**2))/(data)


inv_mass = p_M_lab/(vel_M_lab * gamma_M_lab) 

px_1_lab = px_1
py_1_lab = py_1
pz_1_lab = gamma_M_lab * (pz_1 - (vel_M_lab * en_D_1))

p_1_lab = np.sqrt((px_1_lab)**2 + (py_1_lab)**2 + (pz_1_lab)**2)
en_1_lab = np.sqrt((p_1_lab)**2 + (m_D)**2)

px_2_lab = px_2
py_2_lab = py_2
pz_2_lab = gamma_M_lab * (pz_2 - (vel_M_lab * en_D_2))

p_2_lab = np.sqrt((px_2_lab)**2 + (py_2_lab)**2 + (pz_2_lab)**2)
en_2_lab = np.sqrt((p_2_lab)**2 + (m_D)**2)

mass_M_lab = np.sqrt(2*(m_D)**2 + 2*((en_1_lab)*(en_2_lab) - (px_1_lab*px_2_lab + py_1_lab*py_2_lab + pz_1_lab*pz_2_lab)))

#=====================================================================

plt.figure(6)                          # Uniform distrbution in px_1.

H_6, X_6 = np.histogram(px_1, bins=nbin)
dX_6 = X_6[1] - X_6[0]
X_6 = X_6[:-1]
X_6 = X_6 + dX_6/2.

plt.hist(px_1, bins = nbin)
#plt.plot(X_6,H_6)
plt.title('x comp of momentum of daughter 1 in COM frame.')
plt.xlabel('x comp of momentum of daughter 1')

#=======================================================================

plt.figure(7)                          # Uniform distrbution in py_1.

H_7, X_7 = np.histogram(py_1, bins=nbin)
dX_7 = X_7[1] - X_7[0]
X_7 = X_7[:-1]
X_7 = X_7 + dX_7/2.

plt.hist(py_1, bins = nbin)
#plt.plot(X_7,H_7)
plt.title('y comp of momentum of daughter 1 in COM frame.')
plt.xlabel('y comp of momentum of daughter 1')

#============================================

plt.figure(8)                          # Uniform distrbution in pz_1.

H_8, X_8 = np.histogram(pz_1, bins=nbin)
dX_8 = X_8[1] - X_8[0]
X_8 = X_8[:-1]
X_8 = X_8 + dX_8/2.

plt.hist(pz_1, bins = nbin)
#plt.plot(X_8,H_8)
plt.title('z comp of momentum of daughter 1 in COM frame.')
plt.xlabel('z comp of momentum of daughter 1')

#=============================================

plt.figure(9)                          # Uniform distrbution in px_2.

H_9, X_9 = np.histogram(px_2, bins=nbin)
dX_9 = X_9[1] - X_9[0]
X_9 = X_9[:-1]
X_9 = X_9 + dX_9/2.

plt.hist(px_2, bins = nbin)
#plt.plot(X_9,H_9)
plt.title('x comp of momentum of daughter 2 in COM frame.')
plt.xlabel('x comp of momentum of daughter 2')

#================================================

plt.figure(10)                          # Uniform distrbution in py_2.

H_10, X_10 = np.histogram(py_2, bins=nbin)
dX_10 = X_10[1] - X_10[0]
X_10 = X_10[:-1]
X_10 = X_10 + dX_10/2.

plt.hist(py_2, bins = nbin)
#plt.plot(X_10,H_10)
plt.title('y comp of momentum of daughter 2 in COM frame.')
plt.title('y comp of momentum of daughter 2')


#====================================================

plt.figure(11)                          # Uniform distrbution in pz_2.

H_11, X_11 = np.histogram(pz_2, bins=nbin)
dX_11 = X_11[1] - X_11[0]
X_11 = X_11[:-1]
X_11 = X_11 + dX_11/2.

plt.hist(pz_2, bins = nbin)
#plt.plot(X_11,H_11)
plt.title('z comp of momentum of daughter 2 in COM frame.')
plt.xlabel('z comp of momentum of daughter 2')


#=======================================================

plt.figure(12)                          # Gaussian distribution of reconstructed mass of the mother particle.


H_12, X_12 = np.histogram(mass_M, bins=nbin)
dX_12 = X_12[1] - X_12[0]
X_12 = X_12[:-1]
X_12 = X_12 + dX_12/2.

plt.hist(mass_M, bins = nbin)
#plt.plot(X_12,H_12)
plt.xlabel('Rest mass ( in Mev unit) of the mother particle in COM frame')
plt.title('Reconstruction of the rest mass ( in Mev unit) of the mother particle in COM frame.')

#=========================================================


plt.figure(13)                          


H_13, X_13 = np.histogram(px_1_lab, bins=nbin)
dX_13 = X_13[1] - X_13[0]
X_13 = X_13[:-1]
X_13 = X_13 + dX_13/2.

plt.hist(px_1_lab, bins = nbin)
#plt.plot(X_13,H_13)
plt.title('x comp of momentum of daughter 1 in lab frame.')
plt.xlabel('x comp of momentum of daughter 1')

#==============================================================

plt.figure(14)                          


H_14, X_14 = np.histogram(py_1_lab, bins=nbin)
dX_14 = X_14[1] - X_14[0]
X_14 = X_14[:-1]
X_14 = X_14 + dX_14/2.

plt.hist(py_1_lab, bins = nbin)
#plt.plot(X_14,H_14)
plt.title('y comp of momentum of daughter 1 in lab frame.')
plt.xlabel('y comp of momentum of daughter 1')

#==================================================================

plt.figure(15)                          


H_15, X_15 = np.histogram(pz_1_lab, bins=nbin)
dX_15 = X_15[1] - X_15[0]
X_15 = X_15[:-1]
X_15 = X_15 + dX_15/2.

plt.hist(pz_1_lab, bins = nbin)
#plt.plot(X_15,H_15)
plt.title('z comp of momentum of daughter 1 in lab frame.')
plt.xlabel('z comp of momentum of daughter 1')

#=======================================================================

plt.figure(16)                          


H_16, X_16 = np.histogram(px_2_lab, bins=nbin)
dX_16 = X_16[1] - X_16[0]
X_16 = X_16[:-1]
X_16 = X_16 + dX_16/2.

plt.hist(px_2_lab, bins = nbin)
#plt.plot(X_16,H_16)
plt.title('x comp of momentum of daughter 2 in lab frame.')
plt.xlabel('x comp of momentum of daughter 2')
#=========================================================================


plt.figure(17)                          


H_17, X_17 = np.histogram(py_2_lab, bins=nbin)
dX_17 = X_17[1] - X_17[0]
X_17 = X_17[:-1]
X_17 = X_17 + dX_17/2.

plt.hist(py_2_lab, bins = nbin)
#plt.plot(X_17,H_17)
plt.title('y comp of momentum of daughter 2 in lab frame.')
plt.xlabel('y comp of momentum of daughter 2')

#==========================================================================


plt.figure(18)                          


H_18, X_18 = np.histogram(pz_2_lab, bins=nbin)
dX_18 = X_18[1] - X_18[0]
X_18 = X_18[:-1]
X_18 = X_18 + dX_18/2.

plt.hist(pz_2_lab, bins = nbin)
#plt.plot(X_18,H_18)
plt.title('z comp of momentum of daughter 2 in lab frame.')
plt.xlabel('z comp of momentum of daughter 2')

#============================================================================

plt.figure(19)                          


H_19, X_19 = np.histogram(mass_M_lab, bins=nbin)
dX_19 = X_19[1] - X_19[0]
X_19 = X_19[:-1]
X_19 = X_19 + dX_19/2.

plt.hist(mass_M_lab, bins = nbin)
#plt.plot(X_19,H_19)
plt.xlabel('Rest mass (in Mev unit) of the mother particle in lab frame')
plt.title('Reconstruction of the rest mass (in Mev unit) of the mother particle in lab frame.')
#=============================================================================

plt.figure(20)                          


H_20, X_20 = np.histogram(inv_mass, bins=nbin)
dX_20 = X_20[1] - X_20[0]
X_20 = X_20[:-1]
X_20 = X_20 + dX_20/2.

plt.hist(inv_mass, bins = nbin)
plt.plot(X_20,H_20)
plt.xlabel('Invariant mass (in Mev unit).')


plt.show()
