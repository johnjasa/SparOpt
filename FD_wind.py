import numpy as np
import scipy.integrate as si
import matplotlib.pyplot as plt
import sys
sys.path.append('C:/Code/Scripts')
import spectra.wind as wind
import load.aeroloads_elem as aero_elem

def FDwind(Vhub, rho_wind, bldpitch_0, phi_dot_0, freqs):
	def coh(omega,d):
		f = omega / (2. * np.pi)
		Lambda_U = 42.0 #assumes that hub height is larger than 60m
		L = 8.1 * Lambda_U

		return np.exp(-12. * np.sqrt((0.12 * d / L)**2. + (f * d / Vhub)**2.))

	def coh_int(theta,omega,n,r1,r2,phi):
		d = np.sqrt(r1**2. + r2**2. - 2. * r1 * r2 * np.cos(theta + phi))
		return 1. / np.pi * coh(omega,d) * np.cos(n * theta)

	omega = np.linspace(freqs[0],2.*freqs[-1],1000)
	domega = omega[1] - omega[0]
	Nfreq = len(omega)

	Rhub = 2.8
	Rtip = 89.165
	Nelem = 20

	dr = (Rtip - Rhub) / Nelem
	r  = np.arange(Rhub + dr / 2, Rtip, dr)

	K0_11 = np.zeros((Nfreq,Nelem,Nelem)) #same blade
	K1_11 = np.zeros((Nfreq,Nelem,Nelem))
	K2_11 = np.zeros((Nfreq,Nelem,Nelem))
	K3_11 = np.zeros((Nfreq,Nelem,Nelem))
	K4_11 = np.zeros((Nfreq,Nelem,Nelem))

	for i in xrange(Nfreq):
		for j in xrange(Nelem):
			r1 = r[j]
			for k in xrange(j,Nelem):
				r2 = r[k]
				K0_11[i,j,k] = si.quad(coh_int, 0., np.pi, args=(omega[i],0.,r1,r2,0.))[0]
				K1_11[i,j,k] = si.quad(coh_int, 0., np.pi, args=(omega[i],1.,r1,r2,0.))[0]
				K2_11[i,j,k] = si.quad(coh_int, 0., np.pi, args=(omega[i],2.,r1,r2,0.))[0]
				K3_11[i,j,k] = si.quad(coh_int, 0., np.pi, args=(omega[i],3.,r1,r2,0.))[0]
				K4_11[i,j,k] = si.quad(coh_int, 0., np.pi, args=(omega[i],4.,r1,r2,0.))[0]
				if j != k:
					K0_11[i,k,j] = K0_11[i,j,k]
					K1_11[i,k,j] = K1_11[i,j,k]
					K2_11[i,k,j] = K2_11[i,j,k]
					K3_11[i,k,j] = K3_11[i,j,k]
					K4_11[i,k,j] = K4_11[i,j,k]

	S_wind = wind.kaimal(Vhub, omega)

	G0 = np.zeros((Nfreq/2,Nelem,Nelem), dtype=complex)
	Gm3 = np.zeros((Nfreq/2,Nelem,Nelem), dtype=complex)
	Gp3 = np.zeros((Nfreq/2,Nelem,Nelem), dtype=complex)

	Gm1 = np.zeros((Nfreq/2,Nelem,Nelem), dtype=complex)
	Gp1 = np.zeros((Nfreq/2,Nelem,Nelem), dtype=complex)
	Gm2 = np.zeros((Nfreq/2,Nelem,Nelem), dtype=complex)
	Gp2 = np.zeros((Nfreq/2,Nelem,Nelem), dtype=complex)
	Gm4 = np.zeros((Nfreq/2,Nelem,Nelem), dtype=complex)
	Gp4 = np.zeros((Nfreq/2,Nelem,Nelem), dtype=complex)

	for i in xrange(Nfreq/2):
		idx_m3 = int(np.round(np.abs(omega[i] + 3. * phi_dot_0) / domega))
		idx_p3 = int(np.round(np.abs(omega[i] - 3. * phi_dot_0) / domega))
		
		G0[i] = K0_11[i] * S_wind[i]
		Gm3[i] = K3_11[idx_m3] * S_wind[idx_m3]
		Gp3[i] = K3_11[idx_p3] * S_wind[idx_p3]
		
		idx_m4 = int(np.round(np.abs(omega[i] + 3. * phi_dot_0) / domega))
		idx_m2 = int(np.round(np.abs(omega[i] + 3. * phi_dot_0) / domega))
		idx_m1 = i
		idx_p1 = i
		idx_p2 = int(np.round(np.abs(omega[i] - 3. * phi_dot_0) / domega))
		idx_p4 = int(np.round(np.abs(omega[i] - 3. * phi_dot_0) / domega))
		
		Gm1[i] = K1_11[idx_m1] * S_wind[idx_m1]
		Gp1[i] = K1_11[idx_p1] * S_wind[idx_p1]
		Gm2[i] = K2_11[idx_m2] * S_wind[idx_m2]
		Gp2[i] = K2_11[idx_p2] * S_wind[idx_p2]
		Gm4[i] = K4_11[idx_m4] * S_wind[idx_m4]
		Gp4[i] = K4_11[idx_p4] * S_wind[idx_p4]

	G_FQ = G0 + Gp3 + Gm3
	G_M = Gm1 + Gp1 + Gm2 + Gp2 + Gm4 + Gp4

	dFn_dv_b, dFt_dv_b = aero_elem.BEM(Rtip, rho_wind, Vhub, bldpitch_0, phi_dot_0)

	dFT_dv_b = np.array([dFn_dv_b * dr])
	dMT_dv_b = np.array([dFn_dv_b * r * dr])
	dQA_dv_b = np.array([dFt_dv_b * r * dr])

	FT = np.zeros(Nfreq/2, dtype=complex)
	MT = np.zeros(Nfreq/2, dtype=complex)
	QA = np.zeros(Nfreq/2, dtype=complex)

	for i in xrange(Nfreq/2):
		FT[i] = 3.**2. * np.linalg.multi_dot((dFT_dv_b,G_FQ[i],np.transpose(dFT_dv_b)))[0][0]
		MT[i] = (3./2.)**2. * np.linalg.multi_dot((dMT_dv_b,G_M[i],np.transpose(dMT_dv_b)))[0][0]
		QA[i] = 3.**2. * np.linalg.multi_dot((dQA_dv_b,G_FQ[i],np.transpose(dQA_dv_b)))[0][0]

		FT[i] = np.sqrt(np.abs(FT[i]) / S_wind[i]) / (3. * np.sum(dFT_dv_b))
		MT[i] = np.sqrt(np.abs(MT[i]) / S_wind[i]) / (3. / 2. * np.sum(dMT_dv_b))
		QA[i] = np.sqrt(np.abs(QA[i]) / S_wind[i]) / (3. * np.sum(dQA_dv_b))

	FT = np.interp(freqs,omega[:Nfreq/2],FT)
	MT = np.interp(freqs,omega[:Nfreq/2],MT)
	QA = np.interp(freqs,omega[:Nfreq/2],QA)

	f = open('C:/Code/windspeeds/eq_wind_%d.dat' % Vhub, 'w')

	for i in xrange(len(FT)):
		f.write('%.6e %.6e %.6e %.6e\n' % (freqs[i], FT[i], MT[i], QA[i]))

	f.close()

	#return FT, MT, QA


Vhub = np.arange(50,51,1)
bldpitch_0 = 90. * np.pi / 180.
rho_wind = 1.25
phi_dot_0 = 0.
freqs = np.linspace(0,2.*np.pi,5000)

for i in xrange(len(Vhub)):
	FDwind(Vhub[i], rho_wind, bldpitch_0, phi_dot_0, freqs)

"""
4: 0.0 6.0
5: 0.0 6.0
6: 0.0 6.0
7: 0.0 6.0
8: 0.0 6.43
9: 0.0 7.23
10: 0.0 8.03
11: 0.0 9.6
12: 6.09 9.6
13: 8.33 9.6
14: 10.10 9.6
15: 11.67 9.6
16: 13.09 9.6
17: 14.41 9.6
18: 15.66 9.6
19: 16.85 9.6
20: 17.99 9.6
21: 19.08 9.6
22: 20.14 9.6
23: 21.18 9.6
24: 22.19 9.6
25: 23.17 9.6
"""