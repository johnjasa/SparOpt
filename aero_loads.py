import numpy as np
import scipy.interpolate as si
import scipy.misc as sm

from openmdao.api import ExplicitComponent

class AeroLoads(ExplicitComponent):

	def initialize(self):
		self.options.declare('blades', types=dict)

	def setup(self):
		blades = self.options['blades']
		self.Rtip = blades['Rtip']
		self.Rhub = blades['Rhub']
		self.N_b_elem = blades['N_b_elem']
		self.indfile = blades['indfile']
		self.bladefile = blades['bladefile']
		self.foilnames = blades['foilnames']
		self.foilfolder = blades['foilfolder']

		self.add_input('rho_wind', val=1., units='kg/m**3')
		self.add_input('windspeed_0', val=1., units='m/s')
		self.add_input('bldpitch_0', val=1., units='rad')
		self.add_input('rotspeed_0', val=1., units='rad/s')

		self.add_output('b_elem_r', val=np.ones(self.N_b_elem), units='m')
		self.add_output('b_elem_dr', val=1., units='m')
		self.add_output('Fn_0', val=np.ones(self.N_b_elem), units='N/m')
		self.add_output('Ft_0', val=np.ones(self.N_b_elem), units='N/m')
		self.add_output('dFn_dv', val=np.ones(self.N_b_elem), units='N*s/m**2')
		self.add_output('dFn_dbldpitch', val=np.ones(self.N_b_elem), units='N/(m*rad)')
		self.add_output('dFn_drotspeed', val=np.ones(self.N_b_elem), units='N*s/(m*rad)')
		self.add_output('dFt_dv', val=np.ones(self.N_b_elem), units='N*s/m**2')
		self.add_output('dFt_dbldpitch', val=np.ones(self.N_b_elem), units='N/(m*rad)')
		self.add_output('dFt_drotspeed', val=np.ones(self.N_b_elem), units='N*s/(m*rad)')

	def compute(self, inputs, outputs):
		rho_wind = inputs['rho_wind']
		windspeed_0 = inputs['windspeed_0']
		bldpitch_0 = inputs['bldpitch_0']
		rotspeed_0 = inputs['rotspeed_0']

		Rtip = self.Rtip
		Rhub = self.Rhub
		N_b_elem = self.N_b_elem
		foilnames = self.foilnames

		#read induction factors for each blade element
		all_bldpitch, all_TSR, all_r, all_a_n, all_a_t = np.loadtxt(self.indfile, skiprows=1, unpack=True)

		all_bldpitch = np.unique(all_bldpitch) * np.pi / 180.
		all_TSR = np.unique(all_TSR)
		all_r = np.unique(all_r)
		all_a_n = np.reshape(all_a_n,(len(all_bldpitch),len(all_TSR),len(all_r)))
		all_a_t = np.reshape(all_a_t,(len(all_bldpitch),len(all_TSR),len(all_r)))

		f_a_n = []
		f_a_t = []
		for i in xrange(N_b_elem):
			f_a_n.append(si.RectBivariateSpline(all_bldpitch,all_TSR,all_a_n[:,:,i]))
			f_a_t.append(si.RectBivariateSpline(all_bldpitch,all_TSR,all_a_t[:,:,i]))

		#read blade data
		endradius, chordlength, twistangle, bladefoil = np.loadtxt(self.bladefile, skiprows=1, unpack=True)

		if endradius[0] != Rhub:
			raise Exception('Hub radius in %s is not equal to defined hub radius (%.3f)' % (self.bladefile, Rhub))
		elif endradius[-1] != Rtip:
			raise Exception('Tip radius in %s is not equal to defined tip radius (%.3f)' % (self.bladefile, Rhub))

		radius = [Rhub]
		for i in xrange(1,len(endradius)):
		    radius.append((endradius[i-1] + endradius[i]) / 2.)

		#read foil data
		foildata_alpha = [[] for _ in xrange(len(foilnames))]
		foildata_cl = [[] for _ in xrange(len(foilnames))]
		foildata_cd = [[] for _ in xrange(len(foilnames))]
		for i in xrange(len(foilnames)):
		    f = open(self.foilfolder + foilnames[i] + '.dat','r')
		    for line in f:
		        li = line.split()
		        foildata_alpha[i].append(float(li[0]))
		        foildata_cl[i].append(float(li[1]))
		        foildata_cd[i].append(float(li[2]))
		    f.close()

		#interpolate blade data (evenly spaced elements)
		dr = (Rtip - Rhub) / N_b_elem
		r  = np.arange(Rhub + dr / 2, Rtip, dr)

		outputs['b_elem_r'] = r
		outputs['b_elem_dr'] = dr

		f_chordlength = si.interp1d(radius, chordlength, kind='cubic')
		f_twistangle = si.interp1d(radius, twistangle, kind='cubic')
		chordlength0 = f_chordlength(r)
		twistangle0 = f_twistangle(r)

		foilnum = []
		f_cl = []
		f_cd = []
		for i in xrange(len(r)):
		    if r[i] < endradius[0]:
		        foilnum.append(int(bladefoil[0]))
		    else:
		        for j in xrange(len(endradius)-1):
		            if (r[i] >= endradius[j]) and (r[i] < endradius[j+1]):
		                foilnum.append(int(bladefoil[j+1]))
		                break

			ALPHA = foildata_alpha[foilnum[i]]
			CL = foildata_cl[foilnum[i]]
			CD = foildata_cd[foilnum[i]]
			f_cl.append(si.interp1d(ALPHA,CL,'cubic'))
			f_cd.append(si.interp1d(ALPHA,CD,'cubic'))

		#Calculate linearized aerodynamic forces for each blade element
		TSR0 = rotspeed_0 * Rtip / windspeed_0
		dTSR_dv = -rotspeed_0 * Rtip / windspeed_0**2.
		dTSR_drotspeed = Rtip / windspeed_0

		Fn_0 = np.zeros(N_b_elem)
		dFn_dv = np.zeros(N_b_elem)
		dFn_dbldpitch = np.zeros(N_b_elem)
		dFn_drotspeed = np.zeros(N_b_elem)

		Ft_0 = np.zeros(N_b_elem)
		dFt_dv = np.zeros(N_b_elem)
		dFt_dbldpitch = np.zeros(N_b_elem)
		dFt_drotspeed = np.zeros(N_b_elem)

		for i in xrange(N_b_elem):
			if TSR0 == 0.:
				a_n_0 = 0.
				a_t_0 = 0.
			else:
				a_n_0 = f_a_n[i].ev(bldpitch_0,TSR0)
				a_t_0 = f_a_t[i].ev(bldpitch_0,TSR0)
			v_n_0 = windspeed_0 * a_n_0
			v_t_0 = rotspeed_0 * r[i] * a_t_0
			Vn_0 = windspeed_0 - v_n_0
			Vt_0 = rotspeed_0 * r[i] + v_t_0

			if TSR0 == 0.:
				phi_0 = np.pi / 2.
			else:
				phi_0 = np.arctan(Vn_0 / Vt_0)
			alpha_0 = 180. / np.pi * phi_0 - (twistangle0[i] + bldpitch_0 * 180. / np.pi)

			W0 = np.sqrt(Vn_0**2. + Vt_0**2.)

			L0 = 0.5 * rho_wind * f_cl[i](alpha_0) * chordlength0[i] * W0**2.
			D0 = 0.5 * rho_wind * f_cd[i](alpha_0) * chordlength0[i] * W0**2.

			outputs['Fn_0'][i] = L0 * np.cos(phi_0) + D0 * np.sin(phi_0)
			outputs['Ft_0'][i] = L0 * np.sin(phi_0) - D0 * np.cos(phi_0)

			if TSR0 == 0.:
				da_n_dTSR = 0.
				da_t_dTSR = 0.
				da_n_dbldpitch = 0.
				da_t_dbldpitch = 0.
			else:
				da_n_dTSR = f_a_n[i].ev(bldpitch_0, TSR0, dx=0, dy=1)
				da_t_dTSR = f_a_t[i].ev(bldpitch_0, TSR0, dx=0, dy=1)
				da_n_dbldpitch = f_a_n[i].ev(bldpitch_0, TSR0, dx=1, dy=0)
				da_t_dbldpitch = f_a_t[i].ev(bldpitch_0, TSR0, dx=1, dy=0)
			da_n_dv = da_n_dTSR * dTSR_dv
			da_t_dv = da_t_dTSR * dTSR_dv
			da_n_drotspeed = da_n_dTSR * dTSR_drotspeed
			da_t_drotspeed = da_t_dTSR * dTSR_drotspeed
			dv_n_dv = a_n_0 + windspeed_0 * da_n_dv
			dv_n_dbldpitch = windspeed_0 * da_n_dbldpitch
			dv_n_drotspeed = windspeed_0 * da_n_drotspeed
			dv_t_dv = rotspeed_0 * r[i] * da_t_dv
			dv_t_dbldpitch = rotspeed_0 * r[i] * da_t_dbldpitch
			dv_t_drotspeed = rotspeed_0 * r[i] * da_t_drotspeed + r[i] * a_t_0
			dVn_dv = 1. - dv_n_dv
			dVn_dbldpitch = -dv_n_dbldpitch
			dVn_drotspeed = -dv_n_drotspeed
			dVt_dv = dv_t_dv
			dVt_dbldpitch = dv_t_dbldpitch
			dVt_drotspeed = r[i] + dv_t_drotspeed

			if TSR0 == 0.:
				dphi_dv = 0.
				dphi_dbldpitch = 0.
				dphi_drotspeed = 0.
			else:
				dphi_dv = 1. / (1. + (Vn_0 / Vt_0)**2.) * (dVn_dv * 1. / Vt_0 - Vn_0 / Vt_0**2. * dVt_dv)
				dphi_dbldpitch = 1. / (1. + (Vn_0 / Vt_0)**2.) * (dVn_dbldpitch * 1. / Vt_0 - Vn_0 / Vt_0**2. * dVt_dbldpitch)
				dphi_drotspeed = 1. / (1. + (Vn_0 / Vt_0)**2.) * (dVn_drotspeed * 1. / Vt_0 - Vn_0 / Vt_0**2. * dVt_drotspeed)
			dalpha_dv = 180. / np.pi * dphi_dv
			dalpha_dbldpitch = 180. / np.pi * dphi_dbldpitch - 180. / np.pi
			dalpha_drotspeed = 180. / np.pi * dphi_drotspeed

			dCl_dalpha = sm.derivative(f_cl[i], alpha_0, dx = 1e-8)
			dCd_dalpha = sm.derivative(f_cd[i], alpha_0, dx = 1e-8)
			dCl_dv = dCl_dalpha * dalpha_dv
			dCl_dbldpitch = dCl_dalpha * dalpha_dbldpitch
			dCl_drotspeed = dCl_dalpha * dalpha_drotspeed
			dCd_dv = dCd_dalpha * dalpha_dv
			dCd_dbldpitch = dCd_dalpha * dalpha_dbldpitch
			dCd_drotspeed = dCd_dalpha * dalpha_drotspeed

			dL_dv = 0.5 * rho_wind * chordlength0[i] * (dCl_dv * (Vn_0**2. + Vt_0**2.) + f_cl[i](alpha_0) * (2. * Vn_0 * dVn_dv + 2. * Vt_0 * dVt_dv))
			dL_dbldpitch = 0.5 * rho_wind * chordlength0[i] * (dCl_dbldpitch * (Vn_0**2. + Vt_0**2.) + f_cl[i](alpha_0) * (2. * Vn_0 * dVn_dbldpitch + 2. * Vt_0 * dVt_dbldpitch))
			dL_drotspeed = 0.5 * rho_wind * chordlength0[i] * (dCl_drotspeed * (Vn_0**2. + Vt_0**2.) + f_cl[i](alpha_0) * (2. * Vn_0 * dVn_drotspeed + 2. * Vt_0 * dVt_drotspeed))
			dD_dv = 0.5 * rho_wind * chordlength0[i] * (dCd_dv * (Vn_0**2. + Vt_0**2.) + f_cd[i](alpha_0) * (2. * Vn_0 * dVn_dv + 2. * Vt_0 * dVt_dv))
			dD_dbldpitch = 0.5 * rho_wind * chordlength0[i] * (dCd_dbldpitch * (Vn_0**2. + Vt_0**2.) + f_cd[i](alpha_0) * (2. * Vn_0 * dVn_dbldpitch + 2. * Vt_0 * dVt_dbldpitch))
			dD_drotspeed = 0.5 * rho_wind * chordlength0[i] * (dCd_drotspeed * (Vn_0**2. + Vt_0**2.) + f_cd[i](alpha_0) * (2. * Vn_0 * dVn_drotspeed + 2. * Vt_0 * dVt_drotspeed))

			outputs['dFn_dv'][i] = dL_dv * np.cos(phi_0) - L0 * np.sin(phi_0) * dphi_dv + dD_dv * np.sin(phi_0) + D0 * np.cos(phi_0) * dphi_dv
			outputs['dFn_dbldpitch'][i] = dL_dbldpitch * np.cos(phi_0) - L0 * np.sin(phi_0) * dphi_dbldpitch + dD_dbldpitch * np.sin(phi_0) + D0 * np.cos(phi_0) * dphi_dbldpitch
			outputs['dFn_drotspeed'][i] = dL_drotspeed * np.cos(phi_0) - L0 * np.sin(phi_0) * dphi_drotspeed + dD_drotspeed * np.sin(phi_0) + D0 * np.cos(phi_0) * dphi_drotspeed

			outputs['dFt_dv'][i] = dL_dv * np.sin(phi_0) + L0 * np.cos(phi_0) * dphi_dv - dD_dv * np.cos(phi_0) + D0 * np.sin(phi_0) * dphi_dv
			outputs['dFt_dbldpitch'][i] = dL_dbldpitch * np.sin(phi_0) + L0 * np.cos(phi_0) * dphi_dbldpitch - dD_dbldpitch * np.cos(phi_0) + D0 * np.sin(phi_0) * dphi_dbldpitch
			outputs['dFt_drotspeed'][i] = dL_drotspeed * np.sin(phi_0) + L0 * np.cos(phi_0) * dphi_drotspeed - dD_drotspeed * np.cos(phi_0) + D0 * np.sin(phi_0) * dphi_drotspeed
