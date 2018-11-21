import numpy as np

from openmdao.api import ExplicitComponent

class SparCost(ExplicitComponent):

	def setup(self):
		#self.add_input('D_spar', val=np.zeros(10))
		#self.add_input('wt_spar', val=np.zeros(10))
		#self.add_input('L_spar', val=np.zeros(10))
		self.add_input('tot_M_spar', val=0.)

		self.add_output('spar_cost', val=0.)

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		#Cost models taken from "Optimum Design of Steel Structures" by J. Farkas and K. Jármai
		#D_spar = inputs['D_spar']
		#wt_spar = inputs['wt_spar']
		#L_spar = inputs['L_spar']
		tot_M_spar = inputs['tot_M_spar']

		#D_spar = D_spar * 1000. #mm
		#L_spar = L_spar * 1000. #mm
		#wt_spar = t_spar * 1000. #mm
		#L_spar = L_spar * 1000. #mm

		#Material costs
		k_m = 1.0 #dollar per kg steel

		outputs['spar_cost'] = k_m * tot_M_spar

		"""
		rho_steel = 7.85 * 10**(-6.) #kg/mm^3

		Nsections = 11

		Nstiffeners = 1

		#Fabrication costs
		k_f = k_m #dollar per min

		#forming plates into shell elements
		Lam_df = 3. #coefficient of complexity
		Kf = 0.
		for i in xrange(Nsections):
			mu = 6.8582513 - 4.52721 * t_section**(-0.5) + 0.009531996 * D**0.5 #older book suggest only valid for D <  3m
			Kf += k_f * Lam_df * np.exp(mu)

		#welding three curved shells into a segment
		C1 = 1. #welding technology parameter
		Lam_dw = 3. #coefficient of complexity
		kappa = 3. #number of elements to be welded together
		C2 = 0.1346 * 10**(-3.) #SAW welding technology, V butt weld
		a_w = 10. #welding size (mm)
		Kw1 = 0.
		for i in xrange(Nsections):
			Vsection = np.pi * D * t_section * Lsection
			Lweld = 3. * Lsection
			Kw1 += k_f * (C1 * Lam_dw * np.sqrt(kappa * rho_steel * Vsection) + 1.3 * C2 * a_w**2. * Lweld)

		#welding ring stiffeners together with two welds
		C1 = 1. #welding technology parameter
		Lam_dw = 3. #coefficient of complexity
		kappa = 2. #number of elements to be welded together
		C2 = 0.2349 * 10**(-3.) #SAW welding technology, fillet weld
		a_w = 10. #welding size (mm)
		Kw2 = 0.
		for i in xrange(Nsections):
			Vstiffener_section = 0.
			Lweld = 2. * np.pi * radius_web_flange * 2. #two fillet welds pr stiffener
			Kw2 += k_f * (C1 * Lam_dw * np.sqrt(kappa * rho_steel * Vstiffener_section) + 1.3 * C2 * a_w**2. * Lweld) * Nstiffeners_section

		#welding ring stiffeners to shell segments with two welds
		C1 = 1. #welding technology parameter
		Lam_dw = 3. #coefficient of complexity
		kappa = Nstiffeners_section #number of elements to be welded together
		C2 = 0.2349 * 10**(-3.) #SAW welding technology, fillet weld
		a_w = 10. #welding size (mm)
		Kw3 = 0.
		for i in xrange(Nsections):
			Vstiffener_section = 0.
			Vsection = np.pi * D * t_section * Lsection
			Lweld = 2. * np.pi * radius_web_shell * 2. #two fillet welds pr stiffener
			Kw3 += k_f * (C1 * Lam_dw * np.sqrt(kappa * rho_steel * (Vstiffener_section + Vsection)) + 1.3 * C2 * a_w**2. * Lweld * Nstiffeners_section)

		#welding segments together
		C1 = 1. #welding technology parameter
		Lam_dw = 3.#coefficient of complexity
		kappa = Nsections #number of elements to be welded together
		C2 = 0.1346 * 10**(-3.) #SAW welding technology, V butt weld
		a_w = 10. #welding size (mm)
		Kw4 = C1 * Lam_dw * np.sqrt(kappa * rho_steel * Vi)
		for i in xrange(Nsections-1):
			Lweld = np.pi * D
			Kw4 += k_f * 1.3 * C2 * a_w**2. * Lweld

		#painting (outside)
		Lam_dp = 2. #coefficient of complexity
		a_gc = 3. * 10**(-6.) #min/mm^2
		a_tc = 4.15 * 10**(-6.) #min/mm^2
		As = np.pi * D * L
		Kp = k_f * Lam_dp * (a_gc + a_tc) * As

		#total cost
		K = Km + Kf + Kw1 + Kw2 + Kw3 + Kw4 + Kp
		"""

	def compute_partials(self, inputs, partials):
		#Material costs
		k_m = 1.0 #dollar per kg steel

		partials['spar_cost', 'tot_M_spar'] = k_m