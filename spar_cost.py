import numpy as np

from openmdao.api import ExplicitComponent

class SparCost(ExplicitComponent):

	def setup(self):
		self.add_input('D_spar', val=np.zeros(10), units='m')
		self.add_input('D_spar_p', val=np.zeros(11), units='m')
		self.add_input('wt_spar', val=np.zeros(10), units='m')
		self.add_input('L_spar', val=np.zeros(10), units='m')
		self.add_input('l_stiff', val=np.zeros(10), units='m')
		self.add_input('h_stiff', val=np.zeros(10), units='m')
		self.add_input('t_f_stiff', val=np.zeros(10), units='m')
		self.add_input('A_R', val=np.zeros(10), units='m**2')
		self.add_input('r_f', val=np.zeros(10), units='m')
		self.add_input('r_e', val=np.zeros(10), units='m')
		self.add_input('tot_M_spar', val=0., units='kg')

		self.add_output('spar_cost', val=0.)

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		#Cost models taken from "Optimum Design of Steel Structures" by J. Farkas and K. Jarmai
		D_spar = inputs['D_spar'] * 1000. #mm
		D_spar_p = inputs['D_spar_p'] * 1000. #mm
		wt_spar = inputs['wt_spar'] * 1000. #mm
		L_spar = inputs['L_spar'] * 1000. #mm
		l_stiff = inputs['l_stiff'] * 1000. #mm
		h_stiff = inputs['h_stiff'] * 1000. #mm
		t_f_stiff = inputs['t_f_stiff'] * 1000. #mm
		A_R = inputs['A_R'] * 1000.**2. #mm^2
		r_f = inputs['r_f'] * 1000. #mm
		r_e = inputs['r_e'] * 1000. #mm
		tot_M_spar = inputs['tot_M_spar']

		#Material costs
		k_m = 1.5 #dollar per kg steel

		Km = k_m * tot_M_spar

		rho_steel = 7.85 * 10**(-6.) #kg/mm^3

		#Fabrication costs
		k_f = k_m #dollar per min

		#forming plates into shell elements
		Lam_df = 3. #coefficient of complexity
		mu = 6.8582513 - 4.52721 * wt_spar**(-0.5) + 0.009531996 * D_spar**0.5 #older book suggest only valid for D < 3m
		Kf = k_f * Lam_df * np.sum(np.exp(mu))
		
		#welding three curved shells into a segment
		C1 = 1. #welding technology parameter
		Lam_dw = 3. #coefficient of complexity
		kappa = 3. #number of elements to be welded together
		C2 = 0.1346 * 10**(-3.) #SAW welding technology, V butt weld
		a_w = 10. #welding size (mm)
		V_section = np.pi * D_spar * wt_spar * L_spar
		L_weld = 3. * L_spar
		Kw1 = k_f * np.sum(C1 * Lam_dw * np.sqrt(kappa * rho_steel * V_section) + 1.3 * C2 * a_w**2. * L_weld)
		
		#welding ring stiffeners together with two welds
		C1 = 1. #welding technology parameter
		Lam_dw = 3. #coefficient of complexity
		kappa = 2. #number of elements to be welded together
		C2 = 0.2349 * 10**(-3.) #SAW welding technology, fillet weld
		a_w = 10. #welding size (mm)
		V_stiffener = 2. * np.pi * r_e * A_R #area ringstiff * radius to centre of area * 2 * pi
		N_stiffener = L_spar / l_stiff
		L_weld = 2. * np.pi * (r_f + t_f_stiff) * 2. #two fillet welds pr stiffener
		Kw2 = np.sum(k_f * (C1 * Lam_dw * np.sqrt(kappa * rho_steel * V_stiffener) + 1.3 * C2 * a_w**2. * L_weld) * N_stiffener)

		#welding ring stiffeners to shell segments with two welds
		C1 = 1. #welding technology parameter
		Lam_dw = 3. #coefficient of complexity
		C2 = 0.2349 * 10**(-3.) #SAW welding technology, fillet weld
		a_w = 10. #welding size (mm)
		V_stiffener = 2. * np.pi * r_e * A_R #area ringstiff * radius to centre of area * 2 * pi
		N_stiffener = L_spar / l_stiff
		kappa = N_stiffener #number of elements to be welded together
		V_section = np.pi * D_spar * wt_spar * L_spar
		L_weld = 2. * np.pi * (r_f + t_f_stiff + h_stiff) * 2. #two fillet welds pr stiffener
		Kw3 = np.sum(k_f * (C1 * Lam_dw * np.sqrt(kappa * rho_steel * (V_stiffener * N_stiffener + V_section)) + 1.3 * C2 * a_w**2. * L_weld * N_stiffener))

		#welding segments together
		C1 = 1. #welding technology parameter
		Lam_dw = 3.#coefficient of complexity
		kappa = 10 #number of elements to be welded together (number of sections)
		C2 = 0.1346 * 10**(-3.) #SAW welding technology, V butt weld
		a_w = 10. #welding size (mm)
		V_tot = tot_M_spar / rho_steel
		Kw4 = k_f * C1 * Lam_dw * np.sqrt(kappa * rho_steel * V_tot)
		for i in xrange(1,11-1):
			L_weld = np.pi * D_spar_p[i]
			Kw4 += k_f * 1.3 * C2 * a_w**2. * L_weld

		#painting (outside)
		Lam_dp = 2. #coefficient of complexity
		a_gc = 3. * 10**(-6.) #min/mm^2
		a_tc = 4.15 * 10**(-6.) #min/mm^2
		As = np.sum(np.pi * D_spar * L_spar)
		Kp = k_f * Lam_dp * (a_gc + a_tc) * As

		#total cost
		outputs['spar_cost'] = Km + Kf + Kw1 + Kw2 + Kw3 + Kw4 + Kp


	def compute_partials(self, inputs, partials):
		D_spar = inputs['D_spar'] * 1000. #mm
		D_spar_p = inputs['D_spar_p'] * 1000. #mm
		wt_spar = inputs['wt_spar'] * 1000. #mm
		L_spar = inputs['L_spar'] * 1000. #mm
		l_stiff = inputs['l_stiff'] * 1000. #mm
		h_stiff = inputs['h_stiff'] * 1000. #mm
		t_f_stiff = inputs['t_f_stiff'] * 1000. #mm
		A_R = inputs['A_R'] * 1000.**2. #mm^2
		r_f = inputs['r_f'] * 1000. #mm
		r_e = inputs['r_e'] * 1000. #mm
		tot_M_spar = inputs['tot_M_spar']

		partials['spar_cost', 'D_spar'] = np.zeros((1,10))
		partials['spar_cost', 'D_spar_p'] = np.zeros((1,11))
		partials['spar_cost', 'wt_spar'] = np.zeros((1,10))
		partials['spar_cost', 'L_spar'] = np.zeros((1,10))
		partials['spar_cost', 'l_stiff'] = np.zeros((1,10))
		partials['spar_cost', 'h_stiff'] = np.zeros((1,10))
		partials['spar_cost', 't_f_stiff'] = np.zeros((1,10))
		partials['spar_cost', 'A_R'] = np.zeros((1,10))
		partials['spar_cost', 'r_f'] = np.zeros((1,10))
		partials['spar_cost', 'r_e'] = np.zeros((1,10))
		partials['spar_cost', 'tot_M_spar'] = 0.
		
		#Material costs
		k_m = 1.5 #dollar per kg steel

		rho_steel = 7.85 * 10**(-6.) #kg/mm^3

		partials['spar_cost', 'tot_M_spar'] += k_m

		#Fabrication costs
		k_f = k_m #dollar per min

		#forming plates into shell elements
		Lam_df = 3. #coefficient of complexity
		mu = 6.8582513 - 4.52721 * wt_spar**(-0.5) + 0.009531996 * D_spar**0.5 #older book suggest only valid for D < 3m

		partials['spar_cost', 'D_spar'] += k_f * Lam_df * np.exp(mu) * 0.5 * 0.009531996 * D_spar**(-0.5) * 1000.
		partials['spar_cost', 'wt_spar'] += k_f * Lam_df * np.exp(mu) * 0.5 * 4.52721 * wt_spar**(-1.5) * 1000.
		
		#welding three curved shells into a segment
		C1 = 1. #welding technology parameter
		Lam_dw = 3. #coefficient of complexity
		kappa = 3. #number of elements to be welded together
		C2 = 0.1346 * 10**(-3.) #SAW welding technology, V butt weld
		a_w = 10. #welding size (mm)
		V_section = np.pi * D_spar * wt_spar * L_spar
		L_weld = 3. * L_spar

		partials['spar_cost', 'D_spar'] += k_f * C1 * Lam_dw * 0.5 / np.sqrt(kappa * rho_steel * V_section) * kappa * rho_steel * np.pi * wt_spar * L_spar * 1000.
		partials['spar_cost', 'wt_spar'] += k_f * C1 * Lam_dw * 0.5 / np.sqrt(kappa * rho_steel * V_section) * kappa * rho_steel * np.pi * D_spar * L_spar * 1000.
		partials['spar_cost', 'L_spar'] += k_f * C1 * Lam_dw * 0.5 / np.sqrt(kappa * rho_steel * V_section) * kappa * rho_steel * np.pi * D_spar * wt_spar * 1000. + k_f * 1.3 * C2 * a_w**2. * 3. * 1000. * np.ones(10)
		
		#welding ring stiffeners together with two welds
		C1 = 1. #welding technology parameter
		Lam_dw = 3. #coefficient of complexity
		kappa = 2. #number of elements to be welded together
		C2 = 0.2349 * 10**(-3.) #SAW welding technology, fillet weld
		a_w = 10. #welding size (mm)
		V_stiffener = 2. * np.pi * A_R * r_e #area ringstiff * radius to centre of area * 2 * pi
		N_stiffener = L_spar / l_stiff
		L_weld = 2. * np.pi * (r_f + t_f_stiff) * 2. #two fillet welds pr stiffener

		partials['spar_cost', 'A_R'] += k_f * C1 * Lam_dw * 0.5 / np.sqrt(kappa * rho_steel * V_stiffener) * N_stiffener * kappa * rho_steel * 2. * np.pi * r_e * 1000.**2.
		partials['spar_cost', 'r_e'] += k_f * C1 * Lam_dw * 0.5 / np.sqrt(kappa * rho_steel * V_stiffener) * N_stiffener * kappa * rho_steel * 2. * np.pi * A_R * 1000.
		partials['spar_cost', 'L_spar'] += k_f * (C1 * Lam_dw * np.sqrt(kappa * rho_steel * V_stiffener) + 1.3 * C2 * a_w**2. * L_weld) / l_stiff * 1000.
		partials['spar_cost', 'l_stiff'] += -k_f * (C1 * Lam_dw * np.sqrt(kappa * rho_steel * V_stiffener) + 1.3 * C2 * a_w**2. * L_weld) * L_spar / l_stiff**2. * 1000.
		partials['spar_cost', 'r_f'] += k_f * 1.3 * C2 * a_w**2. *  2. * np.pi * 2. * N_stiffener * 1000.
		partials['spar_cost', 't_f_stiff'] += k_f * 1.3 * C2 * a_w**2. *  2. * np.pi * 2. * N_stiffener * 1000.

		#welding ring stiffeners to shell segments with two welds
		C1 = 1. #welding technology parameter
		Lam_dw = 3. #coefficient of complexity
		C2 = 0.2349 * 10**(-3.) #SAW welding technology, fillet weld
		a_w = 10. #welding size (mm)
		V_stiffener = 2. * np.pi * A_R * r_e #area ringstiff * radius to centre of area * 2 * pi
		N_stiffener = L_spar / l_stiff
		kappa = N_stiffener #number of elements to be welded together
		V_section = np.pi * D_spar * wt_spar * L_spar
		L_weld = 2. * np.pi * (r_f + t_f_stiff + h_stiff) * 2. #two fillet welds pr stiffener
		#k_f * (C1 * Lam_dw * np.sqrt(kappa * rho_steel * (V_stiffener * N_stiffener + V_section)) + 1.3 * C2 * a_w**2. * L_weld * N_stiffener)
		partials['spar_cost', 'D_spar'] += k_f * C1 * Lam_dw * 0.5 / np.sqrt(kappa * rho_steel * (V_stiffener * N_stiffener + V_section)) * kappa * rho_steel * np.pi * wt_spar * L_spar * 1000.
		partials['spar_cost', 'wt_spar'] += k_f * C1 * Lam_dw * 0.5 / np.sqrt(kappa * rho_steel * (V_stiffener * N_stiffener + V_section)) * kappa * rho_steel * np.pi * D_spar * L_spar * 1000.
		partials['spar_cost', 'L_spar'] += k_f * C1 * Lam_dw * 0.5 / np.sqrt(kappa * rho_steel * (V_stiffener * N_stiffener + V_section)) * (kappa * rho_steel * (V_stiffener / l_stiff + np.pi * D_spar * wt_spar) + 1. / l_stiff * rho_steel * (V_stiffener * N_stiffener + V_section)) * 1000. + k_f * 1.3 * C2 * a_w**2. * L_weld / l_stiff * 1000.
		partials['spar_cost', 'l_stiff'] += k_f * (-C1 * Lam_dw * 0.5 / np.sqrt(kappa * rho_steel * (V_stiffener * N_stiffener + V_section)) * (kappa * rho_steel * V_stiffener * L_spar / l_stiff**2. + L_spar / l_stiff**2. * rho_steel * (V_stiffener * N_stiffener + V_section)) - 1.3 * C2 * a_w**2. * L_weld * L_spar / l_stiff**2.) * 1000.
		partials['spar_cost', 'A_R'] += k_f * C1 * Lam_dw * 0.5 / np.sqrt(kappa * rho_steel * (V_stiffener * N_stiffener + V_section)) * kappa * rho_steel * N_stiffener * 2 * np.pi * r_e * 1000.**2.
		partials['spar_cost', 'r_e'] += k_f * C1 * Lam_dw * 0.5 / np.sqrt(kappa * rho_steel * (V_stiffener * N_stiffener + V_section)) * kappa * rho_steel * N_stiffener * 2 * np.pi * A_R * 1000.
		partials['spar_cost', 'r_f'] += k_f * 1.3 * C2 * a_w**2. * N_stiffener * 2. * np.pi * 2. * 1000.
		partials['spar_cost', 't_f_stiff'] += k_f * 1.3 * C2 * a_w**2. * N_stiffener * 2. * np.pi * 2. * 1000.
		partials['spar_cost', 'h_stiff'] += k_f * 1.3 * C2 * a_w**2. * N_stiffener * 2. * np.pi * 2. * 1000.

		#welding segments together
		C1 = 1. #welding technology parameter
		Lam_dw = 3.#coefficient of complexity
		kappa = 10 #number of elements to be welded together (number of sections)
		C2 = 0.1346 * 10**(-3.) #SAW welding technology, V butt weld
		a_w = 10. #welding size (mm)
		V_tot = tot_M_spar / rho_steel
		partials['spar_cost', 'tot_M_spar'] += k_f * C1 * Lam_dw * 0.5 / np.sqrt(kappa * rho_steel * V_tot) * kappa
		for i in xrange(1,11-1):
			partials['spar_cost', 'D_spar_p'][0,i] += k_f * 1.3 * C2 * a_w**2. * np.pi * 1000.

		#painting (outside)
		Lam_dp = 2. #coefficient of complexity
		a_gc = 3. * 10**(-6.) #min/mm^2
		a_tc = 4.15 * 10**(-6.) #min/mm^2

		partials['spar_cost', 'D_spar'] += k_f * Lam_dp * (a_gc + a_tc) * np.pi * L_spar * 1000.
		partials['spar_cost', 'L_spar'] += k_f * Lam_dp * (a_gc + a_tc) * np.pi * D_spar * 1000.