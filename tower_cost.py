import numpy as np

from openmdao.api import ExplicitComponent

class TowerCost(ExplicitComponent):

	def setup(self):
		self.add_input('D_tower', val=np.zeros(10), units='m')
		self.add_input('D_tower_p', val=np.zeros(11), units='m')
		self.add_input('wt_tower', val=np.zeros(10), units='m')
		self.add_input('L_tower', val=np.zeros(10), units='m')
		self.add_input('tot_M_tower', val=0., units='kg')

		self.add_output('tower_cost', val=0.)

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		#Cost models taken from "Optimum Design of Steel Structures" by J. Farkas and K. Jarmai
		D_tower = inputs['D_tower'] * 1000. #mm
		D_tower_p = inputs['D_tower_p'] * 1000. #mm
		wt_tower = inputs['wt_tower'] * 1000. #mm
		L_tower = inputs['L_tower'] * 1000. #mm
		tot_M_tower = inputs['tot_M_tower']

		#Material costs
		k_m = 4.5 #dollar per kg steel

		Km = k_m * tot_M_tower

		rho_steel = 8.5 * 10**(-6.) #kg/mm^3, including secondary structures

		#Fabrication costs
		k_f = k_m #dollar per min

		#forming plates into shell elements
		Lam_df = 3. #coefficient of complexity
		mu = 6.8582513 - 4.52721 * wt_tower**(-0.5) + 0.009531996 * D_tower**0.5 #older book suggest only valid for D < 3m
		Kf = k_f * Lam_df * np.sum(np.exp(mu))

		#welding three curved shells into a segment
		C1 = 1. #welding technology parameter
		Lam_dw = 3. #coefficient of complexity
		kappa = 3. #number of elements to be welded together
		C2 = 0.1346 * 10**(-3.) #SAW welding technology, V butt weld
		a_w = 10. #welding size (mm)
		V_section = np.pi * D_tower * wt_tower * L_tower
		L_weld = 3. * L_tower
		Kw1 = k_f * np.sum(C1 * Lam_dw * np.sqrt(kappa * rho_steel * V_section) + 1.3 * C2 * a_w**2. * L_weld)

		#welding segments together
		C1 = 1. #welding technology parameter
		Lam_dw = 3.#coefficient of complexity
		kappa = 10 #number of elements to be welded together (number of sections)
		C2 = 0.1346 * 10**(-3.) #SAW welding technology, V butt weld
		a_w = 10. #welding size (mm)
		V_tot = tot_M_tower / rho_steel
		Kw2 = k_f * C1 * Lam_dw * np.sqrt(kappa * rho_steel * V_tot)
		for i in xrange(1,11-1):
			L_weld = np.pi * D_tower_p[i]
			Kw2 += k_f * 1.3 * C2 * a_w**2. * L_weld

		#painting (outside)
		Lam_dp = 2. #coefficient of complexity
		a_gc = 3. * 10**(-6.) #min/mm^2
		a_tc = 4.15 * 10**(-6.) #min/mm^2
		As = np.sum(np.pi * D_tower * L_tower)
		Kp = k_f * Lam_dp * (a_gc + a_tc) * As

		#total cost
		outputs['tower_cost'] = Km + Kf + Kw1 + Kw2 + Kp


	def compute_partials(self, inputs, partials):
		D_tower = inputs['D_tower'] * 1000. #mm
		D_tower_p = inputs['D_tower_p'] * 1000. #mm
		wt_tower = inputs['wt_tower'] * 1000. #mm
		L_tower = inputs['L_tower'] * 1000. #mm
		tot_M_tower = inputs['tot_M_tower']

		partials['tower_cost', 'D_tower'] = np.zeros((1,10))
		partials['tower_cost', 'D_tower_p'] = np.zeros((1,11))
		partials['tower_cost', 'wt_tower'] = np.zeros((1,10))
		partials['tower_cost', 'L_tower'] = np.zeros((1,10))
		partials['tower_cost', 'tot_M_tower'] = 0.
		
		#Material costs
		k_m = 4.5 #dollar per kg steel

		rho_steel = 8.5 * 10**(-6.) #kg/mm^3

		partials['tower_cost', 'tot_M_tower'] += k_m

		#Fabrication costs
		k_f = k_m #dollar per min

		#forming plates into shell elements
		Lam_df = 3. #coefficient of complexity
		mu = 6.8582513 - 4.52721 * wt_tower**(-0.5) + 0.009531996 * D_tower**0.5 #older book suggest only valid for D < 3m

		partials['tower_cost', 'D_tower'] += k_f * Lam_df * np.exp(mu) * 0.5 * 0.009531996 * D_tower**(-0.5) * 1000.
		partials['tower_cost', 'wt_tower'] += k_f * Lam_df * np.exp(mu) * 0.5 * 4.52721 * wt_tower**(-1.5) * 1000.

		#welding three curved shells into a segment
		C1 = 1. #welding technology parameter
		Lam_dw = 3. #coefficient of complexity
		kappa = 3. #number of elements to be welded together
		C2 = 0.1346 * 10**(-3.) #SAW welding technology, V butt weld
		a_w = 10. #welding size (mm)
		V_section = np.pi * D_tower * wt_tower * L_tower
		L_weld = 3. * L_tower

		partials['tower_cost', 'D_tower'] += k_f * C1 * Lam_dw * 0.5 / np.sqrt(kappa * rho_steel * V_section) * kappa * rho_steel * np.pi * wt_tower * L_tower * 1000.
		partials['tower_cost', 'wt_tower'] += k_f * C1 * Lam_dw * 0.5 / np.sqrt(kappa * rho_steel * V_section) * kappa * rho_steel * np.pi * D_tower * L_tower * 1000.
		partials['tower_cost', 'L_tower'] += k_f * C1 * Lam_dw * 0.5 / np.sqrt(kappa * rho_steel * V_section) * kappa * rho_steel * np.pi * D_tower * wt_tower * 1000. + k_f * 1.3 * C2 * a_w**2. * 3. * 1000. * np.ones(10)

		#welding segments together
		C1 = 1. #welding technology parameter
		Lam_dw = 3.#coefficient of complexity
		kappa = 10 #number of elements to be welded together (number of sections)
		C2 = 0.1346 * 10**(-3.) #SAW welding technology, V butt weld
		a_w = 10. #welding size (mm)
		V_tot = tot_M_tower / rho_steel
		partials['tower_cost', 'tot_M_tower'] += k_f * C1 * Lam_dw * 0.5 / np.sqrt(kappa * rho_steel * V_tot) * kappa
		for i in xrange(1,11-1):
			partials['tower_cost', 'D_tower_p'][0,i] += k_f * 1.3 * C2 * a_w**2. * np.pi * 1000.

		#painting (outside)
		Lam_dp = 2. #coefficient of complexity
		a_gc = 3. * 10**(-6.) #min/mm^2
		a_tc = 4.15 * 10**(-6.) #min/mm^2

		partials['tower_cost', 'D_tower'] += k_f * Lam_dp * (a_gc + a_tc) * np.pi * L_tower * 1000.
		partials['tower_cost', 'L_tower'] += k_f * Lam_dp * (a_gc + a_tc) * np.pi * D_tower * 1000.