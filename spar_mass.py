import numpy as np

from openmdao.api import ExplicitComponent

class SparMass(ExplicitComponent):

	def setup(self):
		self.add_input('D_spar', val=np.zeros(10), units='m')
		self.add_input('L_spar', val=np.zeros(10), units='m')
		self.add_input('wt_spar', val=np.zeros(10), units='m')
		self.add_input('A_R', val=np.zeros(10), units='m**2')
		self.add_input('r_e', val=np.zeros(10), units='m')
		self.add_input('l_stiff', val=np.zeros(10), units='m')
		
		self.add_output('M_spar', val=np.zeros(10), units='kg')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		D_spar  = inputs['D_spar']
		L_spar  = inputs['L_spar']
		wt_spar  = inputs['wt_spar']
		A_R = inputs['A_R']
		r_e = inputs['r_e']
		l_stiff = inputs['l_stiff']

		outputs['M_spar'] = np.zeros(10)

		for i in range(len(D_spar)):
			outputs['M_spar'][i] += np.pi / 4. * (D_spar[i]**2. - (D_spar[i] - 2. * wt_spar[i])**2.) * L_spar[i] * 7850.
			outputs['M_spar'][i] += 2. * np.pi * r_e[i] * A_R[i] * L_spar[i] / l_stiff[i] * 7850.

	def compute_partials(self, inputs, partials):
		D_spar  = inputs['D_spar']
		L_spar  = inputs['L_spar']
		wt_spar  = inputs['wt_spar']
		A_R = inputs['A_R']
		r_e = inputs['r_e']
		l_stiff = inputs['l_stiff']

		partials['M_spar', 'D_spar'] = np.zeros((len(D_spar),len(D_spar)))
		partials['M_spar', 'L_spar'] = np.zeros((len(D_spar),len(D_spar)))
		partials['M_spar', 'wt_spar'] = np.zeros((len(D_spar),len(D_spar)))
		partials['M_spar', 'A_R'] = np.zeros((len(D_spar),len(D_spar)))
		partials['M_spar', 'r_e'] = np.zeros((len(D_spar),len(D_spar)))
		partials['M_spar', 'l_stiff'] = np.zeros((len(D_spar),len(D_spar)))

		for i in range(len(D_spar)):
			partials['M_spar', 'D_spar'][i,i] = np.pi * wt_spar[i] * L_spar[i] * 7850.
			partials['M_spar', 'L_spar'][i,i] = np.pi / 4. * (D_spar[i]**2. - (D_spar[i] - 2. * wt_spar[i])**2.) * 7850. + 2. * np.pi * r_e[i] * A_R[i] / l_stiff[i] * 7850.
			partials['M_spar', 'wt_spar'][i,i] = np.pi * (D_spar[i] - 2. * wt_spar[i]) * L_spar[i] * 7850.
			partials['M_spar', 'A_R'][i,i] = 2. * np.pi * r_e[i] * L_spar[i] / l_stiff[i] * 7850.
			partials['M_spar', 'r_e'][i,i] = 2. * np.pi * A_R[i] * L_spar[i] / l_stiff[i] * 7850.
			partials['M_spar', 'l_stiff'][i,i] = -2. * np.pi * r_e[i] * A_R[i] * L_spar[i] / l_stiff[i]**2. * 7850.