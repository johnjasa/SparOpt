import numpy as np

from openmdao.api import ExplicitComponent

class SparCoG(ExplicitComponent):

	def setup(self):
		self.add_input('L_spar', val=np.zeros(10), units='m')
		self.add_input('M_spar', val=np.zeros(10), units='kg')
		self.add_input('tot_M_spar', val=0., units='kg')
		self.add_input('spar_draft', val=0., units='m')

		self.add_output('CoG_spar', val=1., units='m')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		L_spar  = inputs['L_spar']
		M_spar  = inputs['M_spar']
		tot_M_spar  = inputs['tot_M_spar']
		spar_draft = inputs['spar_draft']

		CoG_t_mass = 0.

		for i in xrange(10):
			CoG_sec = -spar_draft + np.sum(L_spar[0:i]) + L_spar[i] / 2.
			CoG_t_mass += M_spar[i] * CoG_sec
		
		outputs['CoG_spar'] = CoG_t_mass / tot_M_spar

	def compute_partials(self, inputs, partials):
		L_spar  = inputs['L_spar']
		M_spar  = inputs['M_spar']
		tot_M_spar  = inputs['tot_M_spar']
		spar_draft  = inputs['spar_draft']

		partials['CoG_spar', 'L_spar'] = np.zeros((1,10))
		partials['CoG_spar', 'M_spar'] = np.zeros((1,10))
		partials['CoG_spar', 'tot_M_spar'] = 0.
		partials['CoG_spar', 'spar_draft'] = 0.

		CoG_t_mass = 0.

		for i in xrange(10):
			CoG_sec = -spar_draft + np.sum(L_spar[0:i]) + L_spar[i] / 2.
			
			CoG_t_mass += M_spar[i] * CoG_sec

			partials['CoG_spar', 'L_spar'][0,i] += 0.5 * M_spar[i] / tot_M_spar
			partials['CoG_spar', 'M_spar'][0,i] += CoG_sec / tot_M_spar
			partials['CoG_spar', 'spar_draft'] += - M_spar[i] / tot_M_spar

			for j in xrange(i):
				partials['CoG_spar', 'L_spar'][0,j] += M_spar[i] / tot_M_spar

		partials['CoG_spar', 'tot_M_spar'] = -CoG_t_mass / tot_M_spar**2.