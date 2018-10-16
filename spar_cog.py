import numpy as np

from openmdao.api import ExplicitComponent

class SparCoG(ExplicitComponent):

	def setup(self):
		self.add_input('L_secs', val=np.zeros(3), units='m')
		self.add_input('M_secs', val=np.zeros(3), units='kg')

		self.add_output('CoG_spar', val=0., units='m')

	def compute(self, inputs, outputs):
		L_secs  = inputs['L_secs']
		M_secs  = inputs['M_secs']
		M_spar  = np.sum(M_secs)

		CoG_t_mass = 0.

		for i in xrange(len(L_secs)):
			CoG_sec = 10. - np.sum(L_secs[0:i]) - L_secs[i] / 2.
			
			CoG_t_mass += M_secs[i] * CoG_sec
		
		outputs['CoG_spar'] = CoG_t_mass / M_spar