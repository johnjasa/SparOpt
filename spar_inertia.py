import numpy as np

from openmdao.api import ExplicitComponent

class SparInertia(ExplicitComponent):

	def setup(self):
		self.add_input('D_secs', val=np.zeros(3), units='m')
		self.add_input('L_secs', val=np.zeros(3), units='m')
		self.add_input('M_secs', val=np.zeros(3), units='kg')
		self.add_input('wt_secs', val=np.zeros(3), units='m')
		self.add_input('M_ball', val=0., units='kg')
		self.add_input('CoG_ball', val=0., units='m')

		self.add_output('I_spar', val=0., units='kg*m**2')

	def compute(self, inputs, outputs):
		D_secs  = inputs['D_secs']
		L_secs  = inputs['L_secs']
		M_secs  = inputs['M_secs']
		wt_secs  = inputs['wt_secs']
		M_ball = inputs['M_ball']
		CoG_ball = inputs['CoG_ball']

		outputs['I_spar'] = 0.

		for i in xrange(len(L_secs)):			
			CoG_sec = 10. - np.sum(L_secs[0:i]) - L_secs[i] / 2.
			outputs['I_spar'] += M_secs[i] * CoG_sec**2.

		outputs['I_spar'] += M_ball * CoG_ball**2.