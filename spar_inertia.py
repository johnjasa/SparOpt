import numpy as np

from openmdao.api import ExplicitComponent

class SparInertia(ExplicitComponent):

	def setup(self):
		self.add_input('L_spar', val=np.zeros(10), units='m')
		self.add_input('M_spar', val=np.zeros(10), units='kg')
		self.add_input('spar_draft', val=0., units='m')

		self.add_output('I_spar', val=0., units='kg*m**2')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		L_spar  = inputs['L_spar']
		M_spar  = inputs['M_spar']
		spar_draft = inputs['spar_draft']

		outputs['I_spar'] = 0.

		for i in xrange(len(L_spar)):	
			CoG_sec = -spar_draft + np.sum(L_spar[0:i]) + L_spar[i] / 2.
			outputs['I_spar'] += M_spar[i] * CoG_sec**2.

	def compute_partials(self, inputs, partials):
		L_spar  = inputs['L_spar']
		M_spar  = inputs['M_spar']
		spar_draft = inputs['spar_draft']

		partials['I_spar', 'L_spar'] = np.zeros((1,10))
		partials['I_spar', 'M_spar'] = np.zeros((1,10))
		partials['I_spar', 'spar_draft'] = 0.

		for i in xrange(len(L_spar)):
			CoG_sec = -spar_draft + np.sum(L_spar[0:i]) + L_spar[i] / 2.
			partials['I_spar', 'M_spar'][0,i] += CoG_sec**2.
			partials['I_spar', 'spar_draft'] += -2. *  M_spar[i] * CoG_sec
			partials['I_spar', 'L_spar'][0,i] += 2. *  M_spar[i] * CoG_sec * 0.5

			for j in xrange(i):
				partials['I_spar', 'L_spar'][0,j] += 2. *  M_spar[i] * CoG_sec