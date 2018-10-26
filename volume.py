import numpy as np

from openmdao.api import ExplicitComponent

class Volume(ExplicitComponent):

	def setup(self):
		self.add_input('D_spar', val=np.zeros(10), units='m')
		self.add_input('L_spar', val=np.zeros(10), units='m')

		self.add_output('sub_vol', val=0., units='m**3')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		D_spar  = inputs['D_spar']
		L_spar  = inputs['L_spar']

		outputs['sub_vol'] = 0.

		for i in xrange(len(D_spar) - 1):
			outputs['sub_vol'] += np.pi / 4. * D_spar[i]**2. * L_spar[i]

		outputs['sub_vol'] += np.pi / 4. * D_spar[-1]**2. * (L_spar[-1] - 10.)
		#TODO: last secton from 0 to 10

	def compute_partials(self, inputs, partials):
		D_spar  = inputs['D_spar']
		L_spar  = inputs['L_spar']

		partials['sub_vol', 'D_spar'] = np.zeros((1,10))
		partials['sub_vol', 'L_spar'] = np.zeros((1,10))

		for i in xrange(len(D_spar)):
			partials['sub_vol', 'D_spar'][0,i] = np.pi / 2. * D_spar[i] * L_spar[i]
			partials['sub_vol', 'L_spar'][0,i] = np.pi / 4. * D_spar[i]**2.
		
		partials['sub_vol', 'D_spar'][0,-1] = np.pi / 2. * D_spar[-1] * (L_spar[-1] - 10.)