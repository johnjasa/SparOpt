import numpy as np

from openmdao.api import ExplicitComponent

class NetPressure(ExplicitComponent):

	def setup(self):
		self.add_input('Z_spar', val=np.zeros(11), units='m')
		#self.add_input('spar_draft', val=0., units='m')
		#self.add_input('L_ball', val=0., units='m')
		#self.add_input('rho_ball', val=0., units='kg/m**3')

		self.add_output('net_pressure', val=np.zeros(10), units='MPa')

		self.declare_partials('net_pressure', 'Z_spar', rows=np.arange(10), cols=np.arange(10))

	def compute(self, inputs, outputs):
		Z_spar = inputs['Z_spar']
		#z_ball = -inputs['spar_draft'][0] + inputs['L_ball'][0]
		#rho_ball = inputs['rho_ball']

		outputs['net_pressure'] = 1025. * 9.80665 * Z_spar[:-1] * 1e-6 #positive outwards
		"""
		#if internal pressure due to ballast:
		for i in xrange(len(Z_spar)-1):
			if Z_spar[i] < z_ball:
				outputs['net_pressure'] += rho_ball * (z_ball - Z_spar[i])
		"""

	def compute_partials(self, inputs, partials):
		partials['net_pressure', 'Z_spar'] =  1025. * 9.80665 * 1e-6 * np.ones(10)