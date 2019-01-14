import numpy as np

from openmdao.api import ExplicitComponent

class THull(ExplicitComponent):

	def setup(self):
		#self.add_input('moment_wind', val=0., units='m/s')
		self.add_input('dmoment_dv', val=0., units='N*s')

		self.add_output('T_hull', val=np.zeros(10), units='N*m')

		#self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		# = inputs['']

		outputs['T_hull'] = np.zeros(10) #yaw moment is same as rotor pitching moment with 90 deg phase. Other components?

	#def compute_partials(self, inputs, partials):
	#	partials['', ''] = 