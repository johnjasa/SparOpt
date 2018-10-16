import numpy as np

from openmdao.api import ImplicitComponent

class OpPointBldpitch(ImplicitComponent):

	def setup(self):
		self.add_input('D_secs', val=np.zeros(3), units='m')
		self.add_input('L_secs', val=np.zeros(3), units='m')
		self.add_input('wt_secs', val=np.zeros(3), units='m')

		self.add_output('M_secs', val=np.zeros(3), units='kg')

	def apply_nonlinear(self, inputs, outputs, residuals):
		D_secs  = inputs['D_secs']

		inputs['torque_0'] - inputs['N_gear'] * inputs['gen_torque_0']