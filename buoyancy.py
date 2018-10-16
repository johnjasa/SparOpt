import numpy as np

from openmdao.api import ExplicitComponent

class Buoyancy(ExplicitComponent):

	def setup(self):
		self.add_input('D_secs', val=np.zeros(3), units='m')
		self.add_input('L_secs', val=np.zeros(3), units='m')

		self.add_output('buoy_spar', val=0., units='N')
		self.add_output('CoB', val=0., units='m')

	def compute(self, inputs, outputs):
		D_secs  = inputs['D_secs']
		L_secs  = inputs['L_secs']

		volume = np.pi / 4. * D_secs[0]**2. * (L_secs[0] - 10.) + np.pi / 4. * D_secs[1]**2. * L_secs[1] + np.pi / 4. * D_secs[-1]**2. * L_secs[-1]

		outputs['buoy_spar'] = volume * 1025. * 9.80665
		outputs['CoB'] = (-2. * np.pi / 4. * D_secs[0]**2. * L_secs[0] + -8. * np.pi / 4. * D_secs[1]**2. * L_secs[1] + (-L_secs[-1] / 2. - L_secs[0] - L_secs[1] + 10.) * np.pi/4. * D_secs[-1]**2. * L_secs[-1]) / volume
		#TODO: make this more general with respect to number of sections and where they start