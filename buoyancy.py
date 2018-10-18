import numpy as np

from openmdao.api import ExplicitComponent

class Buoyancy(ExplicitComponent):

	def setup(self):
		self.add_input('D_spar', val=np.zeros(3), units='m')
		self.add_input('L_spar', val=np.zeros(3), units='m')

		self.add_output('buoy_spar', val=0., units='N')
		self.add_output('CoB', val=0., units='m')

	def compute(self, inputs, outputs):
		D_spar  = inputs['D_spar']
		L_spar  = inputs['L_spar']

		volume = np.pi / 4. * D_spar[0]**2. * L_spar[0] + np.pi / 4. * D_spar[1]**2. * L_spar[1] + np.pi / 4. * D_spar[-1]**2. * (L_spar[-1] - 10.)

		outputs['buoy_spar'] = volume * 1025. * 9.80665
		outputs['CoB'] = (-2. * np.pi / 4. * D_spar[-1]**2. * (L_spar[-1] - 10.) + -8. * np.pi / 4. * D_spar[1]**2. * L_spar[1] + (-L_spar[0] / 2. - L_spar[1] - L_spar[-1] + 10.) * np.pi/4. * D_spar[0]**2. * L_spar[0]) / volume
		
		#TODO: make this more general with respect to number of sections and where they start