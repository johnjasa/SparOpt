import numpy as np

from openmdao.api import ExplicitComponent

class Buoyancy(ExplicitComponent):

	def setup(self):
		self.add_input('D_spar', val=np.zeros(3), units='m')
		self.add_input('L_spar', val=np.zeros(3), units='m')
		self.add_input('spar_draft', val=0., units='m')

		self.add_output('buoy_spar', val=0., units='N')
		self.add_output('CoB', val=0., units='m')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		D_spar  = inputs['D_spar']
		L_spar  = inputs['L_spar']
		spar_draft = inputs['spar_draft']

		volume = np.pi / 4. * D_spar[0]**2. * L_spar[0] + np.pi / 4. * D_spar[1]**2. * L_spar[1] + np.pi / 4. * D_spar[-1]**2. * (L_spar[-1] - 10.)

		outputs['buoy_spar'] = volume * 1025. * 9.80665
		outputs['CoB'] = (np.pi / 4. * D_spar[0]**2. * L_spar[0] * (-spar_draft + L_spar[0] / 2.) + np.pi / 4. * D_spar[1]**2. * L_spar[1] * (-spar_draft + L_spar[0] + L_spar[1] / 2.) + np.pi / 4. * D_spar[-1]**2. * (-spar_draft + L_spar[0] + L_spar[1] + (L_spar[-1] - 10.) / 2.)) / volume
		
		#TODO: make this more general with respect to number of sections and where they start

	def compute_partials(self, inputs, partials):
		D_spar  = inputs['D_spar']
		L_spar  = inputs['L_spar']
		spar_draft = inputs['spar_draft']

		partials['buoy_spar', 'D_spar'] = np.array([np.pi / 2. * D_spar[0] * L_spar[0], np.pi / 2. * D_spar[1] * L_spar[1], np.pi / 2. * D_spar[-1] * (L_spar[-1] - 10.)]) * 1025. * 9.80665
		partials['buoy_spar', 'L_spar'] = np.array([np.pi / 4. * D_spar[0]**2., np.pi / 4. * D_spar[1]**2., np.pi / 4. * D_spar[-1]**2.]) * 1025. * 9.80665
		partials['buoy_spar', 'spar_draft'] = 0.

		partials['CoB', 'D_spar'] = 0.
		partials['CoB', 'L_spar'] = 0.
		partials['CoB', 'spar_draft'] = 0.

		#TODO