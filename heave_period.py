import numpy as np

from openmdao.api import ExplicitComponent

class HeavePeriod(ExplicitComponent):

	def setup(self):
		self.add_input('tot_M_spar', val=0., units='kg')
		self.add_input('M_turb', val=0., units='kg')
		self.add_input('M_ball', val=0., units='kg')
		self.add_input('D_spar', val=np.zeros(10), units='m')

		self.add_output('T_heave', val=0., units='s')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		tot_M_spar = inputs['tot_M_spar']
		M_turb = inputs['M_turb']
		M_ball = inputs['M_ball']
		D_spar = inputs['D_spar']

		M = tot_M_spar + M_turb + M_ball
		A33 = 1025. * 2. / np.pi * np.pi / 6. * D_spar[0]**3. #assume 3D circular disc from DNV-RP-H103
		C = 1025. * 9.80665 * np.pi / 4. * D_spar[-1]**2.

		outputs['T_heave'] = 2. * np.pi * np.sqrt((M + A33) / C)

	def compute_partials(self, inputs, partials):
		tot_M_spar = inputs['tot_M_spar']
		M_turb = inputs['M_turb']
		M_ball = inputs['M_ball']
		D_spar = inputs['D_spar']

		M = tot_M_spar + M_turb + M_ball
		A33 = 1025. * 2. / np.pi * np.pi / 6. * D_spar[0]**3.
		C = 1025. * 9.80665 * np.pi / 4. * D_spar[-1]**2.

		partials['T_heave', 'tot_M_spar'] = 2. * np.pi * 0.5 / np.sqrt((M + A33) / C) * 1. / C
		partials['T_heave', 'M_turb'] = 2. * np.pi * 0.5 / np.sqrt((M + A33) / C) * 1. / C
		partials['T_heave', 'M_ball'] = 2. * np.pi * 0.5 / np.sqrt((M + A33) / C) * 1. / C

		partials['T_heave', 'D_spar'][0,0] = 2. * np.pi * 0.5 / np.sqrt((M + A33) / C) * 1025. * 2. / np.pi * np.pi / 2. * D_spar[0]**2. / C
		partials['T_heave', 'D_spar'][0,-1] = 2. * np.pi * 0.5 / np.sqrt((M + A33) / C) * (-((M + A33) / C**2.) * 1025. * 9.80665 * np.pi / 2. * D_spar[-1])