import numpy as np

from openmdao.api import ExplicitComponent

class ViscousDamping(ExplicitComponent):

	def setup(self):
		self.add_input('surge_vel_stddev', val=0., units='m/s')
		self.add_input('pitch_vel_stddev', val=0., units='rad/s')
		self.add_input('bend_vel_stddev', val=0., units='m/s')

		self.add_output('B_visc_11', val=0., units='N*s/m')
		self.add_output('B_visc_15', val=0., units='N*s/m')
		self.add_output('B_visc_17', val=0., units='N*s/m')
		self.add_output('B_visc_55', val=0., units='N*s/m')
		self.add_output('B_visc_57', val=0., units='N*s/m')
		self.add_output('B_visc_77', val=0., units='N*s/m')

	def compute(self, inputs, outputs):
		Cd = inputs['Cd']

		N_elem = 200
		dz = spar_draft / N_elem

		for i in xrange(N_elem):
			z = 
			psi = 
			D = 
			vel_stddev = np.sqrt(surge_vel_stddev**2. + (z * pitch_vel_stddev)**2. + (psi * bend_vel_stddev)**2.)
			outputs['B_visc_11'] += 0.5 * 1025. * Cd * np.sqrt(8./np.pi) * vel_stddev * D * dz
			outputs['B_visc_15'] += 0.5 * 1025. * Cd * np.sqrt(8./np.pi) * vel_stddev * D * z * dz
			outputs['B_visc_17'] += 0.5 * 1025. * Cd * np.sqrt(8./np.pi) * vel_stddev * D * psi * dz
			outputs['B_visc_55'] += 0.5 * 1025. * Cd * np.sqrt(8./np.pi) * vel_stddev * D * z**2. * dz
			outputs['B_visc_57'] += 0.5 * 1025. * Cd * np.sqrt(8./np.pi) * vel_stddev * D * z * psi * dz
			outputs['B_visc_77'] += 0.5 * 1025. * Cd * np.sqrt(8./np.pi) * vel_stddev * D * psi**2. * dz
		