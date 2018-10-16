import numpy as np
import scipy.interpolate as si

from openmdao.api import ExplicitComponent

class BendingMass(ExplicitComponent):

	def setup(self):
		self.add_input('z_sparmode', val=np.zeros(7), units='m')
		self.add_input('x_sparmode', val=np.zeros(7), units='m')
		self.add_input('z_towermode', val=np.zeros(11), units='m')
		self.add_input('x_towermode', val=np.zeros(11), units='m')
		self.add_input('M_secs', val=np.zeros(3), units='kg')
		self.add_input('L_secs', val=np.zeros(3), units='m')
		self.add_input('Z_spar', val=np.zeros(4), units='m')
		self.add_input('spar_draft', val=0., units='m')
		self.add_input('M_tower', val=np.zeros(10), units='kg')
		self.add_input('L_tower', val=np.zeros(10), units='m')
		self.add_input('Z_tower', val=np.zeros(11), units='m')
		self.add_input('M_ball', val=0., units='kg')
		self.add_input('L_ball', val=0., units='m')
		self.add_input('wt_ball', val=0., units='m')
		self.add_input('M_rotor', val=0., units='kg')
		self.add_input('M_nacelle', val=0., units='kg')
		self.add_input('I_rotor', val=0., units='kg*m**2')

		self.add_output('M17', val=0., units='kg')
		self.add_output('M57', val=0., units='kg*m')
		self.add_output('M77', val=0., units='kg')

	def compute(self, inputs, outputs):
		M_secs = inputs['M_secs']
		L_secs = inputs['L_secs']
		Z_spar = inputs['Z_spar']
		M_tower = inputs['M_tower']
		L_tower = inputs['L_tower']
		Z_tower = inputs['Z_tower']
		M_ball = inputs['M_ball']
		L_ball = inputs['L_ball']
		wt_ball = inputs['wt_ball']
		M_rotor = inputs['M_rotor']
		M_nacelle = inputs['M_nacelle']
		I_rotor = inputs['I_rotor']
		z_ball = -inputs['spar_draft'] + inputs['wt_ball'] + inputs['L_ball'] #top of ballast

		f_psi_spar = si.UnivariateSpline(inputs['z_sparmode'], inputs['x_sparmode'], s=0)
		f_psi_tower = si.UnivariateSpline(inputs['z_towermode'], inputs['x_towermode'], s=0)
		f_psi_d_tower = f_psi_tower.derivative(n=1)

		m_elem_tower = M_tower / L_tower
		m_elem_spar = M_secs / L_secs
		m_elem_ball = M_ball / L_ball

		outputs['M17'] = (M_rotor + M_nacelle) * f_psi_tower(Z_tower[-1])
		outputs['M57'] = (M_rotor + M_nacelle) * Z_tower[-1] * f_psi_tower(Z_tower[-1]) + I_rotor * f_psi_d_tower(Z_tower[-1])
		outputs['M77'] = (M_rotor + M_nacelle) * f_psi_tower(Z_tower[-1])**2. + I_rotor * f_psi_d_tower(Z_tower[-1])**2.

		m = 0.

		N_elem = 200

		for i in xrange(N_elem):
			z = inputs['spar_draft'] + i / N_elem * np.sum(L_secs)
			dz = np.sum(L_secs) / N_elem
			for j in xrange(len(Z_spar) - 1):
				if (z < Z_spar[j+1]) and (z >= Z_spar[j]):
					m = m_elem_spar[j]
				if z < z_ball:
					m += m_elem_ball
			outputs['M17'] += dz * m * f_psi_spar(z)
			outputs['M57'] += dz * m * z * f_psi_spar(z)
			outputs['M77'] += dz * m * f_psi_spar(z)**2.

		for i in xrange(N_elem):
			z = 10. + i / N_elem * np.sum(L_tower)
			dz = np.sum(L_tower) / N_elem
			for j in xrange(len(Z_tower) - 1):
				if (z < Z_tower[j+1]) and (z >= Z_tower[j]):
					m = m_elem_tower[j]
			outputs['M17'] += dz * m * f_psi_tower(z)
			outputs['M57'] += dz * m * z * f_psi_tower(z)
			outputs['M77'] += dz * m * f_psi_tower(z)**2.