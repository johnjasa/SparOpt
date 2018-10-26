from __future__ import division
import numpy as np

from openmdao.api import ExplicitComponent

class BendingMass(ExplicitComponent):

	def setup(self):
		self.add_input('z_sparnode', val=np.zeros(14), units='m')
		self.add_input('x_sparelem', val=np.zeros(13), units='m')
		self.add_input('z_towernode', val=np.zeros(11), units='m')
		self.add_input('x_towerelem', val=np.zeros(10), units='m')
		self.add_input('x_d_towertop', val=0., units='m/m')
		self.add_input('M_spar', val=np.zeros(10), units='kg')
		self.add_input('L_spar', val=np.zeros(10), units='m')
		self.add_input('Z_spar', val=np.zeros(11), units='m')
		self.add_input('spar_draft', val=0., units='m')
		self.add_input('M_tower', val=np.zeros(10), units='kg')
		self.add_input('L_tower', val=np.zeros(10), units='m')
		self.add_input('Z_tower', val=np.zeros(11), units='m')
		self.add_input('M_ball', val=0., units='kg')
		self.add_input('L_ball', val=0., units='m')
		self.add_input('M_rotor', val=0., units='kg')
		self.add_input('M_nacelle', val=0., units='kg')
		self.add_input('I_rotor', val=0., units='kg*m**2')

		self.add_output('M17', val=0., units='kg')
		self.add_output('M57', val=0., units='kg*m')
		self.add_output('M77', val=0., units='kg')

	def compute(self, inputs, outputs):
		M_spar = inputs['M_spar']
		L_spar = inputs['L_spar']
		Z_spar = inputs['Z_spar']
		M_tower = inputs['M_tower']
		L_tower = inputs['L_tower']
		Z_tower = inputs['Z_tower']
		M_ball = inputs['M_ball']
		L_ball = inputs['L_ball']
		M_rotor = inputs['M_rotor']
		M_nacelle = inputs['M_nacelle']
		I_rotor = inputs['I_rotor']
		z_ball = -inputs['spar_draft'] + inputs['L_ball'] #top of ballast
		z_sparnode = inputs['z_sparnode']
		x_sparelem = inputs['x_sparelem']
		z_towernode = inputs['z_towernode']
		x_towerelem = inputs['x_towerelem']
		x_d_towertop = inputs['x_d_towertop']

		m_elem_tower = M_tower / L_tower
		m_elem_spar = M_spar / L_spar
		m_elem_ball = M_ball / L_ball

		outputs['M17'] = (M_rotor + M_nacelle)
		outputs['M57'] = (M_rotor + M_nacelle) * Z_tower[-1] + I_rotor * x_d_towertop
		outputs['M77'] = (M_rotor + M_nacelle) + I_rotor * x_d_towertop**2.

		m = 0.

		N_elem_spar = len(x_sparelem)
		N_elem_tower = len(x_towerelem)
		
		for i in xrange(N_elem_spar):
			z = (z_sparnode[i] + z_sparnode[i+1]) / 2
			dz = z_sparnode[i+1] - z_sparnode[i]
			for j in xrange(len(Z_spar) - 1):
				if (z < Z_spar[j+1]) and (z >= Z_spar[j]):
					m = m_elem_spar[j]
			if z <= z_ball:
				m += m_elem_ball
			outputs['M17'] += dz * m * x_sparelem[i]
			outputs['M57'] += dz * m * z * x_sparelem[i]
			outputs['M77'] += dz * m * x_sparelem[i]**2.

		for i in xrange(N_elem_tower):
			z = (z_towernode[i] + z_towernode[i+1]) / 2
			dz = z_towernode[i+1] - z_towernode[i]
			for j in xrange(len(Z_tower) - 1):
				if (z < Z_tower[j+1]) and (z >= Z_tower[j]):
					m = m_elem_tower[j]
					break

			outputs['M17'] += dz * m * x_towerelem[i]
			outputs['M57'] += dz * m * z * x_towerelem[i]
			outputs['M77'] += dz * m * x_towerelem[i]**2.