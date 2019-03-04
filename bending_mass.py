from __future__ import division
import numpy as np

from openmdao.api import ExplicitComponent

class BendingMass(ExplicitComponent):

	def setup(self):
		self.add_input('z_sparnode', val=np.zeros(13), units='m')
		self.add_input('x_sparelem', val=np.zeros(12), units='m')
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
		self.add_input('L_ball_elem', val=np.zeros(10), units='m')
		self.add_input('M_ball_elem', val=np.zeros(10), units='kg')
		self.add_input('L_ball', val=0., units='m')
		self.add_input('M_rotor', val=0., units='kg')
		self.add_input('M_nacelle', val=0., units='kg')
		self.add_input('I_rotor', val=0., units='kg*m**2')

		self.add_output('M17', val=0., units='kg')
		self.add_output('M57', val=0., units='kg*m')
		self.add_output('M77', val=0., units='kg')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		M_spar = inputs['M_spar']
		L_spar = inputs['L_spar']
		Z_spar = inputs['Z_spar']
		M_tower = inputs['M_tower']
		L_tower = inputs['L_tower']
		Z_tower = inputs['Z_tower']
		L_ball_elem = inputs['L_ball_elem']
		M_ball_elem = inputs['M_ball_elem']
		L_ball = inputs['L_ball']
		M_rotor = inputs['M_rotor']
		M_nacelle = inputs['M_nacelle']
		I_rotor = inputs['I_rotor']
		z_ball = -inputs['spar_draft'] + inputs['L_ball']
		z_sparnode = inputs['z_sparnode']
		x_sparelem = inputs['x_sparelem']
		z_towernode = inputs['z_towernode']
		x_towerelem = inputs['x_towerelem']
		x_d_towertop = inputs['x_d_towertop']

		m_elem_tower = M_tower / L_tower
		m_elem_spar = M_spar / L_spar

		outputs['M17'] = (M_rotor + M_nacelle)
		outputs['M57'] = (M_rotor + M_nacelle) * z_towernode[-1] + I_rotor * x_d_towertop
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
					break
			if z <= z_ball:
				m += M_ball_elem[j] / L_ball_elem[j]
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

	def compute_partials(self, inputs, partials):
		M_spar = inputs['M_spar']
		L_spar = inputs['L_spar']
		Z_spar = inputs['Z_spar']
		M_tower = inputs['M_tower']
		L_tower = inputs['L_tower']
		Z_tower = inputs['Z_tower']
		L_ball_elem = inputs['L_ball_elem']
		M_ball_elem = inputs['M_ball_elem']
		L_ball = inputs['L_ball']
		M_rotor = inputs['M_rotor']
		M_nacelle = inputs['M_nacelle']
		I_rotor = inputs['I_rotor']
		z_ball = -inputs['spar_draft'] + inputs['L_ball']
		z_sparnode = inputs['z_sparnode']
		x_sparelem = inputs['x_sparelem']
		z_towernode = inputs['z_towernode']
		x_towerelem = inputs['x_towerelem']
		x_d_towertop = inputs['x_d_towertop']

		m_elem_tower = M_tower / L_tower
		m_elem_spar = M_spar / L_spar

		partials['M17', 'z_sparnode'] = np.zeros((1,13))
		partials['M17', 'x_sparelem'] = np.zeros((1,12))
		partials['M17', 'z_towernode'] = np.zeros((1,11))
		partials['M17', 'x_towerelem'] = np.zeros((1,10))
		partials['M17', 'x_d_towertop'] = 0.
		partials['M17', 'M_spar'] = np.zeros((1,10))
		partials['M17', 'L_spar'] = np.zeros((1,10))
		partials['M17', 'M_tower'] = np.zeros((1,10))
		partials['M17', 'L_tower'] = np.zeros((1,10))
		partials['M17', 'L_ball_elem'] = np.zeros((1,10))
		partials['M17', 'M_ball_elem'] = np.zeros((1,10))
		partials['M17', 'L_ball'] = 0.
		partials['M17', 'M_rotor'] = 1.
		partials['M17', 'M_nacelle'] = 1.
		partials['M17', 'I_rotor'] = 0.

		partials['M57', 'z_sparnode'] = np.zeros((1,13))
		partials['M57', 'x_sparelem'] = np.zeros((1,12))
		partials['M57', 'z_towernode'] = np.zeros((1,11))
		partials['M57', 'z_towernode'][0,-1] += M_rotor + M_nacelle
		partials['M57', 'x_towerelem'] = np.zeros((1,10))
		partials['M57', 'x_d_towertop'] = I_rotor
		partials['M57', 'M_spar'] = np.zeros((1,10))
		partials['M57', 'L_spar'] = np.zeros((1,10))
		partials['M57', 'M_tower'] = np.zeros((1,10))
		partials['M57', 'L_tower'] = np.zeros((1,10))
		partials['M57', 'L_ball_elem'] = np.zeros((1,10))
		partials['M57', 'M_ball_elem'] = np.zeros((1,10))
		partials['M57', 'L_ball'] = 0.
		partials['M57', 'M_rotor'] = z_towernode[-1]
		partials['M57', 'M_nacelle'] = z_towernode[-1]
		partials['M57', 'I_rotor'] = x_d_towertop

		partials['M77', 'z_sparnode'] = np.zeros((1,13))
		partials['M77', 'x_sparelem'] = np.zeros((1,12))
		partials['M77', 'z_towernode'] = np.zeros((1,11))
		partials['M77', 'x_towerelem'] = np.zeros((1,10))
		partials['M77', 'x_d_towertop'] = 2. * I_rotor * x_d_towertop
		partials['M77', 'M_spar'] = np.zeros((1,10))
		partials['M77', 'L_spar'] = np.zeros((1,10))
		partials['M77', 'spar_draft'] = 0.
		partials['M77', 'M_tower'] = np.zeros((1,10))
		partials['M77', 'L_tower'] = np.zeros((1,10))
		partials['M77', 'L_ball_elem'] = np.zeros((1,10))
		partials['M77', 'M_ball_elem'] = np.zeros((1,10))
		partials['M77', 'L_ball'] = 0.
		partials['M77', 'M_rotor'] = 1.
		partials['M77', 'M_nacelle'] = 1.
		partials['M77', 'I_rotor'] = x_d_towertop**2.

		m = 0.

		N_elem_spar = len(x_sparelem)
		N_elem_tower = len(x_towerelem)
		
		for i in xrange(N_elem_spar):
			z = (z_sparnode[i] + z_sparnode[i+1]) / 2
			dz = z_sparnode[i+1] - z_sparnode[i]
			for j in xrange(len(Z_spar) - 1):
				if (z < Z_spar[j+1]) and (z >= Z_spar[j]):
					m = m_elem_spar[j]
					partials['M17', 'M_spar'][0,j] += dz * x_sparelem[i] * 1. / L_spar[j]
					partials['M17', 'L_spar'][0,j] += -dz * x_sparelem[i] * M_spar[j] / L_spar[j]**2.
					partials['M57', 'M_spar'][0,j] += dz * z * x_sparelem[i] * 1. / L_spar[j]
					partials['M57', 'L_spar'][0,j] += -dz * z * x_sparelem[i] * M_spar[j] / L_spar[j]**2.
					partials['M77', 'M_spar'][0,j] += dz * x_sparelem[i]**2. * 1. / L_spar[j]
					partials['M77', 'L_spar'][0,j] += -dz * x_sparelem[i]**2. * M_spar[j] / L_spar[j]**2.
					break
			
			if z <= z_ball:
				m += M_ball_elem[j] / L_ball_elem[j]
				partials['M17', 'M_ball_elem'][0,j] += dz * x_sparelem[i] * 1. / L_ball_elem[j]
				partials['M17', 'L_ball_elem'][0,j] += -dz * x_sparelem[i] * M_ball_elem[j] / L_ball_elem[j]**2.
				partials['M57', 'M_ball_elem'][0,j] += dz * z * x_sparelem[i] * 1. / L_ball_elem[j]
				partials['M57', 'L_ball_elem'][0,j] += -dz * z * x_sparelem[i] * M_ball_elem[j] / L_ball_elem[j]**2.
				partials['M77', 'M_ball_elem'][0,j] += dz * x_sparelem[i]**2. * 1. / L_ball_elem[j]
				partials['M77', 'L_ball_elem'][0,j] += -dz * x_sparelem[i]**2. * M_ball_elem[j] / L_ball_elem[j]**2.

			partials['M17', 'z_sparnode'][0,i] += -m * x_sparelem[i]
			partials['M17', 'z_sparnode'][0,i+1] += m * x_sparelem[i]
			partials['M17', 'x_sparelem'][0,i] += dz * m
			partials['M57', 'z_sparnode'][0,i] += -m * z * x_sparelem[i] + dz * m * x_sparelem[i] * 0.5
			partials['M57', 'z_sparnode'][0,i+1] += m * z * x_sparelem[i] + dz * m * x_sparelem[i] * 0.5
			partials['M57', 'x_sparelem'][0,i] += dz * m * z
			partials['M77', 'z_sparnode'][0,i] += -m * x_sparelem[i]**2.
			partials['M77', 'z_sparnode'][0,i+1] += m * x_sparelem[i]**2.
			partials['M77', 'x_sparelem'][0,i] += 2. * x_sparelem[i] * dz * m

		for i in xrange(N_elem_tower):
			z = (z_towernode[i] + z_towernode[i+1]) / 2
			dz = z_towernode[i+1] - z_towernode[i]
			for j in xrange(len(Z_tower) - 1):
				if (z < Z_tower[j+1]) and (z >= Z_tower[j]):
					m = m_elem_tower[j]
					partials['M17', 'M_tower'][0,j] += dz * x_towerelem[i] * 1. / L_tower[j]
					partials['M17', 'L_tower'][0,j] += -dz * x_towerelem[i] * M_tower[j] / L_tower[j]**2.
					partials['M57', 'M_tower'][0,j] += dz * z * x_towerelem[i] * 1. / L_tower[j]
					partials['M57', 'L_tower'][0,j] += -dz * z * x_towerelem[i] * M_tower[j] / L_tower[j]**2.
					partials['M77', 'M_tower'][0,j] += dz * x_towerelem[i]**2. * 1. / L_tower[j]
					partials['M77', 'L_tower'][0,j] += -dz * x_towerelem[i]**2. * M_tower[j] / L_tower[j]**2.
					break

			partials['M17', 'z_towernode'][0,i] += -m * x_towerelem[i]
			partials['M17', 'z_towernode'][0,i+1] += m * x_towerelem[i]
			partials['M17', 'x_towerelem'][0,i] += dz * m
			partials['M57', 'z_towernode'][0,i] += -m * z * x_towerelem[i] + dz * m * x_towerelem[i] * 0.5
			partials['M57', 'z_towernode'][0,i+1] += m * z * x_towerelem[i] + dz * m * x_towerelem[i] * 0.5
			partials['M57', 'x_towerelem'][0,i] += dz * m * z
			partials['M77', 'z_towernode'][0,i] += -m * x_towerelem[i]**2.
			partials['M77', 'z_towernode'][0,i+1] += m * x_towerelem[i]**2.
			partials['M77', 'x_towerelem'][0,i] += 2. * x_towerelem[i] * dz * m