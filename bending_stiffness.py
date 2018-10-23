from __future__ import division
import numpy as np
import scipy.interpolate as si

from openmdao.api import ExplicitComponent

class BendingStiffness(ExplicitComponent):

	def setup(self):
		self.add_input('x_moor', val=0., units='m')
		self.add_input('x_d_swl', val=0., units='m/m')
		self.add_input('x_sparnode', val=np.zeros(14), units='m')
		self.add_input('z_towernode', val=np.zeros(11), units='m')
		self.add_input('x_d_towerelem', val=np.zeros(10), units='m/m')
		self.add_input('x_dd_towerelem', val=np.zeros(10), units='1/m')
		self.add_input('D_spar', val=np.zeros(10), units='m')
		self.add_input('D_tower', val=np.zeros(10), units='m')
		self.add_input('wt_tower', val=np.zeros(10), units='m')
		self.add_input('M_tower', val=np.zeros(10), units='kg')
		self.add_input('L_tower', val=np.zeros(10), units='m')
		self.add_input('Z_tower', val=np.zeros(11), units='m')
		self.add_input('K_moor', val=0., units='N/m')
		self.add_input('M_moor', val=0., units='kg')
		self.add_input('z_moor', val=0., units='m')
		self.add_input('buoy_spar', val=0., units='N')
		self.add_input('CoB', val=0., units='m')
		self.add_input('M_spar', val=np.zeros(10), units='kg')
		self.add_input('CoG_spar', val=0., units='m')
		self.add_input('M_ball', val=0., units='kg')
		self.add_input('CoG_ball', val=0., units='m')
		self.add_input('M_rotor', val=0., units='kg')
		self.add_input('M_nacelle', val=0., units='kg')

		self.add_output('K17', val=0., units='N/m')
		self.add_output('K57', val=0., units='N*m/m')
		self.add_output('K77', val=0., units='N/m')

	def compute(self, inputs, outputs):
		D_spar = inputs['D_spar']
		D_tower = inputs['D_tower']
		wt_tower = inputs['wt_tower']
		M_tower = inputs['M_tower']
		L_tower = inputs['L_tower']
		Z_tower = inputs['Z_tower']
		K_moor = inputs['K_moor']
		M_moor = inputs['M_moor']
		z_moor = inputs['z_moor']
		buoy_spar = inputs['buoy_spar']
		CoB = inputs['CoB']
		M_spar = inputs['M_spar']
		CoG_spar = inputs['CoG_spar']
		M_ball = inputs['M_ball']
		CoG_ball = inputs['CoG_ball']
		M_rotor = inputs['M_rotor']
		M_nacelle = inputs['M_nacelle']
		z_towernode = inputs['z_towernode']
		x_d_towerelem = inputs['x_d_towerelem']
		x_dd_towerelem = inputs['x_dd_towerelem']
		x_moor = inputs['x_moor']
		x_d_swl = inputs['x_d_swl']

		m_elem_tower = M_tower / L_tower
		EI_tower = np.pi / 64. * (D_tower**4. - (D_tower - 2. * wt_tower)**4.) * 2.1e11

		outputs['K17'] = K_moor * x_moor**2.
		outputs['K57'] = K_moor * z_moor * x_moor + (buoy_spar * CoB - np.sum(M_spar) * 9.80665 * CoG_spar - M_ball * 9.80665 * CoG_ball - M_moor * 9.80665 * z_moor + 1025. * 9.80665 * np.pi/64. * D_spar[-1]**4.) * x_d_swl
		outputs['K77'] = (buoy_spar * CoB - np.sum(M_spar) * 9.80665 * CoG_spar - M_ball * 9.80665 * CoG_ball - M_moor * 9.80665 * z_moor + 1025. * 9.80665 * np.pi/64. * D_spar[-1]**4.) * x_d_swl**2. + K_moor * x_moor**2.
		
		#TODO: Bending stiffness and damping in spar currently neglected as it is assumed to be rigid

		m = 0.
		EI = 0.
		accum_mass = 0.

		N_elem = len(x_d_towerelem)

		for i in xrange(N_elem):
			z = (z_towernode[i] + z_towernode[i+1]) / 2
			dz = z_towernode[i+1] - z_towernode[i]
			for j in xrange(len(Z_tower)-1):
				if (z < Z_tower[j+1]) and (z >= Z_tower[j]):
					m = m_elem_tower[j]
					EI = EI_tower[j]

			outputs['K77'] += dz * EI * x_dd_towerelem[i]**2. - (M_rotor + M_nacelle + np.sum(M_tower) - accum_mass) * 9.80665 * x_d_towerelem[i]**2. * dz
			outputs['K57'] += -(M_rotor + M_nacelle + np.sum(M_tower) - accum_mass) * 9.80665 * x_d_towerelem[i] * dz

			accum_mass += dz * m