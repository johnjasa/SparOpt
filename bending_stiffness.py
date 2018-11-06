from __future__ import division
import numpy as np
import scipy.interpolate as si

from openmdao.api import ExplicitComponent

class BendingStiffness(ExplicitComponent):

	def setup(self):
		self.add_input('x_moor', val=0., units='m')
		self.add_input('x_d_swl', val=0., units='m/m')
		self.add_input('z_towernode', val=np.zeros(11), units='m')
		self.add_input('x_d_towerelem', val=np.zeros(10), units='m/m')
		self.add_input('x_dd_towerelem', val=np.zeros(10), units='1/m')
		self.add_input('D_spar', val=np.zeros(10), units='m')
		self.add_input('D_tower', val=np.zeros(10), units='m')
		self.add_input('wt_tower', val=np.zeros(10), units='m')
		self.add_input('M_tower', val=np.zeros(10), units='kg')
		self.add_input('Z_tower', val=np.zeros(11), units='m')
		self.add_input('K_moor', val=0., units='N/m')
		self.add_input('M_moor', val=0., units='kg')
		self.add_input('z_moor', val=0., units='m')
		self.add_input('buoy_spar', val=0., units='N')
		self.add_input('CoB', val=0., units='m')
		self.add_input('tot_M_spar', val=0., units='kg')
		self.add_input('tot_M_tower', val=0., units='kg')
		self.add_input('CoG_spar', val=0., units='m')
		self.add_input('M_ball', val=0., units='kg')
		self.add_input('CoG_ball', val=0., units='m')
		self.add_input('M_rotor', val=0., units='kg')
		self.add_input('M_nacelle', val=0., units='kg')

		self.add_output('K17', val=0., units='N/m')
		self.add_output('K57', val=0., units='N*m/m')
		self.add_output('K77', val=0., units='N/m')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		D_spar = inputs['D_spar']
		D_tower = inputs['D_tower']
		wt_tower = inputs['wt_tower']
		M_tower = inputs['M_tower']
		Z_tower = inputs['Z_tower']
		K_moor = inputs['K_moor']
		M_moor = inputs['M_moor']
		z_moor = inputs['z_moor']
		buoy_spar = inputs['buoy_spar']
		CoB = inputs['CoB']
		tot_M_spar = inputs['tot_M_spar']
		tot_M_tower = inputs['tot_M_tower']
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

		outputs['K17'] = K_moor * x_moor
		outputs['K57'] = K_moor * z_moor * x_moor + (buoy_spar * CoB - tot_M_spar * 9.80665 * CoG_spar - M_ball * 9.80665 * CoG_ball - M_moor * 9.80665 * z_moor + 1025. * 9.80665 * np.pi/64. * D_spar[-1]**4.) * x_d_swl
		outputs['K77'] = K_moor * x_moor**2. + (buoy_spar * CoB - tot_M_spar * 9.80665 * CoG_spar - M_ball * 9.80665 * CoG_ball - M_moor * 9.80665 * z_moor + 1025. * 9.80665 * np.pi/64. * D_spar[-1]**4.) * x_d_swl**2.
		
		#TODO: Bending stiffness and damping in spar currently neglected as it is assumed to be rigid

		EI_tower = np.pi / 64. * (D_tower**4. - (D_tower - 2. * wt_tower)**4.) * 2.1e11

		EI = 0.

		N_elem = len(x_d_towerelem)

		for i in xrange(N_elem):
			z = (z_towernode[i] + z_towernode[i+1]) / 2
			dz = z_towernode[i+1] - z_towernode[i]
			for j in xrange(len(Z_tower)-1):
				if (z < Z_tower[j+1]) and (z >= Z_tower[j]):
					EI = EI_tower[j]
					break

			outputs['K57'] += (-M_nacelle - M_rotor - tot_M_tower + np.sum(M_tower[:i])) * 9.80665 * x_d_towerelem[i] * dz
			outputs['K77'] += dz * EI * x_dd_towerelem[i]**2. + (-M_nacelle - M_rotor - tot_M_tower + np.sum(M_tower[:i])) * 9.80665 * x_d_towerelem[i]**2. * dz

	def compute_partials(self, inputs, partials):
		D_spar = inputs['D_spar']
		D_tower = inputs['D_tower']
		wt_tower = inputs['wt_tower']
		M_tower = inputs['M_tower']
		Z_tower = inputs['Z_tower']
		K_moor = inputs['K_moor']
		M_moor = inputs['M_moor']
		z_moor = inputs['z_moor']
		buoy_spar = inputs['buoy_spar']
		CoB = inputs['CoB']
		tot_M_spar = inputs['tot_M_spar']
		tot_M_tower = inputs['tot_M_tower']
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

		partials['K17', 'x_moor'] = K_moor
		partials['K17', 'K_moor'] = x_moor

		partials['K57', 'x_moor'] = K_moor * z_moor
		partials['K57', 'x_d_swl'] = buoy_spar * CoB - tot_M_spar * 9.80665 * CoG_spar - M_ball * 9.80665 * CoG_ball - M_moor * 9.80665 * z_moor + 1025. * 9.80665 * np.pi/64. * D_spar[-1]**4.
		partials['K57', 'D_spar'] = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 1025. * 9.80665 * np.pi/16. * D_spar[-1]**3. * x_d_swl]])
		partials['K57', 'K_moor'] = z_moor * x_moor
		partials['K57', 'M_moor'] = -9.80665 * z_moor * x_d_swl
		partials['K57', 'z_moor'] = K_moor * x_moor - M_moor * 9.80665 * x_d_swl
		partials['K57', 'buoy_spar'] = CoB * x_d_swl
		partials['K57', 'CoB'] = buoy_spar * x_d_swl
		partials['K57', 'tot_M_spar'] = -9.80665 * CoG_spar * x_d_swl
		partials['K57', 'CoG_spar'] = -tot_M_spar * 9.80665 * x_d_swl
		partials['K57', 'M_ball'] = -9.80665 * CoG_ball * x_d_swl
		partials['K57', 'CoG_ball'] = -M_ball * 9.80665 * x_d_swl
		
		partials['K77', 'x_moor'] = 2. * K_moor * x_moor
		partials['K77', 'x_d_swl'] = 2. * (buoy_spar * CoB - tot_M_spar * 9.80665 * CoG_spar - M_ball * 9.80665 * CoG_ball - M_moor * 9.80665 * z_moor + 1025. * 9.80665 * np.pi/64. * D_spar[-1]**4.) * x_d_swl
		partials['K77', 'D_spar'] = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 1025. * 9.80665 * np.pi/16. * D_spar[-1]**3. * x_d_swl**2.]])
		partials['K77', 'K_moor'] = x_moor**2.
		partials['K77', 'M_moor'] = -9.80665 * z_moor * x_d_swl**2.
		partials['K77', 'z_moor'] = -M_moor * 9.80665 * x_d_swl**2.
		partials['K77', 'buoy_spar'] = CoB * x_d_swl**2.
		partials['K77', 'CoB'] = buoy_spar * x_d_swl**2.
		partials['K77', 'tot_M_spar'] = -9.80665 * CoG_spar * x_d_swl**2.
		partials['K77', 'CoG_spar'] = -tot_M_spar * 9.80665 * x_d_swl**2.
		partials['K77', 'M_ball'] = -9.80665 * CoG_ball * x_d_swl**2.
		partials['K77', 'CoG_ball'] = -M_ball * 9.80665 * x_d_swl**2.

		partials['K57', 'M_rotor'] = 0.
		partials['K57', 'M_nacelle'] = 0.
		partials['K57', 'tot_M_tower'] = 0.
		partials['K57', 'D_tower'] = np.zeros((1,10))
		partials['K57', 'wt_tower'] = np.zeros((1,10))
		partials['K57', 'M_tower'] = np.zeros((1,10))
		partials['K57', 'z_towernode'] = np.zeros((1,11))
		partials['K57', 'x_d_towerelem'] = np.zeros((1,10))
		partials['K57', 'x_dd_towerelem'] = np.zeros((1,10))

		partials['K77', 'M_rotor'] = 0.
		partials['K77', 'M_nacelle'] = 0.
		partials['K77', 'tot_M_tower'] = 0.
		partials['K77', 'D_tower'] = np.zeros((1,10))
		partials['K77', 'wt_tower'] = np.zeros((1,10))
		partials['K77', 'M_tower'] = np.zeros((1,10))
		partials['K77', 'z_towernode'] = np.zeros((1,11))
		partials['K77', 'x_d_towerelem'] = np.zeros((1,10))
		partials['K77', 'x_dd_towerelem'] = np.zeros((1,10))

		EI_tower = np.pi / 64. * (D_tower**4. - (D_tower - 2. * wt_tower)**4.) * 2.1e11

		EI = 0.

		N_elem = len(x_d_towerelem)

		for i in xrange(N_elem):
			z = (z_towernode[i] + z_towernode[i+1]) / 2
			dz = z_towernode[i+1] - z_towernode[i]
			for j in xrange(len(Z_tower)-1):
				if (z < Z_tower[j+1]) and (z >= Z_tower[j]):
					EI = EI_tower[j]
					partials['K77', 'D_tower'][0,j] += dz * x_dd_towerelem[i]**2. * np.pi / 16. * (D_tower[j]**3. - (D_tower[j] - 2. * wt_tower[j])**3.) * 2.1e11
					partials['K77', 'wt_tower'][0,j] += dz * x_dd_towerelem[i]**2. * np.pi / 8. * (D_tower[j] - 2. * wt_tower[j])**3. * 2.1e11
					break

			partials['K77', 'x_dd_towerelem'][0,i] += 2. * dz * EI * x_dd_towerelem[i]
			partials['K77', 'z_towernode'][0,i] += -EI * x_dd_towerelem[i]**2.
			partials['K77', 'z_towernode'][0,i+1] += EI * x_dd_towerelem[i]**2.

			partials['K57', 'M_rotor'] += -9.80665 * x_d_towerelem[i] * dz
			partials['K57', 'M_nacelle'] += -9.80665 * x_d_towerelem[i] * dz
			partials['K57', 'tot_M_tower'] += -9.80665 * x_d_towerelem[i] * dz
			partials['K57', 'x_d_towerelem'][0,i] += (-M_nacelle - M_rotor - tot_M_tower + np.sum(M_tower[:i])) * 9.80665 * dz
			partials['K57', 'z_towernode'][0,i] += -(-M_nacelle - M_rotor - tot_M_tower + np.sum(M_tower[:i])) * 9.80665 * x_d_towerelem[i]
			partials['K57', 'z_towernode'][0,i+1] += (-M_nacelle - M_rotor - tot_M_tower + np.sum(M_tower[:i])) * 9.80665 * x_d_towerelem[i]

			partials['K77', 'M_rotor'] += -9.80665 * x_d_towerelem[i]**2. * dz
			partials['K77', 'M_nacelle'] += -9.80665 * x_d_towerelem[i]**2. * dz
			partials['K77', 'tot_M_tower'] += -9.80665 * x_d_towerelem[i]**2. * dz
			partials['K77', 'x_d_towerelem'][0,i] += 2. * (-M_nacelle - M_rotor - tot_M_tower + np.sum(M_tower[:i])) * 9.80665 * x_d_towerelem[i] * dz
			partials['K77', 'z_towernode'][0,i] += -(-M_nacelle - M_rotor - tot_M_tower + np.sum(M_tower[:i])) * 9.80665 * x_d_towerelem[i]**2.
			partials['K77', 'z_towernode'][0,i+1] += (-M_nacelle - M_rotor - tot_M_tower + np.sum(M_tower[:i])) * 9.80665 * x_d_towerelem[i]**2.

			for k in xrange(i):
				partials['K57', 'M_tower'][0,k] += 9.80665 * x_d_towerelem[i] * dz
				partials['K77', 'M_tower'][0,k] += 9.80665 * x_d_towerelem[i]**2. * dz