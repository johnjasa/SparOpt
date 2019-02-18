from __future__ import division
import numpy as np
import scipy.interpolate as si

from openmdao.api import ExplicitComponent

class BendingStiffness(ExplicitComponent):

	def setup(self):
		self.add_input('K_moor', val=0., units='N/m')
		self.add_input('z_moor', val=0., units='m')
		self.add_input('x_moor', val=0., units='m')
		self.add_input('z_sparnode', val=np.zeros(14), units='m')
		self.add_input('z_towernode', val=np.zeros(11), units='m')
		self.add_input('x_d_sparelem', val=np.zeros(13), units='m/m')
		self.add_input('x_dd_sparelem', val=np.zeros(13), units='1/m')
		self.add_input('x_d_towerelem', val=np.zeros(10), units='m/m')
		self.add_input('x_dd_towerelem', val=np.zeros(10), units='1/m')
		self.add_input('normforce_mode_elem', val=np.zeros(23), units='N')
		self.add_input('EI_mode_elem', val=np.zeros(23), units='N*m**2')

		self.add_output('K17', val=0., units='N/m')
		self.add_output('K57', val=0., units='N*m/m')
		self.add_output('K77', val=0., units='N/m')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		K_moor = inputs['K_moor']
		z_moor = inputs['z_moor']
		x_moor = inputs['x_moor']
		z_sparnode = inputs['z_sparnode']
		z_towernode = inputs['z_towernode']
		x_d_sparelem = inputs['x_d_sparelem']
		x_dd_sparelem = inputs['x_dd_sparelem']
		x_d_towerelem = inputs['x_d_towerelem']
		x_dd_towerelem = inputs['x_dd_towerelem']
		norm_force = inputs['normforce_mode_elem']
		EI = inputs['EI_mode_elem']

		outputs['K17'] = K_moor * x_moor
		outputs['K57'] = K_moor * z_moor * x_moor
		outputs['K77'] = K_moor * x_moor**2.

		N_sparelem = len(x_d_sparelem)
		N_towerelem = len(x_d_towerelem)

		for i in xrange(N_sparelem):
			dz = z_sparnode[i+1] - z_sparnode[i]

			x_dd_sparelem[i] = 0.

			outputs['K57'] += norm_force[i] * x_d_sparelem[i] * dz
			outputs['K77'] += dz * EI[i] * x_dd_sparelem[i]**2. + norm_force[i] * x_d_sparelem[i]**2. * dz

		for i in xrange(N_towerelem):
			dz = z_towernode[i+1] - z_towernode[i]

			outputs['K57'] += norm_force[N_sparelem+i] * x_d_towerelem[i] * dz
			outputs['K77'] += dz * EI[N_sparelem+i] * x_dd_towerelem[i]**2. + norm_force[N_sparelem+i] * x_d_towerelem[i]**2. * dz

	def compute_partials(self, inputs, partials):
		K_moor = inputs['K_moor']
		z_moor = inputs['z_moor']
		x_moor = inputs['x_moor']
		z_sparnode = inputs['z_sparnode']
		z_towernode = inputs['z_towernode']
		x_d_sparelem = inputs['x_d_sparelem']
		x_dd_sparelem = inputs['x_dd_sparelem']
		x_d_towerelem = inputs['x_d_towerelem']
		x_dd_towerelem = inputs['x_dd_towerelem']
		norm_force = inputs['normforce_mode_elem']
		EI = inputs['EI_mode_elem']

		partials['K17', 'x_moor'] = K_moor
		partials['K17', 'K_moor'] = x_moor

		partials['K57', 'x_moor'] = K_moor * z_moor
		partials['K57', 'K_moor'] = z_moor * x_moor
		partials['K57', 'z_moor'] = K_moor * x_moor
		
		partials['K77', 'x_moor'] = 2. * K_moor * x_moor
		partials['K77', 'K_moor'] = x_moor**2.

		partials['K57', 'z_sparnode'] = np.zeros((1,14))
		partials['K57', 'x_d_sparelem'] = np.zeros((1,13))
		partials['K57', 'z_towernode'] = np.zeros((1,11))
		partials['K57', 'x_d_towerelem'] = np.zeros((1,10))
		partials['K57', 'normforce_mode_elem'] = np.zeros((1,23))

		partials['K77', 'z_sparnode'] = np.zeros((1,14))
		partials['K77', 'x_d_sparelem'] = np.zeros((1,13))
		partials['K77', 'x_dd_sparelem'] = np.zeros((1,13))
		partials['K77', 'z_towernode'] = np.zeros((1,11))
		partials['K77', 'x_d_towerelem'] = np.zeros((1,10))
		partials['K77', 'x_dd_towerelem'] = np.zeros((1,10))
		partials['K77', 'normforce_mode_elem'] = np.zeros((1,23))
		partials['K77', 'EI_mode_elem'] = np.zeros((1,23))

		N_sparelem = len(x_d_sparelem)
		N_towerelem = len(x_d_towerelem)

		for i in xrange(N_sparelem):
			dz = z_sparnode[i+1] - z_sparnode[i]

			partials['K57', 'z_sparnode'][0,i] += -norm_force[i] * x_d_sparelem[i]
			partials['K57', 'z_sparnode'][0,i+1] += norm_force[i] * x_d_sparelem[i]
			partials['K57', 'x_d_sparelem'][0,i] += norm_force[i] * dz
			partials['K57', 'normforce_mode_elem'][0,i] += x_d_sparelem[i] * dz

			partials['K77', 'z_sparnode'][0,i] += -EI[i] * x_dd_sparelem[i]**2. - norm_force[i] * x_d_sparelem[i]**2.
			partials['K77', 'z_sparnode'][0,i+1] += EI[i] * x_dd_sparelem[i]**2. + norm_force[i] * x_d_sparelem[i]**2.
			partials['K77', 'x_d_sparelem'][0,i] += 2. * norm_force[i] * x_d_sparelem[i] * dz
			partials['K77', 'x_dd_sparelem'][0,i] += 2. * dz * EI[i] * x_dd_sparelem[i]
			partials['K77', 'normforce_mode_elem'][0,i] += x_d_sparelem[i]**2. * dz
			partials['K77', 'EI_mode_elem'][0,i] += dz * x_dd_sparelem[i]**2.

		for i in xrange(N_towerelem):
			dz = z_towernode[i+1] - z_towernode[i]

			partials['K57', 'z_towernode'][0,i] += -norm_force[N_sparelem+i] * x_d_towerelem[i]
			partials['K57', 'z_towernode'][0,i+1] += norm_force[N_sparelem+i] * x_d_towerelem[i]
			partials['K57', 'x_d_towerelem'][0,i] += norm_force[N_sparelem+i] * dz
			partials['K57', 'normforce_mode_elem'][0,N_sparelem+i] += x_d_towerelem[i] * dz

			partials['K77', 'z_towernode'][0,i] += -EI[N_sparelem+i] * x_dd_towerelem[i]**2. - norm_force[N_sparelem+i] * x_d_towerelem[i]**2.
			partials['K77', 'z_towernode'][0,i+1] += EI[N_sparelem+i] * x_dd_towerelem[i]**2. + norm_force[N_sparelem+i] * x_d_towerelem[i]**2.
			partials['K77', 'x_d_towerelem'][0,i] += 2. * norm_force[N_sparelem+i] * x_d_towerelem[i] * dz
			partials['K77', 'x_dd_towerelem'][0,i] += 2. * dz * EI[N_sparelem+i] * x_dd_towerelem[i]
			partials['K77', 'normforce_mode_elem'][0,N_sparelem+i] += x_d_towerelem[i]**2. * dz
			partials['K77', 'EI_mode_elem'][0,N_sparelem+i] += dz * x_dd_towerelem[i]**2.