import numpy as np

from openmdao.api import ExplicitComponent

class MeanTowerDrag(ExplicitComponent):

	def setup(self):
		self.add_input('D_tower', val=np.zeros(10), units='m')
		self.add_input('Z_tower', val=np.zeros(11), units='m')
		self.add_input('L_tower', val=np.zeros(10), units='m')
		self.add_input('windspeed_0', val=0., units='m/s')
		self.add_input('Cd_tower', val=0.)
		self.add_input('CoG_rotor', val=0., units='m')
		self.add_input('rho_wind', val=0., units='kg/m**3')

		self.add_output('F0_tower_drag', val=0., units='N')
		self.add_output('Z0_tower_drag', val=0., units='m')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		D_tower = inputs['D_tower']
		Z_tower = inputs['Z_tower']
		L_tower = inputs['L_tower']
		windspeed_0 = inputs['windspeed_0']
		Cd_tower = inputs['Cd_tower']
		CoG_rotor = inputs['CoG_rotor']
		rho_wind = inputs['rho_wind']

		F0 = 0.
		M0 = 0.

		for i in xrange(len(D_tower)):
			z = (Z_tower[i] + Z_tower[i+1]) / 2.
			V = windspeed_0 / (CoG_rotor / z)**0.14 #assumes wind shear with alpha = 0.14
			F0 += 0.5 * rho_wind * Cd_tower * D_tower[i] * L_tower[i] * V**2.
			M0 += 0.5 * rho_wind * Cd_tower * D_tower[i] * L_tower[i] * V**2. * z

		outputs['F0_tower_drag'] = F0
		outputs['Z0_tower_drag'] = M0 / F0

	def compute_partials(self, inputs, partials):
		D_tower = inputs['D_tower']
		Z_tower = inputs['Z_tower']
		L_tower = inputs['L_tower']
		windspeed_0 = inputs['windspeed_0']
		Cd_tower = inputs['Cd_tower']
		CoG_rotor = inputs['CoG_rotor']
		rho_wind = inputs['rho_wind']

		dF_dD_tower = np.zeros((1,10))
		dF_dZ_tower = np.zeros((1,11))
		dF_dL_tower = np.zeros((1,10))
		dF_dwindspeed_0 = 0.
		dF_dCd_tower = 0.
		dF_dCoG_rotor = 0.
		dF_drho_wind = 0.

		dM_dD_tower = np.zeros((1,10))
		dM_dZ_tower = np.zeros((1,11))
		dM_dL_tower = np.zeros((1,10))
		dM_dwindspeed_0 = 0.
		dM_dCd_tower = 0.
		dM_dCoG_rotor = 0.
		dM_drho_wind = 0.

		F0 = 0.
		M0 = 0.

		for i in xrange(len(D_tower)):
			z = (Z_tower[i] + Z_tower[i+1]) / 2.
			V = windspeed_0 / (CoG_rotor / z)**0.14 #assumes wind shear with alpha = 0.14

			F0 += 0.5 * rho_wind * Cd_tower * D_tower[i] * L_tower[i] * V**2.
			M0 += 0.5 * rho_wind * Cd_tower * D_tower[i] * L_tower[i] * V**2. * z
			
			dF_dD_tower[0,i] += 0.5 * rho_wind * Cd_tower * L_tower[i] * V**2.
			dF_dZ_tower[0,i] += 0.5 * rho_wind * Cd_tower * D_tower[i] * L_tower[i] * 2. * V * 0.14 * windspeed_0 / (CoG_rotor / z)**1.14 * CoG_rotor / z**2. / 2.
			dF_dZ_tower[0,i+1] += 0.5 * rho_wind * Cd_tower * D_tower[i] * L_tower[i] * 2. * V * 0.14 * windspeed_0 / (CoG_rotor / z)**1.14 * CoG_rotor / z**2. / 2.
			dF_dL_tower[0,i] += 0.5 * rho_wind * Cd_tower * D_tower[i] * V**2.
			dF_dwindspeed_0 += 0.5 * rho_wind * Cd_tower * D_tower[i] * L_tower[i] * 2. * V * 1. / (CoG_rotor / z)**0.14
			dF_dCd_tower += 0.5 * rho_wind * D_tower[i] * L_tower[i] * V**2.
			dF_dCoG_rotor += -0.5 * rho_wind * Cd_tower * D_tower[i] * L_tower[i] * 2. * V * 0.14 * windspeed_0 / (CoG_rotor / z)**1.14 * 1. / z
			dF_drho_wind += 0.5 * Cd_tower * D_tower[i] * L_tower[i] * V**2.

			dM_dD_tower[0,i] += 0.5 * rho_wind * Cd_tower * L_tower[i] * V**2. * z
			dM_dZ_tower[0,i] += 0.5 * rho_wind * Cd_tower * D_tower[i] * L_tower[i] * (z * 2. * V * 0.14 * windspeed_0 / (CoG_rotor / z)**1.14 * CoG_rotor / z**2. / 2. + V**2. / 2.)
			dM_dZ_tower[0,i+1] += 0.5 * rho_wind * Cd_tower * D_tower[i] * L_tower[i] * (z * 2. * V * 0.14 * windspeed_0 / (CoG_rotor / z)**1.14 * CoG_rotor / z**2. / 2. + V**2. / 2.)
			dM_dL_tower[0,i] += 0.5 * rho_wind * Cd_tower * D_tower[i] *  V**2. * z
			dM_dwindspeed_0 += 0.5 * rho_wind * Cd_tower * D_tower[i] * L_tower[i] * 2. * V * z * 1. / (CoG_rotor / z)**0.14
			dM_dCd_tower += 0.5 * rho_wind * D_tower[i] * L_tower[i] * V**2. * z
			dM_dCoG_rotor += -0.5 * rho_wind * Cd_tower * D_tower[i] * L_tower[i] * 2. * V * z * 0.14 * windspeed_0 / (CoG_rotor / z)**1.14 * 1. / z
			dM_drho_wind += 0.5 * Cd_tower * D_tower[i] * L_tower[i] * V**2. * z

		partials['F0_tower_drag', 'D_tower'] = dF_dD_tower
		partials['F0_tower_drag', 'Z_tower'] = dF_dZ_tower
		partials['F0_tower_drag', 'L_tower'] = dF_dL_tower
		partials['F0_tower_drag', 'windspeed_0'] = dF_dwindspeed_0
		partials['F0_tower_drag', 'Cd_tower'] = dF_dCd_tower
		partials['F0_tower_drag', 'CoG_rotor'] = dF_dCoG_rotor
		partials['F0_tower_drag', 'rho_wind'] = dF_drho_wind

		partials['Z0_tower_drag', 'D_tower'] = dM_dD_tower / F0 - M0 / F0**2. * dF_dD_tower
		partials['Z0_tower_drag', 'Z_tower'] = dM_dZ_tower / F0 - M0 / F0**2. * dF_dZ_tower
		partials['Z0_tower_drag', 'L_tower'] = dM_dL_tower / F0 - M0 / F0**2. * dF_dL_tower
		partials['Z0_tower_drag', 'windspeed_0'] = dM_dwindspeed_0 / F0 - M0 / F0**2. * dF_dwindspeed_0
		partials['Z0_tower_drag', 'Cd_tower'] = dM_dCd_tower / F0 - M0 / F0**2. * dF_dCd_tower
		partials['Z0_tower_drag', 'CoG_rotor'] = dM_dCoG_rotor / F0 - M0 / F0**2. * dF_dCoG_rotor
		partials['Z0_tower_drag', 'rho_wind'] = dM_drho_wind / F0 - M0 / F0**2. * dF_drho_wind