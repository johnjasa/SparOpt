import numpy as np

from openmdao.api import ExplicitComponent

class GlobalStiffness(ExplicitComponent):

	def setup(self):
		self.add_input('D_spar', val=np.zeros(10), units='m')
		self.add_input('tot_M_spar', val=0., units='kg')
		self.add_input('M_turb', val=0., units='kg')
		self.add_input('M_ball', val=0., units='kg')
		self.add_input('CoG_total', val=0., units='m')
		self.add_input('M_moor', val=0., units='kg')
		self.add_input('K_moor', val=0., units='N/m')
		self.add_input('z_moor', val=0., units='m')
		self.add_input('K17', val=0., units='N/m')
		self.add_input('K57', val=0., units='N')
		self.add_input('K77', val=0., units='N/m')
		self.add_input('buoy_spar', val=0., units='N')
		self.add_input('CoB', val=0., units='m')

		self.add_output('K_global', val=np.zeros((3,3)), units='N/m')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		D_spar = inputs['D_spar']
		tot_M_spar = inputs['tot_M_spar']
		M_turb = inputs['M_turb']
		M_ball = inputs['M_ball']
		CoG_total = inputs['CoG_total']
		M_moor = inputs['M_moor']
		K_moor = inputs['K_moor']
		z_moor = inputs['z_moor']
		K17 = inputs['K17']
		K57 = inputs['K57']
		K77 = inputs['K77']
		buoy_spar = inputs['buoy_spar']
		CoB = inputs['CoB']

		hydrostatic_pitch = buoy_spar * CoB - (tot_M_spar + M_ball + M_turb) * 9.80665 * CoG_total + 1025. * 9.80665 * np.pi / 64. * D_spar[-1]**4.

		outputs['K_global'] = np.zeros((3,3))

		outputs['K_global'][0,0] += K_moor
		outputs['K_global'][0,1] += K_moor * z_moor
		outputs['K_global'][0,2] += K17
		outputs['K_global'][1,0] += K_moor * z_moor
		outputs['K_global'][1,1] += K_moor * z_moor**2. - M_moor * 9.80665 * z_moor + hydrostatic_pitch
		outputs['K_global'][1,2] += K57
		outputs['K_global'][2,0] += K17
		outputs['K_global'][2,1] += K57
		outputs['K_global'][2,2] += K77

	def compute_partials(self, inputs, partials):
		D_spar = inputs['D_spar']
		tot_M_spar = inputs['tot_M_spar']
		M_turb = inputs['M_turb']
		M_ball = inputs['M_ball']
		CoG_total = inputs['CoG_total']
		M_moor = inputs['M_moor']
		K_moor = inputs['K_moor']
		z_moor = inputs['z_moor']
		K17 = inputs['K17']
		K57 = inputs['K57']
		K77 = inputs['K77']
		buoy_spar = inputs['buoy_spar']
		CoB = inputs['CoB']

		partials['K_global', 'D_spar'] = np.concatenate((np.zeros((4,10)),np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 1025. * 9.80665 * np.pi / 16. * D_spar[-1]**3.]]),np.zeros((4,10))),0)
		partials['K_global', 'tot_M_spar'] = np.array([0., 0., 0., 0., -9.80665 * CoG_total, 0., 0., 0., 0.], dtype='float')
		partials['K_global', 'M_turb'] = np.array([0., 0., 0., 0., -9.80665 * CoG_total, 0., 0., 0., 0.], dtype='float')
		partials['K_global', 'M_ball'] = np.array([0., 0., 0., 0., -9.80665 * CoG_total, 0., 0., 0., 0.], dtype='float')
		partials['K_global', 'CoG_total'] = np.array([0., 0., 0., 0., -(tot_M_spar + M_ball + M_turb) * 9.80665, 0., 0., 0., 0.], dtype='float')
		partials['K_global', 'M_moor'] = np.array([0., 0., 0., 0., -9.80665 * z_moor, 0., 0., 0., 0.], dtype='float')
		partials['K_global', 'K_moor'] = np.array([1., z_moor, 0., z_moor, z_moor**2., 0., 0., 0., 0.], dtype='float')
		partials['K_global', 'z_moor'] = np.array([0., K_moor, 0., K_moor, 2. * K_moor * z_moor - M_moor * 9.80665, 0., 0., 0., 0.], dtype='float')
		partials['K_global', 'K17'] = np.array([0., 0., 1., 0., 0., 0., 1., 0., 0.], dtype='float')
		partials['K_global', 'K57'] = np.array([0., 0., 0., 0., 0., 1., 0., 1., 0.], dtype='float')
		partials['K_global', 'K77'] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 1.], dtype='float')
		partials['K_global', 'buoy_spar'] = np.array([0., 0., 0., 0., CoB, 0., 0., 0., 0.], dtype='float')
		partials['K_global', 'CoB'] = np.array([0., 0., 0., 0., buoy_spar, 0., 0., 0., 0.], dtype='float')
