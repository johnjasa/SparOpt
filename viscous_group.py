import numpy as np

from openmdao.api import Group

from viscous_damping import ViscousDamping
from vel_distr import VelDistr

class Viscous(Group):

	def initialize(self):
		self.options.declare('freqs', types=dict)

	def setup(self):
		freqs = self.options['freqs']

		self.add_subsystem('vel_distr', VelDistr(freqs=freqs), promotes_inputs=['Re_RAO_wave_vel_surge', 'Re_RAO_wave_vel_pitch', 'Re_RAO_wave_vel_bend', 'Im_RAO_wave_vel_surge', 'Im_RAO_wave_vel_pitch', 'Im_RAO_wave_vel_bend', 'Re_RAO_wind_vel_surge', 'Re_RAO_wind_vel_pitch', 'Re_RAO_wind_vel_bend', 'Im_RAO_wind_vel_surge', 'Im_RAO_wind_vel_pitch', 'Im_RAO_wind_vel_bend', 'Re_RAO_Mwind_vel_surge', 'Re_RAO_Mwind_vel_pitch', 'Re_RAO_Mwind_vel_bend', 'Im_RAO_Mwind_vel_surge', 'Im_RAO_Mwind_vel_pitch', 'Im_RAO_Mwind_vel_bend', 'S_wave', 'S_wind', 'z_sparnode', 'x_sparelem'], promotes_outputs=['stddev_vel_distr'])
		
		self.add_subsystem('viscous_damping', ViscousDamping(), promotes_inputs=['Cd', 'x_sparelem', 'z_sparnode', 'Z_spar', 'D_spar', 'stddev_vel_distr'], promotes_outputs=['B_visc_11', 'B_visc_15', 'B_visc_17', 'B_visc_55', 'B_visc_57', 'B_visc_77'])