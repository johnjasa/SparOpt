import numpy as np

from openmdao.api import Group

from norm_resp_wave import NormRespWave
from norm_resp_wind import NormRespWind
from norm_vel_wave import NormVelWave
from norm_vel_wind import NormVelWind
from resp_spectra import RespSpectra
from vel_spectra import VelSpectra
#from vel_resp_distr import VelRespDistr
from std_dev_resp import StdDevResp
from std_dev_resp_vel import StdDevRespVel

class Postpro(Group):

	def initialize(self):
		self.options.declare('freqs', types=dict)

	def setup(self):
		freqs = self.options['freqs']

	 	self.add_subsystem('norm_resp_wave', NormRespWave(freqs=freqs), promotes_inputs=['Re_wave_forces', 'Im_wave_forces', 'Re_H_feedbk', 'Im_H_feedbk', 'k_i', 'k_p', 'gain_corr_factor'], promotes_outputs=['Re_RAO_wave_surge', 'Re_RAO_wave_pitch', 'Re_RAO_wave_bend', 'Re_RAO_wave_rotspeed', 'Re_RAO_wave_rot_lp', 'Re_RAO_wave_rotspeed_lp', 'Re_RAO_wave_bldpitch', 'Im_RAO_wave_surge', 'Im_RAO_wave_pitch', 'Im_RAO_wave_bend', 'Im_RAO_wave_rotspeed', 'Im_RAO_wave_rot_lp', 'Im_RAO_wave_rotspeed_lp', 'Im_RAO_wave_bldpitch'])

	 	self.add_subsystem('norm_resp_wind', NormRespWind(freqs=freqs), promotes_inputs=['thrust_wind', 'moment_wind', 'torque_wind', 'Re_H_feedbk', 'Im_H_feedbk', 'k_i', 'k_p', 'gain_corr_factor'], promotes_outputs=['Re_RAO_wind_surge', 'Re_RAO_wind_pitch', 'Re_RAO_wind_bend', 'Re_RAO_wind_rotspeed', 'Re_RAO_wind_rot_lp', 'Re_RAO_wind_rotspeed_lp', 'Re_RAO_wind_bldpitch', 'Im_RAO_wind_surge', 'Im_RAO_wind_pitch', 'Im_RAO_wind_bend', 'Im_RAO_wind_rotspeed', 'Im_RAO_wind_rot_lp', 'Im_RAO_wind_rotspeed_lp', 'Im_RAO_wind_bldpitch'])

	 	self.add_subsystem('norm_vel_wave', NormVelWave(freqs=freqs), promotes_inputs=['Re_RAO_wave_surge', 'Re_RAO_wave_pitch', 'Re_RAO_wave_bend', 'Im_RAO_wave_surge', 'Im_RAO_wave_pitch', 'Im_RAO_wave_bend'], promotes_outputs=['Re_RAO_wave_vel_surge', 'Re_RAO_wave_vel_pitch', 'Re_RAO_wave_vel_bend', 'Im_RAO_wave_vel_surge', 'Im_RAO_wave_vel_pitch', 'Im_RAO_wave_vel_bend'])

	 	self.add_subsystem('norm_vel_wind', NormVelWind(freqs=freqs), promotes_inputs=['Re_RAO_wind_surge', 'Re_RAO_wind_pitch', 'Re_RAO_wind_bend', 'Im_RAO_wind_surge', 'Im_RAO_wind_pitch', 'Im_RAO_wind_bend'], promotes_outputs=['Re_RAO_wind_vel_surge', 'Re_RAO_wind_vel_pitch', 'Re_RAO_wind_vel_bend', 'Im_RAO_wind_vel_surge', 'Im_RAO_wind_vel_pitch', 'Im_RAO_wind_vel_bend'])

		self.add_subsystem('resp_spectra', RespSpectra(freqs=freqs), promotes_inputs=['Re_RAO_wave_surge', 'Re_RAO_wave_pitch', 'Re_RAO_wave_bend', 'Re_RAO_wave_rotspeed', 'Re_RAO_wave_rot_lp', 'Re_RAO_wave_rotspeed_lp', 'Re_RAO_wave_bldpitch', 'Im_RAO_wave_surge', 'Im_RAO_wave_pitch', 'Im_RAO_wave_bend', 'Im_RAO_wave_rotspeed', 'Im_RAO_wave_rot_lp', 'Im_RAO_wave_rotspeed_lp', 'Im_RAO_wave_bldpitch', 'Re_RAO_wind_surge', 'Re_RAO_wind_pitch', 'Re_RAO_wind_bend', 'Re_RAO_wind_rotspeed', 'Re_RAO_wind_rot_lp', 'Re_RAO_wind_rotspeed_lp', 'Re_RAO_wind_bldpitch', 'Im_RAO_wind_surge', 'Im_RAO_wind_pitch', 'Im_RAO_wind_bend', 'Im_RAO_wind_rotspeed', 'Im_RAO_wind_rot_lp', 'Im_RAO_wind_rotspeed_lp', 'Im_RAO_wind_bldpitch', 'S_wave', 'S_wind'], promotes_outputs=['resp_surge', 'resp_pitch', 'resp_bend', 'resp_rotspeed', 'resp_rot_lp', 'resp_rotspeed_lp', 'resp_bldpitch'])

	 	self.add_subsystem('vel_spectra', VelSpectra(freqs=freqs), promotes_inputs=['Re_RAO_wave_vel_surge', 'Re_RAO_wave_vel_pitch', 'Re_RAO_wave_vel_bend', 'Im_RAO_wave_vel_surge', 'Im_RAO_wave_vel_pitch', 'Im_RAO_wave_vel_bend', 'Re_RAO_wind_vel_surge', 'Re_RAO_wind_vel_pitch', 'Re_RAO_wind_vel_bend', 'Im_RAO_wind_vel_surge', 'Im_RAO_wind_vel_pitch', 'Im_RAO_wind_vel_bend', 'S_wave', 'S_wind'], promotes_outputs=['resp_vel_surge', 'resp_vel_pitch', 'resp_vel_bend'])

	 	#self.add_subsystem('vel_resp_distr', VelRespDistr(freqs=freqs), promotes_inputs=['Re_RAO_wave_vel_surge', 'Re_RAO_wave_vel_pitch', 'Re_RAO_wave_vel_bend', 'Im_RAO_wave_vel_surge', 'Im_RAO_wave_vel_pitch', 'Im_RAO_wave_vel_bend', 'Re_RAO_wind_vel_surge', 'Re_RAO_wind_vel_pitch', 'Re_RAO_wind_vel_bend', 'Im_RAO_wind_vel_surge', 'Im_RAO_wind_vel_pitch', 'Im_RAO_wind_vel_bend', 'S_wave', 'S_wind', 'z_sparnode', 'x_sparelem'], promotes_outputs=['stddev_vel_distr'])

	 	self.add_subsystem('std_dev_resp', StdDevResp(freqs=freqs), promotes_inputs=['resp_surge', 'resp_pitch', 'resp_bend', 'resp_rotspeed', 'resp_rot_lp', 'resp_rotspeed_lp', 'resp_bldpitch'], promotes_outputs=['stddev_surge', 'stddev_pitch', 'stddev_bend', 'stddev_rotspeed', 'stddev_rot_lp', 'stddev_rotspeed_lp', 'stddev_bldpitch'])

	 	self.add_subsystem('std_dev_resp_vel', StdDevRespVel(freqs=freqs), promotes_inputs=['resp_vel_surge', 'resp_vel_pitch', 'resp_vel_bend'], promotes_outputs=['stddev_vel_surge', 'stddev_vel_pitch', 'stddev_vel_bend'])