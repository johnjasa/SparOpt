import numpy as np

from openmdao.api import Group

from norm_resp_wave_rotspeed import NormRespWaveRotspeed
from norm_resp_wave_bldpitch import NormRespWaveBldpitch
from norm_resp_wind_rotspeed import NormRespWindRotspeed
from norm_resp_wind_bldpitch import NormRespWindBldpitch
from norm_resp_Mwind_rotspeed import NormRespMWindRotspeed
from norm_resp_Mwind_bldpitch import NormRespMWindBldpitch
from norm_acc_wave_surge import NormAccWaveSurge
from norm_acc_wave_pitch import NormAccWavePitch
from norm_acc_wave_bend import NormAccWaveBend
from norm_acc_wind_surge import NormAccWindSurge
from norm_acc_wind_pitch import NormAccWindPitch
from norm_acc_wind_bend import NormAccWindBend
from norm_acc_Mwind_surge import NormAccMWindSurge
from norm_acc_Mwind_pitch import NormAccMWindPitch
from norm_acc_Mwind_bend import NormAccMWindBend
from norm_fairlead_wave import NormFairleadWave
from norm_fairlead_wind import NormFairleadWind
from norm_fairlead_Mwind import NormFairleadMWind
from tower_moment_gains import TowerMomentGains
from norm_tower_moment_wave import NormTowerMomentWave
from norm_tower_moment_wind import NormTowerMomentWind
from norm_tower_moment_Mwind import NormTowerMomentMWind
from tower_moment_spectrum import TowerMomentSpectrum
from tower_stress_spectrum import TowerStressSpectrum
from spar_sec_disp import SparSecDisp
from hull_moment_gains import HullMomentGains
from hull_wave_excit_mom import HullWaveExcitMom
from norm_hull_moment_wave import NormHullMomentWave
from norm_hull_moment_wind import NormHullMomentWind
from norm_hull_moment_Mwind import NormHullMomentMWind
from hull_moment_spectrum import HullMomentSpectrum
from hull_stress_spectrum import HullStressSpectrum
from tower_fatigue import TowerFatigue
from hull_fatigue import HullFatigue

class Postpro(Group):

	def initialize(self):
		self.options.declare('freqs', types=dict)

	def setup(self):
		freqs = self.options['freqs']

	 	self.add_subsystem('norm_resp_wave_rotspeed', NormRespWaveRotspeed(freqs=freqs), promotes_inputs=['Re_wave_force_surge', 'Im_wave_force_surge', 'Re_wave_force_pitch', 'Im_wave_force_pitch', 'Re_wave_force_bend', 'Im_wave_force_bend', 'Re_H_feedbk', 'Im_H_feedbk', 'windspeed_0'], promotes_outputs=['Re_RAO_wave_rotspeed', 'Im_RAO_wave_rotspeed'])

	 	self.add_subsystem('norm_resp_wave_bldpitch', NormRespWaveBldpitch(freqs=freqs), promotes_inputs=['Re_wave_force_surge', 'Im_wave_force_surge', 'Re_wave_force_pitch', 'Im_wave_force_pitch', 'Re_wave_force_bend', 'Im_wave_force_bend', 'Re_H_feedbk', 'Im_H_feedbk', 'k_i', 'k_p', 'gain_corr_factor', 'windspeed_0'], promotes_outputs=['Re_RAO_wave_bldpitch', 'Im_RAO_wave_bldpitch'])

	 	self.add_subsystem('norm_resp_wind_rotspeed', NormRespWindRotspeed(freqs=freqs), promotes_inputs=['thrust_wind', 'torque_wind', 'Re_H_feedbk', 'Im_H_feedbk', 'windspeed_0'], promotes_outputs=['Re_RAO_wind_rotspeed', 'Im_RAO_wind_rotspeed'])

	 	self.add_subsystem('norm_resp_wind_bldpitch', NormRespWindBldpitch(freqs=freqs), promotes_inputs=['thrust_wind', 'torque_wind', 'Re_H_feedbk', 'Im_H_feedbk', 'k_i', 'k_p', 'gain_corr_factor', 'windspeed_0'], promotes_outputs=['Re_RAO_wind_bldpitch', 'Im_RAO_wind_bldpitch'])

	 	self.add_subsystem('norm_resp_Mwind_rotspeed', NormRespMWindRotspeed(freqs=freqs), promotes_inputs=['moment_wind', 'Re_H_feedbk', 'Im_H_feedbk', 'windspeed_0'], promotes_outputs=['Re_RAO_Mwind_rotspeed', 'Im_RAO_Mwind_rotspeed'])

	 	self.add_subsystem('norm_resp_Mwind_bldpitch', NormRespMWindBldpitch(freqs=freqs), promotes_inputs=['moment_wind', 'Re_H_feedbk', 'Im_H_feedbk', 'k_i', 'k_p', 'gain_corr_factor', 'windspeed_0'], promotes_outputs=['Re_RAO_Mwind_bldpitch', 'Im_RAO_Mwind_bldpitch'])

	 	self.add_subsystem('norm_acc_wave_surge', NormAccWaveSurge(freqs=freqs), promotes_inputs=['Re_RAO_wave_surge', 'Im_RAO_wave_surge'], promotes_outputs=['Re_RAO_wave_acc_surge', 'Im_RAO_wave_acc_surge'])

	 	self.add_subsystem('norm_acc_wave_pitch', NormAccWavePitch(freqs=freqs), promotes_inputs=['Re_RAO_wave_pitch', 'Im_RAO_wave_pitch'], promotes_outputs=['Re_RAO_wave_acc_pitch', 'Im_RAO_wave_acc_pitch'])

	 	self.add_subsystem('norm_acc_wave_bend', NormAccWaveBend(freqs=freqs), promotes_inputs=['Re_RAO_wave_bend', 'Im_RAO_wave_bend'], promotes_outputs=['Re_RAO_wave_acc_bend', 'Im_RAO_wave_acc_bend'])

	 	self.add_subsystem('norm_acc_wind_surge', NormAccWindSurge(freqs=freqs), promotes_inputs=['Re_RAO_wind_surge', 'Im_RAO_wind_surge'], promotes_outputs=['Re_RAO_wind_acc_surge', 'Im_RAO_wind_acc_surge'])

	 	self.add_subsystem('norm_acc_wind_pitch', NormAccWindPitch(freqs=freqs), promotes_inputs=['Re_RAO_wind_pitch', 'Im_RAO_wind_pitch'], promotes_outputs=['Re_RAO_wind_acc_pitch', 'Im_RAO_wind_acc_pitch'])

	 	self.add_subsystem('norm_acc_wind_bend', NormAccWindBend(freqs=freqs), promotes_inputs=['Re_RAO_wind_bend', 'Im_RAO_wind_bend'], promotes_outputs=['Re_RAO_wind_acc_bend', 'Im_RAO_wind_acc_bend'])

		self.add_subsystem('norm_acc_Mwind_surge', NormAccMWindSurge(freqs=freqs), promotes_inputs=['Re_RAO_Mwind_surge', 'Im_RAO_Mwind_surge'], promotes_outputs=['Re_RAO_Mwind_acc_surge', 'Im_RAO_Mwind_acc_surge'])

	 	self.add_subsystem('norm_acc_Mwind_pitch', NormAccMWindPitch(freqs=freqs), promotes_inputs=['Re_RAO_Mwind_pitch', 'Im_RAO_Mwind_pitch'], promotes_outputs=['Re_RAO_Mwind_acc_pitch', 'Im_RAO_Mwind_acc_pitch'])

	 	self.add_subsystem('norm_acc_Mwind_bend', NormAccMWindBend(freqs=freqs), promotes_inputs=['Re_RAO_Mwind_bend', 'Im_RAO_Mwind_bend'], promotes_outputs=['Re_RAO_Mwind_acc_bend', 'Im_RAO_Mwind_acc_bend'])

		self.add_subsystem('norm_fairlead_wave', NormFairleadWave(freqs=freqs), promotes_inputs=['Re_RAO_wave_surge', 'Re_RAO_wave_pitch', 'Im_RAO_wave_surge', 'Im_RAO_wave_pitch', 'z_moor'], promotes_outputs=['Re_RAO_wave_fairlead', 'Im_RAO_wave_fairlead'])

	 	self.add_subsystem('norm_fairlead_wind', NormFairleadWind(freqs=freqs), promotes_inputs=['Re_RAO_wind_surge', 'Re_RAO_wind_pitch', 'Im_RAO_wind_surge', 'Im_RAO_wind_pitch', 'z_moor'], promotes_outputs=['Re_RAO_wind_fairlead', 'Im_RAO_wind_fairlead'])

		self.add_subsystem('norm_fairlead_Mwind', NormFairleadMWind(freqs=freqs), promotes_inputs=['Re_RAO_Mwind_surge', 'Re_RAO_Mwind_pitch', 'Im_RAO_Mwind_surge', 'Im_RAO_Mwind_pitch', 'z_moor'], promotes_outputs=['Re_RAO_Mwind_fairlead', 'Im_RAO_Mwind_fairlead'])

		self.add_subsystem('tower_moment_gains', TowerMomentGains(), promotes_inputs=['M_tower', 'M_nacelle', 'M_rotor', 'I_rotor', 'CoG_nacelle', 'CoG_rotor', 'z_towernode', 'x_towerelem', 'x_towernode', 'x_d_towertop', 'dthrust_dv', 'dmoment_dv', 'dthrust_drotspeed', 'dthrust_dbldpitch'], promotes_outputs=['mom_acc_surge', 'mom_acc_pitch', 'mom_acc_bend', 'mom_damp_surge', 'mom_damp_pitch', 'mom_damp_bend', 'mom_grav_pitch', 'mom_grav_bend', 'mom_rotspeed', 'mom_bldpitch'])

	 	self.add_subsystem('norm_tower_moment_wave', NormTowerMomentWave(freqs=freqs), promotes_inputs=['mom_acc_surge', 'mom_acc_pitch', 'mom_acc_bend', 'mom_damp_surge', 'mom_damp_pitch', 'mom_damp_bend', 'mom_grav_pitch', 'mom_grav_bend', 'mom_rotspeed', 'mom_bldpitch', 'Re_RAO_wave_pitch', 'Re_RAO_wave_bend', 'Re_RAO_wave_rotspeed', 'Re_RAO_wave_bldpitch', 'Im_RAO_wave_pitch', 'Im_RAO_wave_bend', 'Im_RAO_wave_rotspeed', 'Im_RAO_wave_bldpitch', 'Re_RAO_wave_vel_surge', 'Re_RAO_wave_vel_pitch', 'Re_RAO_wave_vel_bend', 'Im_RAO_wave_vel_surge', 'Im_RAO_wave_vel_pitch', 'Im_RAO_wave_vel_bend', 'Re_RAO_wave_acc_surge', 'Re_RAO_wave_acc_pitch', 'Re_RAO_wave_acc_bend', 'Im_RAO_wave_acc_surge', 'Im_RAO_wave_acc_pitch', 'Im_RAO_wave_acc_bend'], promotes_outputs=['Re_RAO_wave_tower_moment', 'Im_RAO_wave_tower_moment'])

	 	self.add_subsystem('norm_tower_moment_wind', NormTowerMomentWind(freqs=freqs), promotes_inputs=['mom_acc_surge', 'mom_acc_pitch', 'mom_acc_bend', 'mom_damp_surge', 'mom_damp_pitch', 'mom_damp_bend', 'mom_grav_pitch', 'mom_grav_bend', 'mom_rotspeed', 'mom_bldpitch', 'Re_RAO_wind_pitch', 'Re_RAO_wind_bend', 'Re_RAO_wind_rotspeed', 'Re_RAO_wind_bldpitch', 'Im_RAO_wind_pitch', 'Im_RAO_wind_bend', 'Im_RAO_wind_rotspeed', 'Im_RAO_wind_bldpitch', 'Re_RAO_wind_vel_surge', 'Re_RAO_wind_vel_pitch', 'Re_RAO_wind_vel_bend', 'Im_RAO_wind_vel_surge', 'Im_RAO_wind_vel_pitch', 'Im_RAO_wind_vel_bend', 'Re_RAO_wind_acc_surge', 'Re_RAO_wind_acc_pitch', 'Re_RAO_wind_acc_bend', 'Im_RAO_wind_acc_surge', 'Im_RAO_wind_acc_pitch', 'Im_RAO_wind_acc_bend', 'CoG_rotor', 'Z_tower', 'dthrust_dv', 'thrust_wind'], promotes_outputs=['Re_RAO_wind_tower_moment', 'Im_RAO_wind_tower_moment'])

	 	self.add_subsystem('norm_tower_moment_Mwind', NormTowerMomentMWind(freqs=freqs), promotes_inputs=['mom_acc_surge', 'mom_acc_pitch', 'mom_acc_bend', 'mom_damp_surge', 'mom_damp_pitch', 'mom_damp_bend', 'mom_grav_pitch', 'mom_grav_bend', 'mom_rotspeed', 'mom_bldpitch', 'Re_RAO_Mwind_pitch', 'Re_RAO_Mwind_bend', 'Re_RAO_Mwind_rotspeed', 'Re_RAO_Mwind_bldpitch', 'Im_RAO_Mwind_pitch', 'Im_RAO_Mwind_bend', 'Im_RAO_Mwind_rotspeed', 'Im_RAO_Mwind_bldpitch', 'Re_RAO_Mwind_vel_surge', 'Re_RAO_Mwind_vel_pitch', 'Re_RAO_Mwind_vel_bend', 'Im_RAO_Mwind_vel_surge', 'Im_RAO_Mwind_vel_pitch', 'Im_RAO_Mwind_vel_bend', 'Re_RAO_Mwind_acc_surge', 'Re_RAO_Mwind_acc_pitch', 'Re_RAO_Mwind_acc_bend', 'Im_RAO_Mwind_acc_surge', 'Im_RAO_Mwind_acc_pitch', 'Im_RAO_Mwind_acc_bend', 'dmoment_dv', 'moment_wind'], promotes_outputs=['Re_RAO_Mwind_tower_moment', 'Im_RAO_Mwind_tower_moment'])

	 	self.add_subsystem('tower_moment_spectrum', TowerMomentSpectrum(freqs=freqs), promotes_inputs=['Re_RAO_wave_tower_moment', 'Im_RAO_wave_tower_moment', 'Re_RAO_wind_tower_moment', 'Im_RAO_wind_tower_moment', 'Re_RAO_Mwind_tower_moment', 'Im_RAO_Mwind_tower_moment', 'S_wave', 'S_wind'], promotes_outputs=['resp_tower_moment'])

	 	self.add_subsystem('tower_stress_spectrum', TowerStressSpectrum(freqs=freqs), promotes_inputs=['resp_tower_moment', 'D_tower_p', 'wt_tower_p'], promotes_outputs=['resp_tower_stress'])

	 	self.add_subsystem('spar_sec_disp', SparSecDisp(), promotes_inputs=['z_sparnode', 'x_sparnode', 'x_sparelem', 'spar_draft', 'L_ball', 'z_moor', 'stddev_vel_distr'], promotes_outputs=['X_sparnode', 'X_sparelem', 'stddev_vel_X_sparelem'])

	 	self.add_subsystem('hull_moment_gains', HullMomentGains(), promotes_inputs=['M_tower', 'M_nacelle', 'M_rotor', 'I_rotor', 'CoG_nacelle', 'CoG_rotor', 'z_towernode', 'x_towerelem', 'x_towernode', 'x_d_towertop', 'dthrust_dv', 'dmoment_dv', 'dthrust_drotspeed', 'dthrust_dbldpitch', 'D_spar', 'Z_spar', 'M_spar', 'X_sparnode', 'X_sparelem', 'Cd', 'stddev_vel_X_sparelem', 'M_ball_elem', 'K_moor', 'z_moor'], promotes_outputs=['hull_mom_acc_surge', 'hull_mom_acc_pitch', 'hull_mom_acc_bend', 'hull_mom_damp_surge', 'hull_mom_damp_pitch', 'hull_mom_damp_bend', 'hull_mom_grav_pitch', 'hull_mom_grav_bend', 'hull_mom_rotspeed', 'hull_mom_bldpitch', 'hull_mom_fairlead'])

	 	self.add_subsystem('hull_wave_excit_mom', HullWaveExcitMom(freqs=freqs), promotes_inputs=['D_spar', 'Z_spar', 'wave_number', 'water_depth'], promotes_outputs=['Re_hull_wave_mom', 'Im_hull_wave_mom'])

	 	self.add_subsystem('norm_hull_moment_wave', NormHullMomentWave(freqs=freqs), promotes_inputs=['hull_mom_acc_surge', 'hull_mom_acc_pitch', 'hull_mom_acc_bend', 'hull_mom_damp_surge', 'hull_mom_damp_pitch', 'hull_mom_damp_bend', 'hull_mom_grav_pitch', 'hull_mom_grav_bend', 'hull_mom_rotspeed', 'hull_mom_bldpitch', 'hull_mom_fairlead', 'Re_RAO_wave_pitch', 'Re_RAO_wave_bend', 'Re_RAO_wave_rotspeed', 'Re_RAO_wave_bldpitch', 'Re_RAO_wave_fairlead', 'Im_RAO_wave_pitch', 'Im_RAO_wave_bend', 'Im_RAO_wave_rotspeed', 'Im_RAO_wave_bldpitch', 'Im_RAO_wave_fairlead', 'Re_RAO_wave_vel_surge', 'Re_RAO_wave_vel_pitch', 'Re_RAO_wave_vel_bend', 'Im_RAO_wave_vel_surge', 'Im_RAO_wave_vel_pitch', 'Im_RAO_wave_vel_bend', 'Re_RAO_wave_acc_surge', 'Re_RAO_wave_acc_pitch', 'Re_RAO_wave_acc_bend', 'Im_RAO_wave_acc_surge', 'Im_RAO_wave_acc_pitch', 'Im_RAO_wave_acc_bend', 'Re_hull_wave_excit_mom', 'Im_hull_wave_excit_mom'], promotes_outputs=['Re_RAO_wave_hull_moment', 'Im_RAO_wave_hull_moment'])

	 	self.add_subsystem('norm_hull_moment_wind', NormHullMomentWind(freqs=freqs), promotes_inputs=['hull_mom_acc_surge', 'hull_mom_acc_pitch', 'hull_mom_acc_bend', 'hull_mom_damp_surge', 'hull_mom_damp_pitch', 'hull_mom_damp_bend', 'hull_mom_grav_pitch', 'hull_mom_grav_bend', 'hull_mom_rotspeed', 'hull_mom_bldpitch', 'hull_mom_fairlead', 'Re_RAO_wind_pitch', 'Re_RAO_wind_bend', 'Re_RAO_wind_rotspeed', 'Re_RAO_wind_bldpitch', 'Re_RAO_wind_fairlead', 'Im_RAO_wind_pitch', 'Im_RAO_wind_bend', 'Im_RAO_wind_rotspeed', 'Im_RAO_wind_bldpitch', 'Im_RAO_wind_fairlead', 'Re_RAO_wind_vel_surge', 'Re_RAO_wind_vel_pitch', 'Re_RAO_wind_vel_bend', 'Im_RAO_wind_vel_surge', 'Im_RAO_wind_vel_pitch', 'Im_RAO_wind_vel_bend', 'Re_RAO_wind_acc_surge', 'Re_RAO_wind_acc_pitch', 'Re_RAO_wind_acc_bend', 'Im_RAO_wind_acc_surge', 'Im_RAO_wind_acc_pitch', 'Im_RAO_wind_acc_bend', 'CoG_rotor', 'Z_spar', 'dthrust_dv', 'thrust_wind'], promotes_outputs=['Re_RAO_wind_hull_moment', 'Im_RAO_wind_hull_moment'])

	 	self.add_subsystem('norm_hull_moment_Mwind', NormHullMomentMWind(freqs=freqs), promotes_inputs=['hull_mom_acc_surge', 'hull_mom_acc_pitch', 'hull_mom_acc_bend', 'hull_mom_damp_surge', 'hull_mom_damp_pitch', 'hull_mom_damp_bend', 'hull_mom_grav_pitch', 'hull_mom_grav_bend', 'hull_mom_rotspeed', 'hull_mom_bldpitch', 'hull_mom_fairlead', 'Re_RAO_Mwind_pitch', 'Re_RAO_Mwind_bend', 'Re_RAO_Mwind_rotspeed', 'Re_RAO_Mwind_bldpitch', 'Re_RAO_Mwind_fairlead', 'Im_RAO_Mwind_pitch', 'Im_RAO_Mwind_bend', 'Im_RAO_Mwind_rotspeed', 'Im_RAO_Mwind_bldpitch', 'Im_RAO_Mwind_fairlead', 'Re_RAO_Mwind_vel_surge', 'Re_RAO_Mwind_vel_pitch', 'Re_RAO_Mwind_vel_bend', 'Im_RAO_Mwind_vel_surge', 'Im_RAO_Mwind_vel_pitch', 'Im_RAO_Mwind_vel_bend', 'Re_RAO_Mwind_acc_surge', 'Re_RAO_Mwind_acc_pitch', 'Re_RAO_Mwind_acc_bend', 'Im_RAO_Mwind_acc_surge', 'Im_RAO_Mwind_acc_pitch', 'Im_RAO_Mwind_acc_bend', 'dmoment_dv', 'moment_wind'], promotes_outputs=['Re_RAO_Mwind_hull_moment', 'Im_RAO_Mwind_hull_moment'])

	 	self.add_subsystem('hull_moment_spectrum', HullMomentSpectrum(freqs=freqs), promotes_inputs=['Re_RAO_wave_hull_moment', 'Im_RAO_wave_hull_moment', 'Re_RAO_wind_hull_moment', 'Im_RAO_wind_hull_moment', 'Re_RAO_Mwind_hull_moment', 'Im_RAO_Mwind_hull_moment', 'S_wave', 'S_wind'], promotes_outputs=['resp_hull_moment'])

	 	self.add_subsystem('hull_stress_spectrum', HullStressSpectrum(freqs=freqs), promotes_inputs=['resp_hull_moment', 'D_spar_p', 'wt_spar_p'], promotes_outputs=['resp_hull_stress'])

	 	self.add_subsystem('tower_fatigue', TowerFatigue(freqs=freqs), promotes_inputs=['resp_tower_stress', 'wt_tower_p'], promotes_outputs=['tower_fatigue_damage'])

	 	self.add_subsystem('hull_fatigue', HullFatigue(freqs=freqs), promotes_inputs=['resp_hull_stress', 'wt_spar_p'], promotes_outputs=['hull_fatigue_damage'])