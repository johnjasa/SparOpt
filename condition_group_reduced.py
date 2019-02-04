import numpy as np

from openmdao.api import Group, DirectSolver, BroydenSolver, NonlinearBlockGS, LinearBlockGS, ExecComp

from steady_bldpitch import SteadyBladePitch
from steady_rotspeed import SteadyRotSpeed
from gain_schedule import GainSchedule
from mooring_chain import MooringChain
from aero_group import Aero
from taper_hull import TaperHull
from taper_tower import TaperTower
from towerdim_group import Towerdim
from mean_tower_drag import MeanTowerDrag
from mooring_group import Mooring
from substructure_group import Substructure
from statespace_group import StateSpace
from wave_spectrum import WaveSpectrum
from wind_spectrum import WindSpectrum
from interp_wave_forces import InterpWaveForces
from viscous_group import Viscous
from postpro_group_reduced import Postpro
from tower_buckling_group import TowerBuckling
from extreme_response_group_reduced import ExtremeResponse
from cost_group import Cost

class Condition(Group):

	def initialize(self):
		self.options.declare('blades', types=dict)
		self.options.declare('freqs', types=dict)

	def setup(self):
		blades = self.options['blades']
		freqs = self.options['freqs']

	 	self.add_subsystem('steady_rotspeed', SteadyRotSpeed(), promotes_inputs=['windspeed_0'], promotes_outputs=['rotspeed_0'])

		self.add_subsystem('steady_bldpitch', SteadyBladePitch(), promotes_inputs=['windspeed_0'], promotes_outputs=['bldpitch_0'])

		self.add_subsystem('gain_schedule', GainSchedule(), promotes_inputs=['bldpitch_0'], promotes_outputs=['gain_corr_factor'])

		self.add_subsystem('mooring_chain', MooringChain(), promotes_inputs=['D_moor', 'gamma_F_moor'], promotes_outputs=['mass_dens_moor', 'EA_moor', 'maxval_moor_ten'])

		aero_group = Aero(blades=blades, freqs=freqs)

		self.add_subsystem('aero', aero_group, promotes_inputs=['rho_wind', 'windspeed_0', 'bldpitch_0', 'rotspeed_0'], promotes_outputs=['thrust_wind', \
			'moment_wind', 'torque_wind', 'thrust_0', 'torque_0', 'dthrust_dv', 'dmoment_dv', 'dtorque_dv', 'dthrust_drotspeed', 'dtorque_drotspeed', \
			'dthrust_dbldpitch', 'dtorque_dbldpitch'])

		self.add_subsystem('taper_hull', TaperHull(), promotes_inputs=['D_spar_p'], promotes_outputs=['taper_hull'])

		self.add_subsystem('taper_tower', TaperTower(), promotes_inputs=['D_tower_p'], promotes_outputs=['taper_tower'])

		towerdim_group = Towerdim()

		self.add_subsystem('towerdim', towerdim_group, promotes_inputs=['D_tower_p', 'L_tower'], promotes_outputs=['D_tower', 'Z_tower'])

		self.add_subsystem('mean_tower_drag', MeanTowerDrag(), promotes_inputs=['D_tower', 'Z_tower', 'L_tower', 'windspeed_0', 'Cd_tower', 'CoG_rotor', 'rho_wind'], \
			promotes_outputs=['F0_tower_drag', 'Z0_tower_drag'])

		mooring_group = Mooring()

		mooring_group.linear_solver = DirectSolver(assemble_jac=True)

		self.add_subsystem('mooring', mooring_group, promotes_inputs=['z_moor', 'water_depth', 'EA_moor', 'mass_dens_moor', 'len_hor_moor', 'len_tot_moor', \
			'thrust_0', 'F0_tower_drag'], promotes_outputs=['M_moor_zero', 'K_moor', 'M_moor', 'moor_offset', 'maxval_fairlead', 'mean_moor_ten'])

		substructure_group = Substructure(freqs=freqs)

		substructure_group.linear_solver = DirectSolver(assemble_jac=True)

		self.add_subsystem('substructure', substructure_group, promotes_inputs=['D_spar_p', 'L_spar', 'wt_spar_p', 'L_tower', 'wt_tower_p', \
			'rho_ball', 'wt_ball', 'M_nacelle', 'M_rotor', 'CoG_nacelle', 'CoG_rotor', 'I_rotor', 'water_depth', 'z_moor', 'M_moor_zero', 'K_moor', 'M_moor', \
			'dthrust_dv', 'dmoment_dv', 'struct_damp_ratio', 't_w_stiff', 't_f_stiff', 'h_stiff', 'b_stiff', 'l_stiff', 'D_tower', 'Z_tower'], \
			promotes_outputs=['M_global', 'A_global', 'K_global', 'Re_wave_forces', 'Im_wave_forces', 'x_d_towertop', 'z_sparnode', 'x_sparelem', \
			'Z_spar', 'M_spar', 'M_ball', 'L_ball', 'spar_draft', 'D_spar', 'wt_spar', 'wt_tower', 'tot_M_spar', 'tot_M_tower', 'B_aero_11', \
			'B_aero_15', 'B_aero_17', 'B_aero_55', 'B_aero_57', 'B_aero_77', 'B_struct_77', 'A_R', 'r_e', 'buoy_spar', 'CoB', 'M_turb', 'CoG_total', \
			'wave_number', 'x_sparnode', 'M_ball_elem', 'M_tower', 'z_towernode'])

		statespace_group = StateSpace(freqs=freqs)

		statespace_group.linear_solver = DirectSolver(assemble_jac=True)

		self.add_subsystem('statespace', statespace_group, promotes_inputs=['M_global', 'A_global', 'K_global', 'CoG_rotor', 'I_d', 'dthrust_dv', \
			'dmoment_dv', 'dtorque_dv', 'dthrust_drotspeed', 'dthrust_dbldpitch', 'dtorque_dbldpitch', 'omega_lowpass', 'omega_notch', 'bandwidth_notch', 'k_i', 'k_p', 'k_t', \
			'gain_corr_factor', 'x_d_towertop', 'windspeed_0', 'rotspeed_0'], promotes_outputs=['Astr_stiff', 'Astr_ext', 'A_contrl', 'BsCc', 'BcCs', 'B_feedbk'])

		self.add_subsystem('wave_spectrum', WaveSpectrum(freqs=freqs), promotes_inputs=['Hs', 'Tp'], promotes_outputs=['S_wave'])

		self.add_subsystem('wind_spectrum', WindSpectrum(freqs=freqs), promotes_inputs=['windspeed_0'], promotes_outputs=['S_wind'])

		self.add_subsystem('interp_wave_forces', InterpWaveForces(freqs=freqs), promotes_inputs=['Re_wave_forces', 'Im_wave_forces'], promotes_outputs=[\
			'Re_wave_force_surge', 'Im_wave_force_surge', 'Re_wave_force_pitch', 'Im_wave_force_pitch', 'Re_wave_force_bend', 'Im_wave_force_bend'])

		viscous_group = Viscous(freqs=freqs)

		viscous_group.linear_solver = LinearBlockGS(maxiter=50)
		viscous_group.nonlinear_solver = NonlinearBlockGS(maxiter=50, atol=1e-8, rtol=1e-8)

		self.add_subsystem('viscous', viscous_group, promotes_inputs=['Cd', 'x_sparelem', 'z_sparnode', 'Z_spar', 'D_spar', 'B_aero_11', 'B_aero_15', \
			'B_aero_17', 'B_aero_55', 'B_aero_57', 'B_aero_77', 'B_struct_77', 'M_global', 'A_global', 'CoG_rotor', 'I_d', 'dtorque_dv', 'dtorque_drotspeed', \
			'Astr_stiff', 'Astr_ext', 'A_contrl', 'BsCc', 'BcCs', 'B_feedbk', 'Re_wave_force_surge', 'Im_wave_force_surge', 'Re_wave_force_pitch', \
			'Im_wave_force_pitch', 'Re_wave_force_bend', 'Im_wave_force_bend', 'thrust_wind', 'moment_wind', 'torque_wind', 'S_wave', 'S_wind'], \
			promotes_outputs=['A_feedbk', 'Re_H_feedbk', 'Im_H_feedbk', 'Re_RAO_wave_surge', 'Im_RAO_wave_surge', 'Re_RAO_wave_pitch', 'Im_RAO_wave_pitch', 'Re_RAO_wave_bend', \
			'Im_RAO_wave_bend', 'Re_RAO_wind_surge', 'Im_RAO_wind_surge', 'Re_RAO_wind_pitch', 'Im_RAO_wind_pitch', 'Re_RAO_wind_bend', 'Im_RAO_wind_bend', \
			'Re_RAO_Mwind_surge', 'Im_RAO_Mwind_surge', 'Re_RAO_Mwind_pitch', 'Im_RAO_Mwind_pitch', 'Re_RAO_Mwind_bend', 'Im_RAO_Mwind_bend', 'Re_RAO_wave_vel_surge', \
			'Im_RAO_wave_vel_surge', 'Re_RAO_wave_vel_pitch', 'Im_RAO_wave_vel_pitch', 'Re_RAO_wave_vel_bend', 'Im_RAO_wave_vel_bend', 'Re_RAO_wind_vel_surge', \
			'Im_RAO_wind_vel_surge', 'Re_RAO_wind_vel_pitch', 'Im_RAO_wind_vel_pitch', 'Re_RAO_wind_vel_bend', 'Im_RAO_wind_vel_bend', 'Re_RAO_Mwind_vel_surge', \
			'Im_RAO_Mwind_vel_surge', 'Re_RAO_Mwind_vel_pitch', 'Im_RAO_Mwind_vel_pitch', 'Re_RAO_Mwind_vel_bend', 'Im_RAO_Mwind_vel_bend', 'B_visc_11', 'stddev_vel_distr'])

		postpro_group = Postpro(freqs=freqs)

		self.add_subsystem('postpro', postpro_group, promotes_inputs=['Re_wave_force_surge', 'Im_wave_force_surge', 'Re_wave_force_pitch', 'Im_wave_force_pitch', \
			'Re_wave_force_bend', 'Im_wave_force_bend', 'thrust_wind', 'moment_wind', 'torque_wind', 'Re_H_feedbk', 'Im_H_feedbk', 'k_i', 'k_p', 'gain_corr_factor', \
			'S_wave', 'S_wind', 'Re_H_feedbk', 'Im_H_feedbk', 'Re_RAO_wave_surge', 'Im_RAO_wave_surge', 'Re_RAO_wave_pitch', 'Im_RAO_wave_pitch', 'Re_RAO_wave_bend', \
			'Im_RAO_wave_bend', 'Re_RAO_wind_surge', 'Im_RAO_wind_surge', 'Re_RAO_wind_pitch', 'Im_RAO_wind_pitch', 'Re_RAO_wind_bend', 'Im_RAO_wind_bend', \
			'Re_RAO_Mwind_surge', 'Im_RAO_Mwind_surge', 'Re_RAO_Mwind_pitch', 'Im_RAO_Mwind_pitch', 'Re_RAO_Mwind_bend', 'Im_RAO_Mwind_bend', 'Re_RAO_wave_vel_surge', \
			'Im_RAO_wave_vel_surge', 'Re_RAO_wave_vel_pitch', 'Im_RAO_wave_vel_pitch', 'Re_RAO_wave_vel_bend', 'Im_RAO_wave_vel_bend', 'Re_RAO_wind_vel_surge', \
			'Im_RAO_wind_vel_surge', 'Re_RAO_wind_vel_pitch', 'Im_RAO_wind_vel_pitch', 'Re_RAO_wind_vel_bend', 'Im_RAO_wind_vel_bend', 'Re_RAO_Mwind_vel_surge', \
			'Im_RAO_Mwind_vel_surge', 'Re_RAO_Mwind_vel_pitch', 'Im_RAO_Mwind_vel_pitch', 'Re_RAO_Mwind_vel_bend', 'Im_RAO_Mwind_vel_bend', 'D_tower_p', 'wt_tower_p', \
			'Z_tower', 'dthrust_dv', 'dmoment_dv', 'dthrust_drotspeed', 'dthrust_dbldpitch', 'M_tower', 'M_nacelle', 'M_rotor', 'I_rotor', 'CoG_nacelle', 'CoG_rotor', \
			'z_towernode', 'x_towerelem', 'x_towernode', 'x_d_towertop', 'moor_offset', 'z_moor', 'K_moor', 'thrust_0', 'buoy_spar', 'CoB', 'M_turb', 'tot_M_spar', \
			'M_ball', 'CoG_total', 'F0_tower_drag', 'Z0_tower_drag', 'windspeed_0'], promotes_outputs=['stddev_surge', 'stddev_pitch', 'stddev_bend', 'stddev_rotspeed', \
			'stddev_bldpitch', 'stddev_tower_stress', 'stddev_fairlead', 'stddev_moor_ten', 'mean_surge', 'mean_pitch', 'mean_tower_stress', 'v_z_surge', \
			'v_z_pitch', 'v_z_tower_stress', 'v_z_fairlead', 'v_z_moor_ten', 'tower_fatigue_damage'])

		tower_buckling_group = TowerBuckling()

		self.add_subsystem('tower_buckling', tower_buckling_group, promotes_inputs=['L_tower', 'D_tower_p', 'wt_tower_p', 'f_y', 'gamma_M_tower', 'gamma_F_tower'], \
			promotes_outputs=['maxval_tower_stress'])

		extreme_response_group = ExtremeResponse()

		self.add_subsystem('extreme_response', extreme_response_group, promotes_inputs=['maxval_surge', 'maxval_pitch', 'maxval_tower_stress', 'maxval_fairlead', \
			'maxval_moor_ten', 'stddev_surge', 'stddev_pitch', 'stddev_tower_stress', 'stddev_fairlead', 'stddev_moor_ten', 'mean_surge', 'mean_pitch', 'mean_tower_stress', \
			'moor_offset', 'mean_moor_ten', 'v_z_surge', 'v_z_pitch', 'v_z_tower_stress', 'v_z_fairlead', 'v_z_moor_ten', 'gamma_F_moor_mean', 'gamma_F_moor_dyn'], \
			promotes_outputs=['constr_50_surge', 'constr_50_pitch', 'constr_50_tower_stress', 'constr_50_fairlead', 'constr_50_moor_ten'])

		cost_group = Cost()

		self.add_subsystem('cost', cost_group, promotes_inputs=['D_spar', 'D_spar_p', 'wt_spar', 'L_spar', 'l_stiff', 'h_stiff', 't_f_stiff', 'A_R', 'r_f', 'r_e', \
			'tot_M_spar', 'D_tower', 'D_tower_p', 'wt_tower', 'L_tower', 'tot_M_tower', 'len_tot_moor', 'mass_dens_moor'], promotes_outputs=['spar_cost', 'tower_cost', \
			'mooring_cost', 'total_cost'])