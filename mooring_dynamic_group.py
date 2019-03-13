import numpy as np

from openmdao.api import Group

from mooring_upper_angle import MooringUpperAngle
from mooring_upper_tan_motion import MooringUpperTanMotion
from mooring_sig_surge_offset import MooringSigSurgeOffset
from mooring_offset_coords import MooringOffsetCoords
from mooring_sig_surge_offset_coords import MooringSigSurgeOffsetCoords
from moor_ten_sig_surge import MoorTenSigSurge
from mooring_angles import MooringAngles
from mooring_node_disp import MooringNodeDisp
from mooring_node_norm_disp import MooringNodeNormDisp
from mooring_gen_damp import MooringGenDamp
from mooring_gen_mass import MooringGenMass
from mooring_k_e import MooringKe
from mooring_k_g import MooringKg
from mooring_surge_damp import MooringSurgeDamp
from norm_moor_ten_dyn_wave import NormMoorTenDynWave
from norm_moor_ten_dyn_wind import NormMoorTenDynWind
from norm_moor_ten_dyn_Mwind import NormMoorTenDynMWind
from moor_ten_dyn_spectrum import MoorTenDynSpectrum
from std_dev_moor_ten_dyn import StdDevMoorTenDyn
from norm_moor_tan_vel_wave import NormMoorTanVelWave
from norm_moor_tan_vel_wind import NormMoorTanVelWind
from norm_moor_tan_vel_Mwind import NormMoorTanVelMWind
from moor_tan_vel_spectrum import MoorTanVelSpectrum
from std_dev_moor_tan_vel import StdDevMoorTanVel
from mooring_gen_damp_Q import MooringGenDampQ

class MooringDynamic(Group):

	def initialize(self):
		self.options.declare('freqs', types=dict)

	def setup(self):
		freqs = self.options['freqs']

		self.add_subsystem('mooring_upper_angle', MooringUpperAngle(), promotes_inputs=['mass_dens_moor', 'moor_tension_offset_ww', 'eff_length_offset_ww'], promotes_outputs=['phi_upper_end'])

		self.add_subsystem('mooring_upper_tan_motion', MooringUpperTanMotion(), promotes_inputs=['stddev_surge_WF', 'phi_upper_end'], promotes_outputs=['sig_tan_motion'])

		self.add_subsystem('mooring_sig_surge_offset', MooringSigSurgeOffset(), promotes_inputs=['z_moor', 'water_depth', 'EA_moor', 'mass_dens_moor', 'len_hor_moor', 'len_tot_moor', 'moor_offset', 'stddev_surge_WF'], promotes_outputs=['moor_tension_sig_surge_offset', 'eff_length_sig_surge_offset'])

		self.add_subsystem('mooring_offset_coords', MooringOffsetCoords(), promotes_inputs=['mass_dens_moor', 'moor_tension_offset_ww', 'eff_length_offset_ww', 'EA_moor'], promotes_outputs=['moor_offset_x', 'moor_offset_z'])

		self.add_subsystem('mooring_sig_surge_offset_coords', MooringSigSurgeOffsetCoords(), promotes_inputs=['mass_dens_moor', 'moor_tension_sig_surge_offset', 'eff_length_sig_surge_offset', 'EA_moor', 'stddev_surge_WF'], promotes_outputs=['moor_sig_surge_offset_x', 'moor_sig_surge_offset_z'])

		self.add_subsystem('moor_ten_sig_surge', MoorTenSigSurge(), promotes_inputs=['moor_tension_sig_surge_offset', 'z_moor', 'water_depth', 'EA_moor', 'mass_dens_moor'], promotes_outputs=['moor_ten_sig_surge_tot'])

		self.add_subsystem('mooring_angles', MooringAngles(), promotes_inputs=['mass_dens_moor', 'moor_tension_offset_ww', 'eff_length_offset_ww'], promotes_outputs=['phi_moor'])

		self.add_subsystem('mooring_node_disp', MooringNodeDisp(), promotes_inputs=['moor_offset_x', 'moor_offset_z', 'moor_sig_surge_offset_x', 'moor_sig_surge_offset_z'], promotes_outputs=['r_moor', 'beta_moor'])

		self.add_subsystem('mooring_node_norm_disp', MooringNodeNormDisp(), promotes_inputs=['sig_tan_motion', 'phi_moor', 'r_moor', 'beta_moor'], promotes_outputs=['norm_r_moor'])

		self.add_subsystem('mooring_gen_damp', MooringGenDamp(), promotes_inputs=['norm_r_moor', 'Cd_moor', 'D_moor', 'eff_length_offset_ww'], promotes_outputs=['gen_c_moor'])

		self.add_subsystem('mooring_gen_mass', MooringGenMass(), promotes_inputs=['mass_dens_moor', 'norm_r_moor', 'eff_length_offset_ww'], promotes_outputs=['gen_m_moor'])

		self.add_subsystem('mooring_k_e', MooringKe(), promotes_inputs=['EA_moor', 'eff_length_offset_ww'], promotes_outputs=['k_e_moor'])

		self.add_subsystem('mooring_k_g', MooringKg(), promotes_inputs=['sig_tan_motion', 'mass_dens_moor', 'moor_tension_offset_ww', 'eff_length_offset_ww', 'moor_tension_sig_surge_offset', 'eff_length_sig_surge_offset'], promotes_outputs=['k_g_moor'])

		#self.add_subsystem('mooring_surge_damp', MooringSurgeDamp(), promotes_inputs=['k_e_moor', 'k_g_moor', 'gen_c_moor', 'stddev_surge_vel_WF', 'phi_upper_end'], promotes_outputs=['moor_surge_damp'])

	 	self.add_subsystem('norm_moor_ten_dyn_wave', NormMoorTenDynWave(freqs=freqs), promotes_inputs=['Re_RAO_wave_fairlead', 'Im_RAO_wave_fairlead', 'phi_upper_end', 'k_e_moor', 'k_g_moor', 'gen_m_moor', 'gen_c_moor'], promotes_outputs=['Re_RAO_wave_moor_ten_dyn', 'Im_RAO_wave_moor_ten_dyn'])

	 	self.add_subsystem('norm_moor_ten_dyn_wind', NormMoorTenDynWind(freqs=freqs), promotes_inputs=['Re_RAO_wind_fairlead', 'Im_RAO_wind_fairlead', 'phi_upper_end', 'k_e_moor', 'k_g_moor', 'gen_m_moor', 'gen_c_moor'], promotes_outputs=['Re_RAO_wind_moor_ten_dyn', 'Im_RAO_wind_moor_ten_dyn'])

		self.add_subsystem('norm_moor_ten_dyn_Mwind', NormMoorTenDynMWind(freqs=freqs), promotes_inputs=['Re_RAO_Mwind_fairlead', 'Im_RAO_Mwind_fairlead', 'phi_upper_end', 'k_e_moor', 'k_g_moor', 'gen_m_moor', 'gen_c_moor'], promotes_outputs=['Re_RAO_Mwind_moor_ten_dyn', 'Im_RAO_Mwind_moor_ten_dyn'])

	 	self.add_subsystem('moor_ten_dyn_spectrum', MoorTenDynSpectrum(freqs=freqs), promotes_inputs=['Re_RAO_wave_moor_ten_dyn', 'Im_RAO_wave_moor_ten_dyn', 'Re_RAO_wind_moor_ten_dyn', 'Im_RAO_wind_moor_ten_dyn', 'Re_RAO_Mwind_moor_ten_dyn', 'Im_RAO_Mwind_moor_ten_dyn', 'S_wave', 'S_wind'], promotes_outputs=['resp_moor_ten_dyn'])

	 	self.add_subsystem('std_dev_moor_ten_dyn', StdDevMoorTenDyn(freqs=freqs), promotes_inputs=['resp_moor_ten_dyn'], promotes_outputs=['stddev_moor_ten_dyn'])

	 	self.add_subsystem('norm_moor_tan_vel_wave', NormMoorTanVelWave(freqs=freqs), promotes_inputs=['Re_RAO_wave_fairlead', 'Im_RAO_wave_fairlead', 'phi_upper_end', 'k_e_moor', 'k_g_moor', 'gen_m_moor', 'gen_c_moor'], promotes_outputs=['Re_RAO_wave_moor_tan_vel', 'Im_RAO_wave_moor_tan_vel'])

	 	self.add_subsystem('norm_moor_tan_vel_wind', NormMoorTanVelWind(freqs=freqs), promotes_inputs=['Re_RAO_wind_fairlead', 'Im_RAO_wind_fairlead', 'phi_upper_end', 'k_e_moor', 'k_g_moor', 'gen_m_moor', 'gen_c_moor'], promotes_outputs=['Re_RAO_wind_moor_tan_vel', 'Im_RAO_wind_moor_tan_vel'])

		self.add_subsystem('norm_moor_tan_vel_Mwind', NormMoorTanVelMWind(freqs=freqs), promotes_inputs=['Re_RAO_Mwind_fairlead', 'Im_RAO_Mwind_fairlead', 'phi_upper_end', 'k_e_moor', 'k_g_moor', 'gen_m_moor', 'gen_c_moor'], promotes_outputs=['Re_RAO_Mwind_moor_tan_vel', 'Im_RAO_Mwind_moor_tan_vel'])

	 	self.add_subsystem('moor_tan_vel_spectrum', MoorTanVelSpectrum(freqs=freqs), promotes_inputs=['Re_RAO_wave_moor_tan_vel', 'Im_RAO_wave_moor_tan_vel', 'Re_RAO_wind_moor_tan_vel', 'Im_RAO_wind_moor_tan_vel', 'Re_RAO_Mwind_moor_tan_vel', 'Im_RAO_Mwind_moor_tan_vel', 'S_wave', 'S_wind'], promotes_outputs=['resp_moor_tan_vel'])

	 	self.add_subsystem('std_dev_moor_tan_vel', StdDevMoorTanVel(freqs=freqs), promotes_inputs=['resp_moor_tan_vel'], promotes_outputs=['stddev_moor_tan_vel'])

	 	self.add_subsystem('mooring_gen_damp_Q', MooringGenDampQ(), promotes_inputs=['gen_c_moor', 'stddev_moor_tan_vel'], promotes_outputs=['gen_c_moor_Q'])