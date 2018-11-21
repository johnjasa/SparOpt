import numpy as np

from openmdao.api import Group

#from moor_ten_zero import MoorTenZero
from mooring_offset import MooringOffset
from diff_moor_ten import DiffMoorTen
from mooring_mass import MooringMass
from mooring_stiffness import MooringStiffness

class Mooring(Group):

	 def setup(self):
	 	#self.add_subsystem('moor_ten_zero', MoorTenZero(), promotes_inputs=['z_moor', 'water_depth', 'EA_moor', 'mass_dens_moor', 'len_hor_moor', 'len_tot_moor'], promotes_outputs=['moor_tension_zero', 'eff_length_zero'])

	 	self.add_subsystem('mooring_offset', MooringOffset(), promotes_inputs=['thrust_0', 'z_moor', 'water_depth', 'EA_moor', 'mass_dens_moor', 'len_hor_moor', 'len_tot_moor'], promotes_outputs=['moor_tension_offset_ww', 'eff_length_offset_ww', 'moor_tension_offset_lw', 'eff_length_offset_lw', 'moor_offset'])

	 	self.add_subsystem('diff_moor_ten', DiffMoorTen(), promotes_inputs=['EA_moor', 'mass_dens_moor', 'eff_length_offset_ww', 'moor_tension_offset_ww', 'eff_length_offset_lw', 'moor_tension_offset_lw'], promotes_outputs=['deff_length_ww_dx', 'dmoor_tension_ww_dx', 'deff_length_lw_dx', 'dmoor_tension_lw_dx'])

	 	self.add_subsystem('mooring_mass', MooringMass(), promotes_inputs=['eff_length_offset_ww', 'eff_length_offset_lw', 'mass_dens_moor'], promotes_outputs=['M_moor'])

	 	self.add_subsystem('mooring_stiffness', MooringStiffness(), promotes_inputs=['dmoor_tension_ww_dx', 'dmoor_tension_lw_dx'], promotes_outputs=['K_moor'])