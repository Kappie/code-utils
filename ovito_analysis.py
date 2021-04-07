#!/home/geert/Code/Python/env3.8/bin/python
import numpy as np
from ovito.io import import_file, export_file
from ovito.modifiers import *
import argparse 
from io import StringIO

p = argparse.ArgumentParser()
p.add_argument("--input_file", type=str)
p.add_argument("--centrosymmetry", action="store_true")
p.add_argument("--ackland-jones", action="store_true")
p.add_argument("--acna", action="store_true")
args = p.parse_args()

input_file = args.input_file

pipeline = import_file(input_file)
pipeline.source.data.cell_.pbc = (True,True,True)

pipeline.modifiers.append(AcklandJonesModifier())
pipeline.modifiers.append(CentroSymmetryModifier(mode=CentroSymmetryModifier.Mode.Conventional, num_neighbors=12))
data = pipeline.compute()

structure_types_ackland = data.particles.structure_types.array

# Default, as described in Structure identification methods for atomistic simulations of crystalline materials, 2012.
pipeline.modifiers.append(CommonNeighborAnalysisModifier(mode=CommonNeighborAnalysisModifier.Mode.AdaptiveCutoff))
data = pipeline.compute()

structure_types_acna = data.particles.structure_types.array
centrosymmetry = data.particles.centrosymmetry.array

cols = np.column_stack((centrosymmetry, structure_types_ackland, structure_types_acna))
data_string = StringIO()
np.savetxt(data_string, cols, fmt="%g %d %d")

print(data_string.getvalue())
