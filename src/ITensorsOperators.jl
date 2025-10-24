module ITensorsOperators
#= load external modules =#
using ITensors, ITensorMPS 
# using SparseArrays, LinearAlgebra, IterTools, CUDA, JLD2


include("BLBQ_model.jl")

export decoupled_lattice,
	   BLBQ_Hamiltonian_MPO,
	   BLBQ_Hamiltonian_bdg,
	   string_correlator_MPO,
	   dimer_phase_order_param,
	   local_density_density_corr_fct,
	   cartesianToSpin1
end