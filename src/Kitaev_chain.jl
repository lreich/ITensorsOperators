"""
```julia
kitaev_model(μ::Real, Δ::Real, J::Real, sites::Vector{<:Index}; type::String = "spin")
```

Routing function for different Kitaev chain types.

where:
- J is the hopping parameter
- Δ is the superconducting pairing parameter
- μ is the chemical potential

# Arguments
- `μ::Real`: Chemical potential
- `Δ::Real`: Superconducting pairing parameter  
- `J::Real`: Hopping parameter
- `sites::Vector{<:Index}`: ITensors site indices for spin-1/2 chain

# Returns
- `MPO`: Matrix Product Operator representation of the Kitaev chain Hamiltonian
"""
function kitaev_model(μ::Real, Δ::Real, J::Real, sites::Vector{<:Index}; type::String = "spin")
    if type == "spin"
        return kitaev_model(Val(:Spin), μ, Δ, J, sites)
    else
        return kitaev_model(Val(:Fermion), μ, Δ, J, sites)
    end
end

"""
```julia
kitaev_model(::Val{:Spin}, μ::Real, Δ::Real, J::Real, sites::Vector{<:Index})
```

Creates the Kitaev chain Hamiltonian as an MPO for spin-1/2 sites.

The Kitaev chain Hamiltonian is given by:
```math
H = \\sum_{j=1}^{L-1} \\left[ -J (S_j^+ S_{j+1}^- + S_j^- S_{j+1}^+) - \\Delta (S_j^+ S_{j+1}^+ + S_j^- S_{j+1}^-) \\right] + \\mu \\sum_{j=1}^L S_j^z
```

where:
- J is the hopping parameter
- Δ is the superconducting pairing parameter
- μ is the chemical potential

# Arguments
- `μ::Real`: Chemical potential
- `Δ::Real`: Superconducting pairing parameter  
- `J::Real`: Hopping parameter
- `sites::Vector{<:Index}`: ITensors site indices for spin-1/2 chain

# Returns
- `MPO`: Matrix Product Operator representation of the Kitaev chain Hamiltonian
"""
function kitaev_model(::Val{:Spin}, μ::Real, Δ::Real, J::Real, sites::Vector{<:Index})
	L = length(sites)
	os = OpSum()

    @assert hastags(sites[1], "S=1/2") "The sites must be spin-1/2 sites for the Kitaev model."

	for j in 1:L-1
        os += -J, "S+", j, "S-", j+1
        os += -J, "S+", j+1, "S-", j
        
        os += -Δ, "S+", j+1, "S+", j
        os += -Δ, "S-", j, "S-", j+1
		os += -μ, "Sz", j
	end
    # add last part
    os += -μ, "Sz", L

	return MPO(os, sites)
end

"""
```julia
kitaev_model(::Val{:Fermion}, μ::Real, Δ::Real, J::Real, sites::Vector{<:Index})
```

Creates the Kitaev chain Hamiltonian as an MPO for fermionic sites.

The Kitaev chain Hamiltonian is given by:
```math
H = \\sum_{j=1}^{L-1} \\left[ -J (C_j^† C_{j+1} + C_{j+1}^† C_j) - \\Delta (C_j^† C_{j+1}^† + C_{j+1} C_j) \\right] + \\mu \\sum_{j=1}^L N_j
```

where:
- J is the hopping parameter
- Δ is the superconducting pairing parameter
- μ is the chemical potential

# Arguments
- `μ::Real`: Chemical potential
- `Δ::Real`: Superconducting pairing parameter  
- `J::Real`: Hopping parameter
- `sites::Vector{<:Index}`: ITensors site indices for fermionic chain

# Returns
- `MPO`: Matrix Product Operator representation of the Kitaev chain Hamiltonian
"""
function kitaev_model(::Val{:Fermion}, μ::Real, Δ::Real, J::Real, sites::Vector{<:Index})
	L = length(sites)
	os = OpSum()

    @assert hastags(sites[1], "Fermion") "The sites must be fermionic sites for the Kitaev model."

	for j in 1:L-1
        os += -J, "Cdag", j, "C", j+1
        os += -J, "Cdag", j+1, "C", j

        os += Δ, "Cdag", j+1, "Cdag", j
        os += Δ, "C", j, "C", j+1
		os += -μ, "N", j
        os += μ/2,"Id",j 
	end
    # add last part
    os += -μ, "N", L
    os += μ/2,"Id",L

	return MPO(os, sites)
end

"""
```julia
kitaev_model_bdg(μ::Real, Δ::Real, J::Real, L::Int; periodic=false, parity=1)
```

Creates the Bogoliubov-de Gennes (BdG) Hamiltonian matrix for the Kitaev chain model.

# Arguments
- `μ::Real`: Chemical potential
- `Δ::Real`: Superconducting pairing parameter
- `J::Real`: Hopping parameter
- `L::Int`: System size (number of sites)

# Keyword Arguments
- `periodic::Bool=false`: Whether to use periodic boundary conditions
- `parity::Int=1`: Parity sector for periodic boundary conditions

# Returns
- `Matrix{Float64}`: 2L×2L BdG Hamiltonian matrix

"""
function kitaev_model_bdg(μ::Real, Δ::Real, J::Real, L::Int; periodic=false, parity=1)
    T = spdiagm(0 => -μ .* ones(L), 1 => -J .* ones(L-1), -1 => -J .* ones(L-1))
    D = spdiagm(1 => -Δ .* ones(L-1), -1 => Δ .* ones(L-1))
    
    if periodic
        T[1, end] = J * parity
        T[end, 1] = J * parity
        D[1, end] = -Δ * parity
        D[end, 1] = Δ * parity
    end

    H = zeros(Float64, 2*L, 2*L)
    H[1:L,1:L] = T
	H[L+1:end,L+1:end] = -transpose(T)
	H[1:L,L+1:end] = D
	H[L+1:end,1:L] = -D

	return H
end