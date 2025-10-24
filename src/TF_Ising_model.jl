"""
```julia
TFIM_MPO(sites::Vector{<:Index}, h::Real, J::Real; periodic = false)
```

Creates the Transverse Field Ising Model (TFIM) Hamiltonian as an MPO for spin-1/2 sites.

The TFIM Hamiltonian is given by:
```math
H = -J \\sum_{j=1}^{L-1} S_j^x S_{j+1}^x - h \\sum_{j=1}^L S_j^z
```

where:
- J is the Ising coupling strength (ferromagnetic if J > 0)
- h is the transverse magnetic field strength

This model exhibits a quantum phase transition at h_c = J between an ordered
ferromagnetic phase (h < J) and a disordered paramagnetic phase (h > J).

# Arguments
- `sites::Vector{<:Index}`: ITensors site indices for spin-1/2 chain
- `h::Real`: Transverse field strength  
- `J::Real`: Ising coupling parameter

# Keyword Arguments
- `periodic::Bool=false`: Whether to use periodic boundary conditions

# Returns
- `MPO`: Matrix Product Operator representation of the TFIM Hamiltonian

# Example
```julia
L = 20
sites = siteinds("S=1/2", L)
H = TFIM_MPO(sites, 0.8, 1.0)  # Close to critical point
```

# References
- Sachdev, Subir. "Quantum phase transitions." Cambridge University Press (2011).
- Pfeuty, Pierre. "The one-dimensional Ising model with a transverse field." Annals of Physics 57.1 (1970): 79-90.
"""
function TFIM_MPO(sites::Vector{<:Index}, h::Real, J::Real; periodic = false)
    L = length(sites)

    @assert hastags(sites[1], "S=1/2") "The sites must be spin-1/2 sites for the TFIM model."

    # create MPO
    os = OpSum()
    for j=1:L-1
        os += -J,"Sx",j,"Sx",j+1
        os += -h,"Sz",j
    end

    # PBC
    if(periodic)
        os += -J,"Sx",L,"Sx",1
    end

    # add last term
    os += -h,"Sz",L
    
    H = MPO(os,sites)

    return H
end

"""
```julia
TFIM_bdg(L::Int, h::Real, J::Real; periodic::Bool = false, parity::Int = 1)
```

Creates the BdG Hamiltonian matrix for the Transverse Field Ising Model.

The TFIM can be mapped to free fermions via Jordan-Wigner transformation, 
resulting in a quadratic fermionic Hamiltonian that can be written in BdG form.

# Arguments
- `L::Int`: System size (number of spins)
- `h::Real`: Transverse field strength
- `J::Real`: Ising coupling parameter

# Keyword Arguments
- `periodic::Bool=false`: Periodic boundary conditions
- `parity::Int=1`: Parity sector for periodic boundary conditions

# Returns
- `Matrix{Float64}`: 2L×2L BdG Hamiltonian matrix

"""
function TFIM_bdg(L::Int, h::Real, J::Real; periodic::Bool = false, parity::Int = 1)
    T = spdiagm(0 => h .* ones(L), 1 => -J/4 .* ones(L-1), -1 => -J/4 .* ones(L-1))
    D = spdiagm(1 => -J/4 .* ones(L-1), -1 => J/4 .* ones(L-1))
    
	H = zeros(Float64, 2*L, 2*L)
	
    if periodic
        T[1, end] = J/4 * parity
        T[end, 1] = J/4 * parity
        D[1, end] = -J/4 * parity
        D[end, 1] = J/4 * parity
    end

	H[1:L,1:L] = T
	H[L+1:end,L+1:end] = -transpose(T)
	H[1:L,L+1:end] = D
	H[L+1:end,1:L] = -D
	
	return H
end
