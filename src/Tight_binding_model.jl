"""
    Tight_binding_MPO(sites::Vector{<:Index}; t=1.0)

Creates tight binding Hamiltonian MPO using spin operators for spin-1/2 sites.

The tight binding Hamiltonian using spin operators is given by:
```math
H = -t \\sum_{j=1}^{L-1} \\left( S_j^+ S_{j+1}^- + S_j^- S_{j+1}^+ \\right)
```

# Arguments
- `sites::Vector{<:Index}`: ITensors site indices for spin-1/2 chain
- `t::Real=1.0`: Hopping parameter

# Returns
- `MPO`: Matrix Product Operator representation of the tight binding Hamiltonian
"""
function Tight_binding_MPO(sites::Vector{<:Index}; 
    t::Real=1.0)
    
    @assert hastags(sites[1], "S=1/2") "The sites must be spin-1/2 sites for the tight binding model."

    L = length(sites)

    os = OpSum()

    for j in 1:L-1
        os += -t,"S+",j,"S-",j+1
        os += -t,"S-",j,"S+",j+1
    end

    return MPO(os,sites)
end

"""
    Tight_binding_bdg(::Val{:Standard}; L=20, t=1.0)

Creates BdG Hamiltonian for standard tight binding model.

# Arguments
- `L::Int=20`: System size (number of sites)
- `t::Real=1.0`: Hopping parameter

# Returns
- `Matrix{Float64}`: 2L×2L BdG Hamiltonian matrix
"""
function Tight_binding_bdg(L::Int=20; t::Real=1.0) 
    
    T = t .* spdiagm(1 => -1 .* ones(L-1), -1 => -1 .* ones(L-1))
    
	H = zeros(Float64, 2*L, 2*L)
	
	H[1:L,1:L] = T
	H[L+1:end,L+1:end] = -transpose(T)
	H[1:L,L+1:end] = zeros(L,L)
	H[L+1:end,1:L] = -zeros(L,L)
	
	return H
end
