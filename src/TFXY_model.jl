"""
```julia
TF_XY_Model(sites::Vector{<:Index}; type::String="Anisotropic", kwargs...)
```

Main dispatcher function for creating Transverse Field XY Model Hamiltonian MPOs.

# Arguments
- `sites::Vector{<:Index}`: ITensor sites object defining the system
- `type::String`: Parameterization type ("Anisotropic", "JxJy")
- `kwargs...`: Additional keyword arguments passed to specific implementations

# Type Options
- `"Anisotropic"`: Uses J and γ parameters
- `"JxJy"`: Uses separate Jx and Jy parameters

# Returns
- MPO representation of the TFXY Hamiltonian

# Examples
```julia
sites = siteinds("S=1/2", 10)
H1 = TF_XY_Model(sites; type="Anisotropic", J=1.0, γ=0.5, h=0.8)
H2 = TF_XY_Model(sites; type="JxJy", Jx=0.75, Jy=0.25, h=0.8)

```
"""
function TF_XY_Model(sites::Vector{<:Index}; 
    type::String="Anisotropic", 
    kwargs...)

    # add new types here if you want
    type = begin
        if type == "Anisotropic"
            Val(:Anisotropic)
        elseif type == "JxJy"
            Val(:JxJy)
        else
            error("Unknown type: $type. Use 'Anisotropic' or 'JxJy'.")
        end
    end

    return TF_XY_Model(type, sites; kwargs...)
end

"""
    TF_XY_Model(::Val{:Anisotropic}, sites::Vector{<:Index}; J=1.0, γ=0.5, h=1.0, periodic=false)

Creates TFXY Hamiltonian MPO using anisotropy parameterization (J, γ).

The TFXY Hamiltonian is given by:
```math
H = -J \\sum_{j=1}^{L-1} \\left[ (1+γ) S_j^x S_{j+1}^x + (1-γ) S_j^y S_{j+1}^y \\right] - h \\sum_{j=1}^L S_j^z
```

# Arguments
- `sites::Vector{<:Index}`: ITensors site indices for spin-1/2 chain
- `J::Real=1.0`: Exchange coupling parameter
- `γ::Real=0.5`: Anisotropy parameter (0 ≤ γ ≤ 1)
- `h::Real=1.0`: Transverse field strength
- `periodic::Bool=false`: Whether to use periodic boundary conditions

# Returns
- `MPO`: Matrix Product Operator representation of the TFXY Hamiltonian
"""
function TF_XY_Model(::Val{:Anisotropic}, sites::Vector{<:Index}; 
    J::Real=1.0, 
    γ::Real=0.5, 
    h::Real=1.0, 
    periodic::Bool=false)
    
    L = length(sites)

    @assert hastags(sites[1], "S=1/2") "The sites must be spin-1/2 sites for the TFXY model."
    @assert 0 <= γ <= 1 "Anisotropy parameter γ must be in the range [0, 1]."

    # create MPO
    os = OpSum()

    for j=1:L-1
        os += -2J*(1+γ),"Sx",j,"Sx",j+1
        os += -2J*(1-γ),"Sy",j,"Sy",j+1
        os += -2h,"Sz",j
    end

    if(periodic)
        # implement periodic bc
        os += -2J*(1+γ),"Sx",L,"Sx",1
        os += -2J*(1-γ),"Sy",L,"Sy",1
    end

    # add last term
    os += -2h,"Sz",L

    H = MPO(os,sites)

    return H
end

"""
    TF_XY_Model(::Val{:JxJy}, sites::Vector{<:Index}; Jx=1.0, Jy=1.0, h=1.0, periodic=false)

Creates TFXY Hamiltonian MPO using separate x and y coupling parameters (Jx, Jy).

The TFXY Hamiltonian is given by:
```math
H = - \\sum_{j=1}^{L-1} \\left[ J_x S_j^x S_{j+1}^x + J_y S_j^y S_{j+1}^y \\right] - h \\sum_{j=1}^L S_j^z
```

# Arguments
- `sites::Vector{<:Index}`: ITensors site indices for spin-1/2 chain
- `Jx::Real=1.0`: Exchange coupling in x-direction
- `Jy::Real=1.0`: Exchange coupling in y-direction  
- `h::Real=1.0`: Transverse field strength
- `periodic::Bool=false`: Whether to use periodic boundary conditions

# Returns
- `MPO`: Matrix Product Operator representation of the TFXY Hamiltonian
"""
function TF_XY_Model(::Val{:JxJy}, sites::Vector{<:Index}; 
    Jx::Real=1.0, 
    Jy::Real=1.0, 
    h::Real=1.0, 
    periodic::Bool=false)
    
    L = length(sites)

    @assert hastags(sites[1], "S=1/2") "The sites must be spin-1/2 sites for the TFXY model."

    # create MPO
    os = OpSum()

    for j=1:L-1
        os += 4*Jx,"Sx",j,"Sx",j+1
        os += 4*Jy,"Sy",j,"Sy",j+1
        os += 2*h,"Sz",j
    end

    if(periodic)
        # implement periodic bc
        os += 4*Jx,"Sx",L,"Sx",1
        os += 4*Jy,"Sy",L,"Sy",1
    end

    # add last term
    os += 2*h,"Sz",L

    H = MPO(os,sites)

    return H
end

"""
```julia
TF_XY_Model_bdg(; type::String="Anisotropic", kwargs...)
```

Main dispatcher function for creating Transverse Field XY Model BdG Hamiltonians as sparse matrices.

The BdG formalism treats the TFXY model by mapping it to free fermions via Jordan-Wigner transformation.
The resulting Hamiltonian has the block structure:

```
H_BdG = [  T    D  ]
        [ -D*  -T* ]
```

where T is the kinetic + field matrix and D is the anisotropy-induced pairing matrix.

# Arguments
- `type::String="Anisotropic"`: Parameterization type
  - `"Anisotropic"`: Uses J and γ parameters  
  - `"JxJy"`: Uses separate Jx and Jy parameters 
- `kwargs...`: Additional parameters passed to specific implementations

# Returns
- `Matrix{Float64}`: 2L×2L BdG Hamiltonian matrix where L is the number of spins

# Examples
```julia
H1 = TF_XY_Model_bdg(; type="Anisotropic", L=20, J=1.0, γ=0.5, h=0.8)
H2 = TF_XY_Model_bdg(; type="JxJy", L=20, Jx=0.75, Jy=0.25, h=0.8)
```
"""
function TF_XY_Model_bdg(; type::String="Anisotropic", kwargs...)
    # add new types here if you want
    type = begin
        if type == "Anisotropic"
            Val(:Anisotropic)
        elseif type == "JxJy"
            Val(:JxJy)
        else
            error("Unknown type: $type. Use 'Anisotropic' or 'JxJy'.")
        end
    end

    return TF_XY_Model_bdg(type; kwargs...)
end

"""
    TF_XY_Model_bdg(::Val{:Anisotropic}; L=20, J=1.0, γ=0.5, h=1.0, periodic=false, parity=1)

Creates BdG Hamiltonian for TFXY model using anisotropy parameterization.

The TFXY model can be mapped to a free fermion model via Jordan-Wigner transformation:
```math
H = -J/2 \\sum_{j+1}^L \\left[ (1+γ) S_j^x S_{j+1}^x + (1-γ) S_j^y S_{j+1}^y \\right] - h \\sum_{j+1}^L S_j^z
```

The relation between (J,γ) and (Jₓ,Jᵧ) parameters is:
```math
J_x = J(1+γ)/2 \\\\
J_y = J(1-γ)/2 \\\\
J = J_x + J_y \\\\
γ = (J_x - J_y)/(J_x + J_y)
```

# Arguments
- `L::Int=20`: System size
- `J::Real=1.0`: Exchange coupling parameter
- `γ::Real=0.5`: Anisotropy parameter
- `h::Real=1.0`: Transverse field strength
- `periodic::Bool=false`: Periodic boundary conditions
- `parity::Int=1`: Parity sector for periodic boundary conditions

# Returns
- `Matrix{Float64}`: 2L×2L BdG Hamiltonian matrix
"""
function TF_XY_Model_bdg(::Val{:Anisotropic}; 
    L::Int=20, 
    J::Real=1.0, 
    γ::Real=0.5, 
    h::Real=1.0, 
    periodic::Bool=false, 
    parity::Int=1)
    
    T = spdiagm(0 => 2h .* ones(L), 1 => -J .* ones(L-1), -1 => -J .* ones(L-1))
    D = γ * spdiagm(1 => -J .* ones(L-1), -1 => J .* ones(L-1))
    
    if periodic
        T[1, end] = J * parity
        T[end, 1] = J * parity
        D[1, end] = -J*γ * parity
        D[end, 1] = J*γ * parity
    end

    H = zeros(Float64, 2*L, 2*L)

    H[1:L,1:L] = T
    H[L+1:end,L+1:end] = -transpose(T)
    H[1:L,L+1:end] = D
    H[L+1:end,1:L] = -D
        
    return H
end

"""
    TF_XY_Model_bdg(::Val{:JxJy}; L=20, Jx=1.0, Jy=1.0, h=1.0, periodic=false, parity=1)

Creates BdG Hamiltonian for TFXY model using separate x and y coupling parameters.

# Arguments
- `L::Int=20`: System size
- `Jx::Real=1.0`: Exchange coupling in x-direction
- `Jy::Real=1.0`: Exchange coupling in y-direction
- `h::Real=1.0`: Transverse field strength
- `periodic::Bool=false`: Periodic boundary conditions
- `parity::Int=1`: Parity sector for periodic boundary conditions

# Returns
- `Matrix{Float64}`: 2L×2L BdG Hamiltonian matrix
"""
function TF_XY_Model_bdg(::Val{:JxJy}; 
    L::Int=20, 
    Jx::Real=1.0, 
    Jy::Real=1.0, 
    h::Real=1.0, 
    periodic::Bool=false, 
    parity::Int=1)
    
    J_plus = (Jx + Jy)
    J_minus = (Jx - Jy)

    T = spdiagm(0 => -2h .* ones(L), 1 => J_plus .* ones(L-1), -1 => J_plus .* ones(L-1))
    D = J_minus * spdiagm(1 => ones(L-1), -1 => -1 .* ones(L-1))
    
    if periodic
        T[1, end] = -J_plus * parity
        T[end, 1] = -J_plus * parity
        D[1, end] = J_minus * parity
        D[end, 1] = -J_minus * parity
    end

    H = zeros(Float64, 2*L, 2*L)

    H[1:L,1:L] = T
    H[L+1:end,L+1:end] = -transpose(T)
    H[1:L,L+1:end] = D
    H[L+1:end,1:L] = -D
        
    return H
end
