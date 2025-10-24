# returns j+i with respecting periodic boundary conditions
function jp(j,i,N) return mod( (j+i) - 1, N) + 1 end

"""
    decoupled_lattice(L::Int) -> Dict{Tuple{Int,Symbol},Int}

    Create mapping from (i, flavor) with i = 1..L and flavor ∈ {:x,:y,:z} to
    interleaved lattice indices j = 1..3L such that
        (i,:x) => 3(i-1)+1
        (i,:y) => 3(i-1)+2
        (i,:z) => 3(i-1)+3

    Example:
        julia> decoupled_lattice(2)
        Dict(
            (1, :x) => 1,
            (1, :y) => 2,
            (1, :z) => 3,
            (2, :x) => 4,
            (2, :y) => 5,
            (2, :z) => 6
        )
"""
function decoupled_lattice(L::Int)
    @assert L > 0 "L must be positive"
    mapping = Dict{Tuple{Int,Symbol},Int}()
    @inbounds for i in 1:L
        j = 3*(i-1) + 1
        mapping[(i,:x)] = j
        mapping[(i,:y)] = j + 1
        mapping[(i,:z)] = j + 2
    end
    return sort(mapping)
end

"""
    BLBQ_Hamiltonian_MPO(sites; type::String="Spin", kwargs...)

Main dispatcher function for creating BLBQ (Bilinear-Biquadratic) Hamiltonian MPOs.

# Arguments
- `sites::Vector{<:Index}`: ITensor sites object defining the system
- `type::String`: Type of representation ("Spin", "Fermion", "MF1", "MF2")
- `kwargs...`: Additional keyword arguments passed to specific implementations

# Returns
- MPO representation of the BLBQ Hamiltonian

# Examples
```julia
sites = siteinds("S=1", 10)
H = BLBQ_Hamiltonian_MPO(sites; type="Spin", J=1.0, K=0.5)

sites = siteinds("Fermion", 12)
H = BLBQ_Hamiltonian_MPO(sites; type="MF1", J=1.0, K=0.0, λ=1.0)
```
"""
function BLBQ_Hamiltonian_MPO(sites::Vector{<:Index}; 
    type::String="Spin", 
    kwargs...)

    # add new types here if you want
    type = begin
        if type == "Spin"
            Val(:Spin)
        elseif type == "MF1"
            Val(:MF1)
        elseif type == "MF1_rescaled"
            Val(:MF1_rescaled)
        elseif type == "MF2"
            Val(:MF2)
        elseif type == "MF2_rescaled"
            Val(:MF2_rescaled)
        elseif type == "MF2_ULS"
            Val(:MF2_ULS)
        else
            error("Unknown type: $type")
        end
    end

    return BLBQ_Hamiltonian_MPO(type,sites; kwargs...)
end

"""
    BLBQ_Hamiltonian_MPO(::Val{:Spin}, sites; J=1, K=1, splitted=false, periodic=false, periodic_mode=Val(:default))

Creates BLBQ Hamiltonian MPO for spin-1 systems using spin operators.

The Hamiltonian is: H = Σᵢ [J Sᵢ·Sᵢ₊₁ + K (Sᵢ·Sᵢ₊₁)²]

# Arguments
- `sites::Vector{<:Index}`: ITensor sites for spin-1 system
- `J::Real=1`: Bilinear coupling strength
- `K::Real=1`: Biquadratic coupling strength  
- `periodic::Bool=false`: Enable periodic boundary conditions
- `periodic_mode::Union{Val{:optimized},Val{:default}}=Val(:default)`: Optimization mode for periodic boundaries

# Returns
- If `splitted=false`: Single MPO containing full Hamiltonian
- If `splitted=true`: Tuple of (bilinear_MPO, biquadratic_MPO)

"""
function BLBQ_Hamiltonian_MPO(::Val{:Spin}, sites::Vector{<:Index};
    J::Real=1, 
    K::Real=1, 
    periodic::Bool=false, 
    periodic_mode::Union{Val{:optimized},Val{:default}}=Val(:default))

    @assert hastags(sites[1], "S=1") "The sites must be spin-1 sites for the BLBQ model."

    L = length(sites)
    L_bounds = periodic ? L : L - 1
    site_ordering = periodic ? get_pbc_order_1D(L; mode=periodic_mode) : collect(1:L)
    
    # Bilinear term
    os1 = OpSum()
    # Biquadratic term
    os2 = OpSum()
    for i in 1:L_bounds
        j = site_ordering[i]
        jp1 = site_ordering[jp(i,1,L)]

        # Bilinear term
        os1 += 1/2 * J,"S+",j,"S-",jp1
        os1 += 1/2 * J,"S-",j,"S+",jp1
        os1 += J,"Sz",j,"Sz",jp1

        # Biquadratic term
        os2 += 1/4 * K,"S+",j,"S-",jp1,"S+",j,"S-",jp1
        os2 += 1/4 * K,"S+",j,"S-",jp1,"S-",j,"S+",jp1
        os2 += 1/4 * K,"S-",j,"S+",jp1,"S+",j,"S-",jp1
        os2 += 1/4 * K,"S-",j,"S+",jp1,"S-",j,"S+",jp1

        os2 += 1/2 * K,"S+",j,"S-",jp1,"Sz",j,"Sz",jp1
        os2 += 1/2 * K,"S-",j,"S+",jp1,"Sz",j,"Sz",jp1

        os2 += 1/2 * K,"Sz",j,"Sz",jp1,"S+",j,"S-",jp1
        os2 += 1/2 * K,"Sz",j,"Sz",jp1,"S-",j,"S+",jp1

        os2 += K,"Sz",j,"Sz",jp1,"Sz",j,"Sz",jp1
    end

    os = os1 + os2
    
    return MPO(os, sites)
end;

"""
    BLBQ_Hamiltonian_MPO(::Val{:MF1}, sites; J=1, K=1, λ=1, Δ=1, χ=1, splitted=false, periodic=true, parity=1, coordinates=Val(:cartesian))

Mean-field approximation of BLBQ model with Lagrange multiplier constraint.

Conserves the particle number on average using a Lagrange multiplier λ.
In cartesian coordinates, decouples into three independent Kitaev chains.

# Arguments
- `sites::Vector{<:Index}`: Fermion sites (length multiple of 3)
- `J::Real=1`: Bilinear coupling
- `K::Real=1`: Biquadratic coupling  
- `λ::Real=1`: Lagrange multiplier for particle number constraint
- `Δ::Real=1`: Pairing amplitude
- `χ::Real=1`: Hopping amplitude renormalization
- `splitted::Bool=false`: Return separate MPOs for each flavor (cartesian only - does not make sense for other coordinates)
- `periodic::Bool=true`: Periodic boundaries
- `parity::Int=1`: APBC parity factor (±1)
- `coordinates::Union{Val{:cartesian},Val{:default}}=Val(:cartesian)`: Coordinate representation

# Returns
- If `splitted=false`: Single MPO
- If `splitted=true` (cartesian): Tuple of (x_flavor_MPO, y_flavor_MPO, z_flavor_MPO)

"""
function BLBQ_Hamiltonian_MPO(::Val{:MF1}, sites::Vector{<:Index}; 
    J::Real=1, 
    K::Real=1,
    λ::Real=1, 
    Δ::Real=1, 
    χ::Real=1,
    splitted::Bool=false, 
    periodic::Bool=true, 
    parity::Int = 1, #APBC
    coordinates::Union{Val{:cartesian},Val{:default}}=Val(:cartesian))

    @assert hastags(sites[1], "Fermion") "The sites must be Fermion sites for the BLBQ model with Abrikosov fermions."
    @assert mod(length(sites),3) == 0 "length(sites) must be a multiple of 3 for the BLBQ model with Abrikosov fermions."

    N = length(sites)
    N_bounds = periodic ? N : N-3
    site_ordering = collect(1:N)

    # here we can split the Hamiltonian into 3 parts (3 decoupled Kitaev chains + Interaction term)
    if coordinates==Val(:cartesian)
        # x flavor
        os1 = OpSum()
        for i in 1:3:N_bounds
            j = site_ordering[i]
            jp3 = site_ordering[jp(i,3,N)]

            # respect boundary conditions with given parity
            p3 = 1<=jp(j,3,N)<=3 ? -parity : parity

            # fermion hopping part
            os1 += -p3*J*χ,"Cdag",j,"C",jp3
            os1 += -p3*J*χ,"Cdag",jp3,"C",j

            # singlet pairing part
            os1 += p3*Δ*(J-K),"C",j,"C",jp3
            os1 += p3*Δ*(J-K),"Cdag",jp3,"Cdag",j

            # interaction term
            os1 += λ,"N",j
        end

        # y flavor
        os2 = OpSum()
        for i in 2:3:N_bounds
            j = site_ordering[i]
            jp3 = site_ordering[jp(i,3,N)]

            # respect boundary conditions with given parity
            p3 = 1<=jp(i,3,N)<=3 ? -parity : parity

            # fermion hopping part
            os2 += -p3*J*χ,"Cdag",j,"C",jp3
            os2 += -p3*J*χ,"Cdag",jp3,"C",j

            # singlet pairing part
            os2 += p3*Δ*(J-K),"C",j,"C",jp3
            os2 += p3*Δ*(J-K),"Cdag",jp3,"Cdag",j

            # interaction term
            os2 += λ,"N",j
        end
        # z flavor
        os3 = OpSum()
        for i in 3:3:N_bounds
            j = site_ordering[i]
            jp3 = site_ordering[jp(i,3,N)]

            # respect boundary conditions with given parity
            p3 = 1<=jp(i,3,N)<=3 ? -parity : parity

            # fermion hopping part
            os3 += -p3*J*χ,"Cdag",j,"C",jp3
            os3 += -p3*J*χ,"Cdag",jp3,"C",j

            # singlet pairing part
            os3 += p3*Δ*(J-K),"C",j,"C",jp3
            os3 += p3*Δ*(J-K),"Cdag",jp3,"Cdag",j

            # interaction term
            os3 += λ,"N",j
        end

        # interaction term - lagrange multiplier
        for i in N_bounds:N
            os1 += λ,"N",site_ordering[i]
            os2 += λ,"N",site_ordering[i]
            os3 += λ,"N",site_ordering[i]
        end

        if splitted
            return MPO(os1, sites), MPO(os2, sites), MPO(os3, sites)
        else
            os = os1 + os2 + os3
            
            return MPO(os, sites)
        end
    # here we do not split the Hamiltonian into parts
    elseif coordinates==Val(:default)
        os = OpSum()
        for i in 1:N_bounds
            j = site_ordering[i]
            jp3 = site_ordering[jp(i,3,N)]

            # respect boundary conditions with given parity
            p3 = 1<=jp(i,3,N)<=3 ? -parity : parity

            # fermion hopping part
            os += p3*(-χ * J),"Cdag",j,"C",jp3
            os += p3*(-χ * J),"Cdag",jp3,"C",j

            # interaction term - lagrange multiplier
            os += λ,"N",j
        end
        for j in N_bounds:N
            os += λ,"N",j
        end

        # singlet pairing term
        for i in 1:3:N_bounds
            j = site_ordering[i]
            jp1 = site_ordering[jp(i,1,N)]
            jp2 = site_ordering[jp(i,2,N)]
            jp3 = site_ordering[jp(i,3,N)]
            jp4 = site_ordering[jp(i,4,N)]
            jp5 = site_ordering[jp(i,5,N)]

            # respect boundary conditions with given parity
            p1 = jp(j,1,N)==1 ? -parity : parity
            p2 = 1<=jp(i,2,N)<=2 ? -parity : parity
            p3 = 1<=jp(i,3,N)<=3 ? -parity : parity
            p4 = 1<=jp(i,4,N)<=4 ? -parity : parity
            p5 = 1<=jp(i,5,N)<=5 ? -parity : parity

            os += -p5*Δ*(J-K),"Cdag",jp5,"Cdag",j
            os += -p5*Δ*(J-K),"C",j,"C",jp5

            os += p4*p1*Δ*(J-K),"Cdag",jp4,"Cdag",jp1
            os += p4*p1*Δ*(J-K),"C",jp1,"C",jp4

            os += -p3*p2*Δ*(J-K),"Cdag",jp3,"Cdag",jp2
            os += -p3*p2*Δ*(J-K),"C",jp2,"C",jp3
        end

        return MPO(os, sites)
    end
end

"""
    BLBQ_Hamiltonian_MPO(::Val{:MF2}, sites; J=1, K=0, U=1, Δ=1, χ=1, periodic=true, parity=1)

Alternative mean-field BLBQ with interaction term U(nᵢ-1)².
Can be used instead of a Gutzwiller projection for U → ∞.
Only available in cartesian coordinates. Default parameters gives rescaled version.

# Arguments
- `sites::Vector{<:Index}`: Fermion sites (length multiple of 3)
- `J::Real=1`: Bilinear coupling
- `K::Real=0`: Biquadratic coupling (default 0 for rescaled version)
- `U::Real=1`: Interaction strength for constraint enforcement
- `Δ::Real=1`: Pairing amplitude
- `χ::Real=1`: Hopping renormalization
- `periodic::Bool=true`: Periodic boundaries
- `parity::Int=1`: APBC parity

# Returns
- MPO with interaction-based constraint

# Notes
- U >> other energy scales, enforces single occupancy constraint
"""
function BLBQ_Hamiltonian_MPO(::Val{:MF2}, sites::Vector{<:Index}; 
    J::Real=1,
    K::Real=0,
    U::Real=1,
    Δ::Real=1,
    χ::Real=1, 
    λ::Real=1,
    periodic::Bool=true, 
    parity::Int = 1, #APBC
    )

    @assert hastags(sites[1], "Fermion") "The sites must be Fermion sites for the BLBQ model with Abrikosov fermions."
    @assert mod(length(sites),3) == 0 "length(sites) must be a multiple of 3 for the BLBQ model with Abrikosov fermions."

    N = length(sites)
    N_bounds = periodic ? N : N-3
    # site_ordering = periodic ? get_pbc_order_1D(N; mode=periodic_mode) : collect(1:N)
    site_ordering = collect(1:N)

    os = OpSum()
    for i in 1:N_bounds
        j = site_ordering[i]
        jp3 = site_ordering[jp(i,3,N)]

        # respect boundary conditions with given parity
        p3 = 1<=jp(i,3,N)<=3 ? -parity : parity

        # fermion hopping part
        os += -p3*J*χ,"Cdag",j,"C",jp3
        os += -p3*J*χ,"Cdag",jp3,"C",j

        # singlet pairing part
        os += p3*Δ*(J-K),"C",j,"C",jp3
        os += p3*Δ*(J-K),"Cdag",jp3,"Cdag",j

        # chem. potential
        os += λ,"N",j
    end
    for j in N_bounds:N
        os += λ,"N",j
    end

    # interaction term 
    for i in 1:3:N_bounds
        j = site_ordering[i]
        jp1 = site_ordering[jp(i,1,N)]
        jp2 = site_ordering[jp(i,2,N)]

        os += U,"N",j,"N",j
        os += U,"N",j,"N",jp1
        os += U,"N",j,"N",jp2
        os += -2U,"N",j

        os += U,"N",jp1,"N",j
        os += U,"N",jp1,"N",jp1
        os += U,"N",jp1,"N",jp2
        os += -2U,"N",jp1

        os += U,"N",jp2,"N",j
        os += U,"N",jp2,"N",jp1
        os += U,"N",jp2,"N",jp2
        os += -2U,"N",jp2

        os += U/3,"Id",j
        os += U/3,"Id",jp1
        os += U/3,"Id",jp2
    end
    return MPO(os, sites)
end

"""
    BLBQ_Hamiltonian_MPO(::Val{:MF2}, sites; J=1, K=0, U=1, Δ=1, χ=1, periodic=true, parity=1, periodic_mode=Val(:default))

Alternative mean-field BLBQ with interaction term U(nᵢ-1)².
Only available in cartesian coordinates. Default parameters give rescaled version.
Does not have the pairing term so that quantum numbers can be used.

# Arguments
- `sites::Vector{<:Index}`: Fermion sites (length multiple of 3)
- `J::Real=1`: Bilinear coupling
- `K::Real=0`: Biquadratic coupling (default 0 for rescaled version)
- `U::Real=1`: Interaction strength for constraint enforcement
- `Δ::Real=1`: Pairing amplitude
- `χ::Real=1`: Hopping renormalization
- `periodic::Bool=true`: Periodic boundaries
- `parity::Int=1`: APBC parity
- `periodic_mode`: Boundary optimization mode

# Returns
- MPO with interaction-based constraint

# Notes
- U >> other energy scales, enforces single occupancy constraint
"""
function BLBQ_Hamiltonian_MPO(::Val{:MF2_ULS}, sites::Vector{<:Index}; 
    U::Real=1,
    periodic::Bool=true, 
    parity::Int = 1, #APBC
    )

    @assert hastags(sites[1], "Fermion") "The sites must be Fermion sites for the BLBQ model with Abrikosov fermions."
    @assert mod(length(sites),3) == 0 "length(sites) must be a multiple of 3 for the BLBQ model with Abrikosov fermions."

    N = length(sites)
    @assert mod(N,3) == 0 "N must be a multiple of 3"
    L = div(N,3)
    L_bounds = periodic ? L : L-1
    lattice_mapping = decoupled_lattice(L)

    os = OpSum()
    for i in 1:L_bounds, α in [:x,:y,:z]
        # respect boundary conditions with given parity
        p = jp(i,1,L)==1 ? -parity : parity

        # fermion hopping part
        os += -p,"Cdag",lattice_mapping[i,α],"C",lattice_mapping[jp(i,1,L),α]
        os += -p,"Cdag",lattice_mapping[jp(i,1,L),α],"C",lattice_mapping[i,α]
    end

    # interaction term 
    for i in 1:L_bounds
        os += U,"N",lattice_mapping[i,:x],"N",lattice_mapping[i,:y]
        os += U,"N",lattice_mapping[i,:x],"N",lattice_mapping[i,:z]
        os += -U,"N",lattice_mapping[i,:x]

        os += U,"N",lattice_mapping[i,:y],"N",lattice_mapping[i,:x]
        os += U,"N",lattice_mapping[i,:y],"N",lattice_mapping[i,:z]
        os += -U,"N",lattice_mapping[i,:y]

        os += U,"N",lattice_mapping[i,:z],"N",lattice_mapping[i,:x]
        os += U,"N",lattice_mapping[i,:z],"N",lattice_mapping[i,:y]
        os += -U,"N",lattice_mapping[i,:z]

        os += U/3,"Id",lattice_mapping[i,:x]
        os += U/3,"Id",lattice_mapping[i,:y]
        os += U/3,"Id",lattice_mapping[i,:z]
    end
    return MPO(os, sites)
end

#= 
    BdG Hamiltonians
=#

"""
    BLBQ_Hamiltonian_bdg(; type::String="Cartesian", kwargs...)

Main dispatcher function for creating BLBQ Bogoliubov-de Gennes (BdG) Hamiltonians as sparse matrices.

The BdG formalism treats superconducting systems by doubling the Hilbert space to include
both particle and hole degrees of freedom. The resulting Hamiltonian has the block structure:

```
H_BdG = [  T    D  ]
        [ -D*  -T* ]
```

where T is the kinetic + chemical potential matrix and D is the pairing matrix.

# Arguments
- `type::String="Cartesian"`: Coordinate representation
  - `"Cartesian"`: Uses Cartesian (x,y,z) fermion flavors  
  - `"Abrikosov"`: Uses Abrikosov fermion representation 
- `kwargs...`: Additional parameters passed to specific implementations

# Returns
- `Matrix{Float64}`: 2N×2N BdG Hamiltonian matrix where N is the number of fermion sites

"""
function BLBQ_Hamiltonian_bdg(; type::String="Cartesian", kwargs...)
    # add new types here if you want
    type = begin
        if type == "Cartesian"
            Val(:Cartesian)
        elseif type == "Abrikosov"
            Val(:Abrikosov)
        else
            error("Unknown type: $type. Use 'Cartesian', 'Abrikosov'.")
        end
    end

    return BLBQ_Hamiltonian_bdg(type; kwargs...)
end

"""
    BLBQ_Hamiltonian_bdg(::Val{:Cartesian}; λ=1, Δ=0, χ=1, J=1, K=1, N=6, periodic=false, parity=1)

Creates BdG Hamiltonian for BLBQ model in Cartesian coordinates.

In Cartesian coordinates, the three fermion flavors (x,y,z) are decoupled and each follows
an independent Kitaev chain. The BdG Hamiltonian describes the mean-field
superconducting state with nearest-neighbor hopping and pairing.

# Arguments
- `λ::Real=1`: Chemical potential (on-site energy)
- `Δ::Real=0`: Superconducting pairing amplitude
- `χ::Real=1`: Hopping amplitude renormalization factor
- `J::Real=1`: Bilinear coupling strength from original spin model
- `K::Real=1`: Biquadratic coupling strength from original spin model
- `N::Int=6`: Number of fermion sites (must be multiple of 3)
- `periodic::Bool=false`: Enable periodic boundary conditions
- `parity::Int=1`: Antiperiodic boundary condition parity (±1 for fermions)

# Returns
- `Matrix{Float64}`: 2N×2N BdG Hamiltonian matrix

"""
function BLBQ_Hamiltonian_bdg(::Val{:Cartesian}; 
    λ::Real=1, 
    Δ::Real=0, 
    χ::Real=1, 
    J::Real=1, 
    K::Real=1, 
    N::Int=6, 
    periodic::Bool = false, 
    parity::Int = 1)

    T = spdiagm(0 => λ .* ones(N), 3 => -χ*J .* ones(N-3), -3 => -χ*J .* ones(N-3))
    D = spdiagm(3 => -(J-K)*Δ .* ones(N-3), -3 => (J-K)*Δ .* ones(N-3))

    if periodic
        T[1, N-2] = χ*J * parity
        T[2, N-1] = χ*J * parity
        T[3, N] = χ*J * parity
        T[N-2, 1] = χ*J * parity
        T[N-1, 2] = χ*J * parity
        T[N, 3] = χ*J * parity

        D[1, N-2] = -(J-K)*Δ * parity
        D[2, N-1] = -(J-K)*Δ* parity
        D[3, N] = -(J-K)*Δ * parity
        D[N-2, 1] = (J-K)*Δ * parity
        D[N-1, 2] = (J-K)*Δ * parity
        D[N, 3] = (J-K)*Δ * parity
    end

	H = zeros(Float64,2*N, 2*N)
	
	H[1:N,1:N] = T
	H[N+1:end,N+1:end] = -transpose(T)
	H[1:N,N+1:end] = D
	H[N+1:end,1:N] = -D
    
	return H
end

"""
    BLBQ_Hamiltonian_bdg(::Val{:Abrikosov}; λ=1, Δ=0, χ=1, J=1, K=1, N=6, periodic=false, parity=1)

Creates BdG Hamiltonian for BLBQ model using Abrikosov fermion representation.

# Arguments
- `λ::Real=1`: Chemical potential (Lagrange multiplier for constraint)
- `Δ::Real=0`: Superconducting pairing amplitude
- `χ::Real=1`: Hopping amplitude renormalization
- `J::Real=1`: Bilinear coupling from spin model
- `K::Real=1`: Biquadratic coupling from spin model  
- `N::Int=6`: Number of fermion sites (must be multiple of 3)
- `periodic::Bool=false`: Periodic boundary conditions
- `parity::Int=1`: Fermion antiperiodic boundary parity

# Returns
- `Matrix{Float64}`: 2N×2N BdG Hamiltonian
"""
function BLBQ_Hamiltonian_bdg(::Val{:Abrikosov}; 
    λ::Real=1, 
    Δ::Real=0, 
    χ::Real=1, 
    J::Real=1, 
    K::Real=1, 
    N::Int=6, 
    periodic::Bool = false, 
    parity::Int = 1)

    T = spdiagm(0 => λ .* ones(N), 3 => -χ*J .* ones(N-3), -3 => -χ*J .* ones(N-3))

    #= 
        creates: 
        col_indices = [4,5,6,...,N]
        row_indices = [3,2,1,6,5,4,...,N-3,N-4,N-5]
    =#
    col_indices = 4:N
    row_indices = []
    for i in 1:(div(N,3)-1)
        push!(row_indices,(i-1)*3 + 3)
        push!(row_indices,(i-1)*3 + 2)
        push!(row_indices,(i-1)*3 + 1)
    end

    D_pattern = Δ*(J-K) .* repeat([1, -1,1], Int((N-3)/3))
    D = sparse(row_indices, col_indices, D_pattern, N, N) + sparse(col_indices, row_indices, -D_pattern, N, N)

    if periodic
        T[1, N-2] = χ*J * parity
        T[2, N-1] = χ*J * parity
        T[3, N] = χ*J * parity
        T[N-2, 1] = χ*J * parity
        T[N-1, 2] = χ*J * parity
        T[N, 3] = χ*J * parity

        D[1,N] = Δ*(J-K) * parity
        D[2,N-1] = -Δ*(J-K) * parity
        D[3,N-2] = Δ*(J-K) * parity
        D[N,1] = -Δ*(J-K) * parity
        D[N-1,2] = Δ*(J-K) * parity
        D[N-2,3] = -Δ*(J-K) * parity
    end

	H = zeros(Float64,2*N, 2*N)
	
	H[1:N,1:N] = T
	H[N+1:end,N+1:end] = -transpose(T)
	H[1:N,N+1:end] = D
	H[N+1:end,1:N] = -D
    
	return H
end

#= Operators =#

# Simplification of exp(iπ*S^z)
"""
    ITensors.op(::OpName"S_tilde", ::SiteType"S=1")

Matrix representation of the operator exp(iπ Sᶻ) for spin-1 sites.

# Returns
- `Matrix{Int}`: Diagonal matrix diag(-1, 1, -1) acting on a spin-1 site.
"""
ITensors.op(::OpName"S_tilde",::SiteType"S=1") =
    [-1 0 0
    0 1 0
    0 0 -1]

"""
    string_correlator_MPO(i::Int, j::Int, sites::Vector{<:Index}; type="Spin")

Build the non-local string correlator MPO ⟨Sᶻᵢ exp(iπ ∑ Sᶻ) Sᶻⱼ⟩ between sites `i` and `j`.

# Arguments
- `i`, `j`: Site indices with `i < j`.
- `sites`: Collection of ITensor site indices.
- `type`: Either "Spin" or "Fermion" selecting the implementation.

# Returns
- `MPO`: String correlator in the requested representation.
"""
function string_correlator_MPO(i::Int,j::Int,sites::Vector{<:Index}; type="Spin")
    if type == "Spin"
        return string_correlator_MPO(Val(:Spin), i, j, sites)
    elseif type == "Fermion"
        return string_correlator_MPO(Val(:Fermion), i, j, sites)
    end
end;

"""
    string_correlator_MPO(::Val{:Spin}, i::Int, j::Int, sites::Vector{<:Index})

Construct the spin-1 string correlator MPO using Sᶻ and the diagonal S̃ operator.

# Returns
- `MPO`: String correlator acting on spin-1 sites.
"""
function string_correlator_MPO(::Val{:Spin}, i::Int,j::Int,sites::Vector{<:Index})
    @assert hastags(sites[1], "S=1") "String correlator MPO only defined on S=1 sites"

    os = OpSum()
    os += "Sz",i

    for k in i+1:j-1
        os *= "S_tilde",k
    end

    os *= "Sz",j

    return MPO(os,sites)
end;

"""
    string_correlator_MPO(::Val{:Fermion}, i::Int, j::Int, sites::Vector{<:Index})

Construct the fermionic string correlator MPO in the cartesian Abrikosov representation.

# Returns
- `MPO`: String correlator acting on three-flavor fermionic sites.
"""
function string_correlator_MPO(::Val{:Fermion}, i::Int,j::Int,sites::Vector{<:Index})
    @assert hastags(sites[1], "Fermion") "String correlator MPO only defined on Fermion sites"
    N = length(sites)
    @assert mod(N,3) == 0 "N must be a multiple of 3"
    L = Int(N/3)

    lattice_mapping = decoupled_lattice(N)

    os = OpSum()
    os += 1im,"Cdag",lattice_mapping[i,:x],"C",lattice_mapping[i,:y]
    os += -1im,"Cdag",lattice_mapping[i,:y],"C",lattice_mapping[i,:x]

    for k in i+1:j-1
        os_temp = OpSum()
        os_temp += "Cdag",lattice_mapping[k,:z],"C",lattice_mapping[k,:z]
        os_temp += -1,"Cdag",lattice_mapping[k,:x],"C",lattice_mapping[k,:x]
        os_temp += -1,"Cdag",lattice_mapping[k,:y],"C",lattice_mapping[k,:y]

        os *= os_temp
    end

    os_Sz_j = OpSum()
    os_Sz_j += 1im,"Cdag",lattice_mapping[j,:x],"C",lattice_mapping[j,:y]
    os_Sz_j += -1im,"Cdag",lattice_mapping[j,:y],"C",lattice_mapping[j,:x]

    os *= os_Sz_j

    return MPO(Ops.expand(os),sites)
end;

"""
    dimer_phase_order_param(i::Int, sites::Vector{<:Index}; coordinate::String="z")

Create MPOs for the dimer order parameter on bonds (2i-1, 2i) and (2i, 2i+1).

# Returns
- `(MPO, MPO)`: Pair of MPOs whose expectation values form the dimer difference.
"""
function dimer_phase_order_param(i::Int,sites::Vector{<:Index}; coordinate::String="z")
    @assert hastags(sites[1], "S=1") "String correlator MPO only defined on S=1 sites"
    @assert coordinate in ["x", "y", "z"] "Coordinate must be one of 'x', 'y', or 'z'"

    L = length(sites)

    i1 = mod( (2i-1) - 1, L) + 1
    i2 = mod( (2i) - 1, L) + 1
    i3 = mod( (2i+1) - 1, L) + 1

    os1 = OpSum()
    os2 = OpSum()

    os1 += "S$coordinate",i1,"S$coordinate",i2
    os2 += "S$coordinate",i2,"S$coordinate",i3

    return MPO(os1,sites), MPO(os2,sites)
end

"""
    local_density_density_corr_fct(psi, sites, i::Int, j::Int)

Evaluate ⟨(nᵢₓ + nᵢᵧ + nᵢ_z)(nⱼₓ + nⱼᵧ + nⱼ_z)⟩ and subtract the disconnected contribution.

# Arguments
- `psi`: MPS state defined on fermionic sites.
- `sites`: Site indices compatible with `psi`.
- `i`, `j`: Unit-cell indices for the density operators.

# Returns
- `Number`: Connected density-density correlator between cells `i` and `j`.
"""
function local_density_density_corr_fct(psi, sites, i, j)
    @assert hastags(psi[1], "Fermion") "Operator only defined on Fermion sites"
    N = length(psi)
    @assert mod(N,3) == 0 "N must be a multiple of 3"
    L = Int(N/3)

    lattice_mapping = decoupled_lattice(N)

    os1 = OpSum()
    os1 += "N",lattice_mapping[i,:x]
    os1 += "N",lattice_mapping[i,:y]
    os1 += "N",lattice_mapping[i,:z]

    os2 = OpSum()
    os2 += "N",lattice_mapping[j,:x]
    os2 += "N",lattice_mapping[j,:y]
    os2 += "N",lattice_mapping[j,:z]

    D_ij_1 = MPO(Ops.expand(os1*os2), sites)
    D_ij_2 = MPO(os1, sites)
    D_ij_3 = MPO(os2, sites)

    return inner(psi',D_ij_1,psi) - inner(psi',D_ij_2,psi)*inner(psi',D_ij_3,psi)
end

#= 
    Coordinate transformations
=#

"""
```julia
cartesianToSpin1(A::MPS)
```

Maps the MPS A defined in cartesian coordinates [x,y,z] to the Abrikosov Fermion representation of spin-1.

It is assumed that the MPS A is already Gutzwiller projected and the sites are spin-1 sites in the cartesian basis.

# Abrikosov Fermion representation of spin-1:
    •|1> = |100>
    •|0> = |010>
    •|-1> = |001>

# Mapping:
    •|x> -> 1/√2 (|-1> - |1>) = 1/√2 (|001> - |100>)
    •|y> -> i/√2 (|-1> + |1>) = i/√2 (|001> + |100>)
    •|z> -> |0> = |010>

# Returns
    • B::MPS in the basis of Abrikosov fermions [1,0,-1] defined over sites with spin-1

# Keyword arguments
    • A::MPS
...
"""
function cartesianToSpin1(A::MPS)
    L = length(A)

    B_array= map(1:L) do j
        B = A[j]

        siteIndices = [i for i in inds(B) if hastags(i, "Site")]
        linkIndices = [i for i in inds(B) if hastags(i, "Link")]

        B_mapped = ITensor(ComplexF64,(siteIndices,linkIndices)) # creates empty tensor with same structure as B

        # creates all pairs for the links
        allLinkPointer = map(1:length(linkIndices)) do i
            return map(1:dim(linkIndices[i])) do k
                return linkIndices[i] => k
            end
        end
        allLinksComb = collect(IterTools.product(allLinkPointer...)) # all permutations of linkPointer

        for s in 1:3
            for linkPointer in allLinksComb
                coef = B[siteIndices[1]=>s, linkPointer...]  

                pointerB_Plus = [siteIndices[1]=>1, linkPointer...]
                pointerB_Null = [siteIndices[1]=>2, linkPointer...]
                pointerB_Minus = [siteIndices[1]=>3, linkPointer...]

                if s==1 # tau = x 
                    B_mapped[pointerB_Minus...] += coef/sqrt(2) #  |001>
                    B_mapped[pointerB_Plus...] += -coef/sqrt(2) # -|100>
                    # B_mapped[pointerB_Plus...] += 1im*coef/sqrt(2) #  |001>
                    # B_mapped[pointerB_Minus...] += -1im*coef/sqrt(2) # -|100>
                end
                if s==2 # tau = y 
                    B_mapped[pointerB_Minus...] += 1im*coef/sqrt(2) # |001>
                    B_mapped[pointerB_Plus...] += 1im*coef/sqrt(2) # |100>
                    # B_mapped[pointerB_Plus...] += coef/sqrt(2) # |001>
                    # B_mapped[pointerB_Minus...] += coef/sqrt(2) # |100>
                end
                if s==3 # tau = z 
                    B_mapped[pointerB_Null...] += coef #  |010>
                    # B_mapped[pointerB_Null...] += -1im*coef #  |010>
                end
            end
        end
        return B_mapped
    end

    return MPS(B_array) 
end