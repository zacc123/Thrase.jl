using Thrase
using LinearAlgebra # Sparse Linear Algebra Operations
using SparseArrays
using Krylov
using DifferentialEquations, DiffEqCallbacks # ODE Solver
using NaNMath # File I/O and Formatting
using DelimitedFiles
using Printf
using Dates
using Plots

using CUDA # CUDA Support for CSR Format + CG
using CUDA.CUSPARSE
using CUDSS

using JLD2 # Saving JSON like Files with SBP Operators

include("./ops_bp5.jl")    # Metrics, SBP operators, and Rate and State functions
include("./odefun_bp5.jl") # RHS of ODE, looks a lot like driver, but marches forward in time
include("../utils_3D.jl")  # File Writing Functions for BP5 output  

const localARGS = ["../../examples/bp5-qd.dat"]
const device_num::Int64 = 0

function main()

    device!(device_num)

    # Read in params from DAT file for problem
    (pth, stride_space, stride_time, SBPp,
     xc, yc, zc,
     Hx, Hy, Hz, 
     Nx, Ny, Nz, 
     shift_flag, gpu_flag,
     ρ, cs, ν, 
     RSamin, RSamax, RSb,
     σn, RSDc, Vp, RSVinit,
     RSV0, RSf0, RShs,
     RSht, RSH, RSl, RSWf,
     RSlf, RSw, Δz,
     sim_years) = read_params_BP5(localARGS[1])

    # Setup the ouptut dir, try / catch to cover lazy users like me lol
    pth = pth * "$(Dates.format(now(), "yyyy_mm_dd_HHMM"))/"
    try
      mkdir(pth)
    catch
        # folder already exists and data will be overwritten.
        print("Directory: $(pth) already exists \nDo you want to overwrite the existing data? (Y or N)\n")
        # response = readline()
        response = ['y']
        if (response[1] == 'Y' || response[1] == 'y')
            print("Continuing...\n")
        
        else
            print("aborting...\n")
            return nothing
        end
    end

    # parameter house keeping and setting up the problem domain
    year_seconds = 31556926

    # All physics parameters coming from stress and strain relations
    μ = cs^2 * ρ 
    μshear = cs^2 * ρ
    η = μshear / (2 * cs)
    λ = 2*μ*ν / (1 - 2*ν)
    Κ = 2*μ*(ν + 1) / (3* (1 - 2*ν))

    ################################## COORDINATE TRANSFORM ###################################
    # Physical Domain: (x, y, z) in (Lx1, Lx2) x (Ly1, Ly2) x (Lz1, Lz2)

    dx = (xc[2] - xc[1]) / Nx
    dy = (yc[2] - yc[1]) / Ny
    dz = (zc[2] - zc[1]) / Nz

    x = xc[1]:dx:xc[2]
    y = yc[1]:dy:yc[2]
    z = zc[1]:dz:zc[2]

    if dz < 2*Hz/Nz # TODO Ask Brittany about these checks in 3D
      #for bp1-qd, need dz to be greater than 2Hz/Ns or will get errors
      print("need more grid points or increase dz\n")
      return
    end

    # Move to Logical Names for the rest of simulation
    Nq = Nx
    Nr = Ny
    Ns = Nz

    Nqp = Nq + 1
    Nrp = Nr + 1 
    Nsp = Ns + 1

    Np = Nqp * Nrp * Nsp # total size of 1 comp of operator (i.e xx part)

    # Get stretch factors to move between 
    α_x = (xc[2] - xc[1]) / 2
    α_y = (yc[2] - yc[1]) / 2
    α_z = (zc[2] - zc[1]) / 2

    β_x = (xc[2] + xc[1]) / 2
    β_y = (yc[2] + yc[1]) / 2
    β_z = (zc[2] + zc[1]) / 2

    # Affine Transform to "squeeze" to logic domain
    xt=(q,r,s) -> ((q .* α_x) .+ β_x, ones(size(q)) .* α_x, zeros(size(r)),       zeros(size(s)))
    yt=(q,r,s) -> ((r .* α_y) .+ β_y, zeros(size(q)),       ones(size(r)) .* α_y, zeros(size(s)))
    zt=(q,r,s) -> ((s .* α_z) .+ β_z, zeros(size(q)),       zeros(size(r)),       ones(size(s)) .* α_z)

    # Can do variable coefs, but constant for BP5
    λ_f(x, y, z, B_p) = λ
    μ_f(x, y, z, B_p) = μ 
    K = Κ 
    B_p = 1
    
    # Grab metrics / ops load or regen
    print("Load Metrics from Cache? (Y | N)\n");
    ans = readline();
    metrics_load = (ans[1] == 'y' || ans[1] == "Y");

    print("Load SBP OP from Cache? (Y | N)\n");
    ans = readline();
    op_load = (ans[1] == 'y' || ans[1] == "Y");

    ################################## CREATE METRICS and TRANSFORM PROBLEM TO LOGICAL SPACE ###################################
    print("\nCreating metrics....\n")
    if metrics_load
        print("\nLoading Metrics from ../../../../offline_stashed_ops/metrics_cache.jld2 ....");
        @load "../../../../offline_stashed_ops/metrics_cache.jld2" metrics;
        print("Done\n");
    else
        print("\nCreating Metrics:");
        @time metrics = create_metrics_bp5(SBPp, Nq, Nr, Ns, λ_f, μ_f, K, B_p; xf=xt, yf=yt, zf=zt)
        GC.gc()
        print("\nSaving Metrics to Disk...");
        @save "../../../../offline_stashed_ops/metrics_cache.jld2" metrics
        print("Done\n");
    end

    ################################## CREATE SBP-SAT OPERATORS ###################################
    # create finite difference operators on computational domain:
    # Notation: 
        # M == D2 + SAT terms for RHS, 
        # B == Boundary Coefs,
        # JH == Det of the Jacobian x H tilde,
        # A == D2, 
        # S == SAT Coefs

    if op_load
        print("\nLoading Operators from ../../../../offline_stashed_ops/operators_cache.jld2 ....");
        @load "../../../../offline_stashed_ops/operators_cache.jld2" M B JH HA HS T e H JHI HM Z Hf sJs JIm HB
        print("Done\n");
    else
        print("\nCreating Operators:");
        @time (M, B, JH, HA, HS, T, e, H, JHI, HM, Z, Hf, sJs, JIm, HB) = locoperator_bp5(SBPp, Nq, Nr, Ns, metrics; par=true, nest=true, mode="bp5")
        GC.gc();
        print("\nSaving Metrics to Disk...");
        @save "../../../../offline_stashed_ops/operators_cache.jld2" M B JH HA HS T e H JHI HM Z Hf sJs JIm HB
        print("Done\n");
    end
    
    # initialize time and vector b that stores boundary data (linear system will be Au = b, where b = B*g)
    t = 0 # Inital Time = 0 sec
    b = zeros(3 * Nqp * Nrp * Nsp) # this sucker is bigggggg 

    # initial slip vector
    δ = zeros(2 * Nrp * Nsp) # 2D Plane aghhhh 2 components :|

    # get grid size for setting b
    params = (Nqp, Nrp, Nsp)
    
    # Set face two, remote boundary is opposite the fault
            # Set to a small creep of plate
    remote_boundary = zeros(Nrp * Nsp * 3)
    remote_boundary[Nrp*Nsp+1: 2*Nrp*Nsp] += (t * Vp/2) .* ones(Nrp*Nsp)
    
    # set b for inital displacement calc
    # Note slip will always be multiplied by 1/2 when updating boundary data because of symmetric half space assumption
    bdry_vec_strip!(b, HB, δ ./ 2, remote_boundary, params)

    print("\nCopying HM to Device in CSR format...")
    HM_cu = CuSparseMatrixCSR(HM) # Move PD matrix to GPU
    print("Done\n")
    
    u = zeros(size(b)) # need to initialize u
    b_cu = CuArray(b)
    workspace = Krylov.CgWorkspace(HM_cu, b_cu)
    
    print("\nTime for 1st solve:")
    Krylov.cg!(workspace, HM_cu, b_cu)
    @time Krylov.cg!(workspace, HM_cu, b_cu) # warmup
    u .= Array(workspace.x)
  
    # Following vectors, τ, RSa, θ will only apply to Face 1, and are size 1x(NspxNrp)
    # initialize change in shear stress due to quasi-static deformation
 
    # Set friction coefficients for rate and state
    RS_params = RSht, RSl, RSlf, RSw, RSWf, RShs, RSH, RSamin, RSamax, RSDc, RSVinit
    
    grid_params = (xc[1]:dx:xc[2], yc[1]:dy:yc[2], zc[1]:dz:zc[2],
                    Nqp, Nrp, Nsp)

    # Set initial state variable according to benchmark
    θ, RS_indices, Nθ = set_theta(RS_params, grid_params)
    RSDc = RSDc .* ones(length(θ))
    # Initialize psi version of state variable
    
    # Update friction coefficients based on RS zone
    RSa = initialize_friction_params_vec(RS_params, grid_params, Nθ, RS_indices)
    # Set pre-stress according to benchmark

    σ11_0 = zeros(Nθ)
    σ22_0 = zeros(Nθ)
    σ33_0 = zeros(Nθ)
    # A bit tricky, τ has y and z comp.  scalar pres stress initialized according to BP5 eq 22
    τ0 = σn .* RSa .* asinh.((RSVinit / (2 * RSV0)) .* exp.((RSf0 + RSb * log.(RSV0 / RSVinit)) ./ RSa)) .+ (η * RSVinit)

    Δτ_vec = zeros(2 * length(τ0)) # this will be how stresses change through sim
    τ0_vec = zeros(length(Δτ_vec))

    RSVzero = 1e-20
    V = [RSVinit, RSVzero]
    V_mag = norm(V, 2)

    τ0_vec[1:Nθ] .= τ0
    τ0_vec[1+Nθ: 2*Nθ] .= τ0

    # Set Pre Stress
    Vi = 0.03
    τ_params = Vi, RSV0, RSVinit, σn, η, RSb, RSf0
    set_prestress_QD!(τ0_vec, RS_params, grid_params, τ_params, Nθ, RS_indices, RSDc)

    θ = (θ ./ 0.14) .* RSDc
    ψ = RSf0 .+ RSb .* log.(RSV0 .* θ ./ RSDc)
    τ0_vec[1+Nθ:2*Nθ] .= τ0_vec[1:Nθ]  
    τ0_vec[1:Nθ] .= (τ0_vec[1:Nθ]  .* V[1] ./ V_mag) # set y and z comps
    τ0_vec[1+Nθ:2*Nθ] .= (τ0_vec[1+Nθ:2*Nθ] .* V[2] ./ V_mag)

    # Quick sanity checks
    @assert length(τ0) == length(RSa)
    @assert length(τ0) ==  (RS_indices[1, 2] - RS_indices[1, 1] + 1) * (RS_indices[2, 2] - RS_indices[2, 1] + 1)
    
    # Set initial condition for index 1 DAE - this is a stacked vector of psi, followed by slip
    ψδ = zeros(Nθ + (2* Nrp * Nsp))  #because length(ψ) = 1 * Nrp * Nsp,  length(δ) = 2 * Nrp * Nsp 
    ψδ[1:Nθ] .= ψ[:]
    ψδ[Nθ+1:end] .= δ[:]

    # Plot RS fault and initial stresses for sanity check
    # Comment out if not-needed
    m = length(z)
    n = length(y)
    # Plotting Heat Map of a-b for sanity check 
    cols = RS_indices[1, 2] - RS_indices[1, 1] + 1 # Ys
    rows = RS_indices[2, 2] - RS_indices[2, 1] + 1
    matrix = zeros(m, n) # Z on rows, Y on cols
    ty_matrix = zeros(m, n) # Z on rows, Y on cols
    tz_matrix = zeros(m, n) # Z on rows, Y on cols
    theta_matrix = zeros(m, n) # Z on rows, Y on cols
    for i in RS_indices[2, 1]:RS_indices[2, 2]
        for j in RS_indices[1, 1]:RS_indices[1, 2]
            idx = (i - RS_indices[2, 1]) * cols + (j - RS_indices[1, 1] + 1)
            matrix[i, j] =  (RSa[idx])
            ty_matrix[i, j] = τ0_vec[idx]
            tz_matrix[i, j] = τ0_vec[idx + cols*rows]
            theta_matrix[i, j] = θ[idx]
        end
    end

    ys = y[RS_indices[1, 1]:RS_indices[1, 2]]
    zs = z[RS_indices[2, 1]:RS_indices[2, 2]]
    heatmap(y, z, matrix, title="B - A for Fault")
    png("./BAHEATMAP.png")

    heatmap(y, z, ty_matrix, title="TY on FAULT")
    png("./TYHEATMAP.png")

    heatmap(y, z, tz_matrix, title="Tz on FAULT")
    png("./TzHEATMAP.png")

    heatmap(y, z, theta_matrix, title="theta on FAULT")
    png("./ThetaHEATMAP.png")

    # Setting up the ODE problem
    # Set up stations on fault using Y, Z indices
    stations = [(0.0, 0.0), (0.0, 10.0), (0.0, 22.0), (16.0, 0.0), (16.0, 10.0), (36.0, 0.0), (-16.0, 0.0), (-16.0, 10.0), (-24.0, 10.0), (-36.0, 0.0)] # km
    station_indices = find_station_index(stations, y, z)
    station_strings = [ "0000", "0010", "0022", "1600", "1610", "3600", "-1600", "-1610", "-2410", "-3600"] # str names "$(x_digits)$(y_digits)" where each gets 2 digits e.g y=16,z=10 = "1610"
    
    # Set time span over which to solve:
    tspan = (0, sim_years * year_seconds)

    # Insert problem setups
    flt_loc_y = y[RS_indices[1, 1]:stride_space:RS_indices[1, 2]]
    flt_loc_z = z[RS_indices[2, 1]:stride_space:RS_indices[2, 2]] 
               
    flt_loc_indices = RS_indices
    
    # Set call-back function so that files are written to after successful time steps only.
    cb_fun = SavingCallback((ψδ, t, i) -> write_to_file_BP5(pth, ψδ, t, i, y, z, flt_loc_y, flt_loc_z, flt_loc_indices,station_strings, station_indices, odeparam, "BP5_", 0.1 * year_seconds), SavedValues(Float64, Float64))

    # Start here getting all this machinery working : ()
    # Make text files to store on-fault time series and slip data,
    # Also initialize with initial data:
    create_text_files(pth, flt_loc_y, flt_loc_z, flt_loc_indices, stations, station_strings, station_indices, 0, RSVinit, RSVzero, δ, τ0_vec, θ, y, z)
    
    # set up parameters sent to the right hand side of the DAE:
    odeparam = (reject_step = [false], 
                sim_years =  sim_years,
                Vp=Vp,
                u=u,
                Δτ = Δτ_vec,
                τf = τ0_vec.*ones(length(τ0_vec)),
                b = b,
                μshear=μshear,
                RSa=RSa,
                RSb=RSb,
                σn=σn,
                η=η,
                RSV0=RSV0,
                Nθ = Nθ,
                τ0=τ0_vec,
                RSDc=RSDc,
                RSf0=RSf0,
                HB = HB,
                T = T,
                x = x,
                y = y, 
                z = z,
                e = e,
                sJ = metrics.sJ,
                save_stride_fields = stride_time, # save every save_stride_fields time steps
                RS_params = RS_params,
                RS_indices = RS_indices,
                t_prv = [0.0],
                H = H,
                HM_cu = HM_cu,
                workspace=workspace
                )
    prob = ODEProblem(odefun_bp5, ψδ, tspan, odeparam)
    
    function stepcheck(_, odeparam, _)
        if odeparam.reject_step[1]
            odeparam.reject_step[1] = false
            println("reject")
            return true
        end
        return false
    end
    
    # Solve DAE using Tsit5()
    @time sol = solve(prob, Tsit5(); 
                        isoutofdomain=stepcheck, 
                        dt=0.001,
                        dtmin=1e-8,
                        abstol = 1e-8, reltol = 1e-10, 
                        save_everystep=false, callback=cb_fun)        
    # (sol, z, pth)

    # For plotting

    
end

main()


