#=
GREASYPOP's synthpop.jl without the edge-building steps.

FLUFFHEAD rebuilds edges itself from group memberships, so generate_networks()
and generate_location_matrices() (which hold every edge of the region in memory)
are skipped. Only the person index that generate_networks() would have saved is
written, so export_groups.jl sees the same ordering as GREASYPOP's adj_mat_keys.
=#

include("households.jl")
include("schools.jl")
include("workplaces.jl")
include("netw.jl")

println("creating people, households, and group quarters")
generate_people()
println("creating schools")
generate_schools()
println("creating workplaces")
generate_commute_matrices()
generate_jobs_and_workers()
println("indexing people (edges are built by FLUFFHEAD)")
p_idxs, dummy_idxs = person_indices()
ser_path("jlse/adj_dummy_keys.jlse", Dict(v => k for (k, v) in dummy_idxs))
merge!(p_idxs, dummy_idxs)
ser_path("jlse/adj_mat_keys.jlse", first.(sort(collect(p_idxs), by = p -> p[2])))
println("done")
