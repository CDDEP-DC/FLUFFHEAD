#=
Export a GREASYPOP synthetic population as group memberships instead of edges.

Run in a GREASYPOP working directory after synthpop.jl:

    julia --project=<greasypop_env> export_groups.jl

FLUFFHEAD rebuilds within-group edges itself (all at once, or lazily per group)
with the same rules netw.jl uses: households complete, workplaces and schools
stochastic block models, group quarters small-world. Writes to "groups_export":

    people.csv       idx (0-based, same order as GREASYPOP's adj_mat_keys), person
                     attributes; dummies (live outside, work inside) have hh_id -1
    members_hh.csv   group, idx
    members_wp.csv   group, idx, block (income category)
    members_sch.csv  group, idx, block (grade)
    members_gq.csv   group, idx
    wp.csv           group, geo (employer location), type
    sch.csv          group, code
    gq.csv           group, cbg_id, type
    cbg.csv          cbg_id, geocode
    outside_workers.csv   idx of residents working outside the area
    params.json      network settings from config.json
=#

include("netw.jl")

mkpath("groups_export")

keys_vec = dser_path("jlse/adj_mat_keys.jlse")
p_idx = Dict(k => i - 1 for (i, k) in enumerate(keys_vec))

println("people")
people = dser_path("jlse/people.jlse")
n = length(keys_vec)
age = fill(-1, n); female = fill(-1, n); working = fill(-1, n); commuter = fill(-1, n)
grade = fill("", n)
for (k, v) in people
    i = p_idx[k] + 1
    ## attributes can be missing (e.g. no school grade); use -1 / ""
    age[i] = coalesce(mcon(Int, v.age), -1); female[i] = coalesce(mcon(Int, v.female), -1)
    working[i] = coalesce(mcon(Int, v.working), -1); commuter[i] = coalesce(mcon(Int, v.commuter), -1)
    grade[i] = coalesce(mcon(String, v.sch_grade), "")
end
dummies = Set(get_dummies())
write_df("groups_export/people.csv", DataFrame(
    idx = 0:n-1,
    p_id = [Int(k[1]) for k in keys_vec],
    hh_id = [k in dummies ? -1 : Int(k[2]) for k in keys_vec],
    cbg_id = [Int(k[3]) for k in keys_vec],
    age = age, female = female, working = working, commuter = commuter, sch_grade = grade))
people = nothing; GC.gc()

d = dser_path("jlse/cbg_idxs.jlse")
write_df("groups_export/cbg.csv", DataFrame(sort([(cbg_id = Int(k), geocode = String(v)) for (k, v) in d], by = x -> x.cbg_id)))

## one row per membership; blocks are the SBM groups netw.jl uses
function write_members(path, groups; block = nothing)
    g = Int[]; idx = Int[]; b = String[]
    for (gi, members) in enumerate(groups)
        for m in unique(members)
            push!(g, gi - 1); push!(idx, p_idx[m[1:3]])
            block === nothing || push!(b, string(m[4]))
        end
    end
    df = block === nothing ? DataFrame(group = g, idx = idx) : DataFrame(group = g, idx = idx, block = b)
    write_df(path, df)
end

println("households")
hh = read_hh_ppl()
write_members("groups_export/members_hh.csv", collect(values(hh)))
hh = nothing; GC.gc()

println("workplaces")
cw = dser_path("jlse/company_workers.jlse")
wkeys = collect(keys(cw))
write_members("groups_export/members_wp.csv", [cw[k] for k in wkeys]; block = true)
write_df("groups_export/wp.csv", DataFrame(group = 0:length(wkeys)-1,
    geo = [String(k[3]) for k in wkeys], type = [Int(k[2]) for k in wkeys]))
cw = nothing; GC.gc()

println("schools")
sch = read_sch_ppl()
skeys = collect(keys(sch))
write_members("groups_export/members_sch.csv", [sch[k] for k in skeys]; block = true)
write_df("groups_export/sch.csv", DataFrame(group = 0:length(skeys)-1, code = [String(k) for k in skeys]))
sch = nothing; GC.gc()

println("group quarters")
gq = read_gq_ppl()
gqinfo = dser_path("jlse/gqs.jlse")
gkeys = collect(keys(gq))
write_members("groups_export/members_gq.csv", [gq[k] for k in gkeys])
write_df("groups_export/gq.csv", DataFrame(group = 0:length(gkeys)-1, cbg_id = [Int(k[2]) for k in gkeys],
    type = [haskey(gqinfo, k) ? String(gqinfo[k].type) : "" for k in gkeys]))

ow = dser_path("jlse/outside_workers.jlse")
write_df("groups_export/outside_workers.csv", DataFrame(idx = sort([p_idx[only(x)[1:3]] for x in values(ow)])))

dConfig = tryJSON("config.json")
open("groups_export/params.json", "w") do f
    JSON.print(f, Dict(
        "workplace_K" => get(dConfig, "workplace_K", 8), "school_K" => get(dConfig, "school_K", 12),
        "gq_K" => get(dConfig, "gq_K", 12), "netw_B" => get(dConfig, "netw_B", 0.25),
        "income_associativity_coefficient" => get(dConfig, "income_associativity_coefficient", 0.9),
        "school_associativity_coefficient" => get(dConfig, "school_associativity_coefficient", 0.9)), 2)
end
println("done")
