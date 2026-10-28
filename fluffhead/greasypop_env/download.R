# Download GREASYPOP inputs for a list of states into one shared store.
#
#   Rscript download.R <store_dir> <greasypop_src> <state_fips>...
#
# Uses GREASYPOP's own pull_datasets.R. Run with the working directory set to
# <store_dir>, which holds a .here marker so here() resolves to it. Every state
# gets ACS tables, PUMS and block group shapefiles; a region then links only the
# files it needs. Steps already on disk are skipped, so an interrupted download
# can be rerun. LODES are fetched by greasypop.py instead (lehdr holds a whole
# state's block-level file in memory; California alone exhausts 16 GB).

suppressPackageStartupMessages({
  library(here)
})

args <- commandArgs(trailingOnly = TRUE)
store <- normalizePath(args[1])
src <- normalizePath(args[2])
states <- args[-(1:2)]
setwd(store)
source(file.path(src, "pull_datasets.R"))

key <- Sys.getenv("CENSUS_API_KEY")
if (key == "") stop("CENSUS_API_KEY is not set")

main_year <- 2019
decennial_year <- 2010
acs_required <- c("B01001", "B09018", "B09019", "B09020", "B09021", "B11004", "B11012",
                  "B11016", "B19001", "B22010", "B23009", "B23025", "B25006", "B11001H",
                  "B11001I", "C24010", "C24030")
dec_required <- c("P43")

for (st in states) {
  abbr <- fips_info(st)$abbr
  message("=== ", st, " ", abbr)
  # one state's failure shouldn't stop the others; rerun to retry
  tryCatch({
    if (length(list.files(file.path("census", toupper(abbr)))) < length(acs_required) + length(dec_required)) {
      pull_census_data(state_fips = st, year_ACS = main_year, year_DEC = decennial_year,
                       ACS_table_codes = acs_required, DEC_table_codes = dec_required, key = key)
    }
    if (!file.exists(file.path("pums", paste0("psam_p", st, ".csv")))) {
      pull_pums_data(states = abbr, year = main_year)
    }
    if (!file.exists(file.path("geo", paste0("tl_", main_year, "_", st, "_bg.zip")))) {
      download_shapefiles(st, main_year)
    }
    message("=== done ", st)
  }, error = function(e) message("=== FAILED ", st, ": ", conditionMessage(e)))
}
