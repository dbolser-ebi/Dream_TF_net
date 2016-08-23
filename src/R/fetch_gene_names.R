####R code for fetching gene names with biomaRt #####
require(biomaRt)
csv.data <- read.delim('../../data/rnaseq/gene_expression.A549.biorep1.tsv', header=TRUE)
annot.table  <- csv.data[,1] # This is your annotation table, it can't be NULL
# Collect ensembl IDs from annot.table before converting to normal gene names.
ensembl_ids <- annot.table # Get this from may be from annot.table
# Prepare gene table with some simple caching to avoid stressing the Ensembl server by many repeated runs
genes.table = NULL

if (!file.exists("cache.genes.table")) {
    message("Retrieving genes table from Ensembl...")
    mart <- useMart("ensembl")
    #listDatasets(mart=mart)
    mart <- useDataset("hsapiens_gene_ensembl", mart = mart)
    genes.table <- getBM(filters= "ensembl_gene_id",
                         attributes= c("ensembl_gene_id", "hgnc_symbol"), values=gsub("\\..*","",csv.data$gene_id), mart= mart)
    save(genes.table, file= "cache.genes.table")
} else {
    load("cache.genes.table")
    message("Reading gene annotation information from cache file:
cache/cache.genes.table
            Remove the file if you want to force retrieving data from
Ensembl")
}

write.table(genes.table, "../../data/preprocess/gene_ids.data", quote=FALSE, col.names = FALSE, row.names = FALSE)
