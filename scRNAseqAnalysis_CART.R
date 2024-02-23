library(Seurat)
library(dplyr)
library(patchwork)
library(ggplot2)
library(RColorBrewer)
library(rcna)
library(tidyverse)
library(ggrepel)
library(ggthemes)
library(glue)
library(harmony)
library(Matrix)
library(data.table)

# reads in the Seurat object
cart <- readRDS("CART_Data.RDS")

# creates column in metadata for QC scores
cart[["percent.mt"]] <- PercentageFeatureSet(cart, pattern = "^MT-")

# mitochondrial genes (can indicate cell state) 
cart[["percent.mt"]] <- PercentageFeatureSet(cart, pattern = "^MT-")

# ribosomal proteins
cart[["percent.rb"]] <- PercentageFeatureSet(cart, pattern = "^RP[SL]")

# filter out RNA sequences with high ribosome counts
cart <- subset(cart, subset = percent.mt < 10)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# read patient data from csv file
patientInfo <- data.frame(read.csv("/Users/shivam/Documents/11th Grade Science Fair/Programs/patientInfo.csv"))
assign("numFeatures", 16)

# function to find the patient's array index
findIDIndex <- function(patientID, allIDs) {
  for (i in 1:length(allIDs)) {
    if (patientID == allIDs[i])
      return(i)
  }
}

# creates a list for the clinical features of each patient (ac01, ac02, etc.)
allData <- NULL
for (i in 1:nrow(patientInfo)) {
  patientID <- patientInfo$Sample.ID[i]
  info <- vector(mode = 'list', length = 16)
  for (j in 1:numFeatures) {
    info[j] <- patientInfo[i, j + 1]
  }
  allData <- c(allData, info)
}

# creates new columns in cart@meta.data for all column names
columnNames <- c("Histology", "Age", "Gender", "ECOG", "Stage", "IPIScore", "NumLinesOfPrevTherapy", "PrevRefractory", "PriorAutoTransplant", "MaxCRS", "ThreeMoPETorCT", "MaxICANS", "ICANSGroup", "ICANSTimeToOnset", "ICANSDuration", "ProlongedCytopenia")
x <- cart@meta.data

# Histology Feature
newCol <- NULL
for (i in 1:nrow(x)) {
  patientID <- x$orig.ident[i]
  patientIndex <- findIDIndex(patientID, patientInfo$Sample.ID)
  infoIndex <- patientIndex * 16 - 15
  newCol <- c(newCol, allData[infoIndex])
}
cart@meta.data$Histology <- newCol

# Age Feature
newCol <- NULL
for (i in 1:nrow(x)) {
  patientID <- x$orig.ident[i]
  patientIndex <- findIDIndex(patientID, patientInfo$Sample.ID)
  infoIndex <- patientIndex * 16 - 14
  newCol <- c(newCol, allData[infoIndex])
}
cart@meta.data$Age <- newCol

# Gender Feature
newCol <- NULL
for (i in 1:nrow(x)) {
  patientID <- x$orig.ident[i]
  patientIndex <- findIDIndex(patientID, patientInfo$Sample.ID)
  infoIndex <- patientIndex * 16 - 13
  newCol <- c(newCol, allData[infoIndex])
}
cart@meta.data$Gender <- newCol

# ECOG Feature
newCol <- NULL
for (i in 1:nrow(x)) {
  patientID <- x$orig.ident[i]
  patientIndex <- findIDIndex(patientID, patientInfo$Sample.ID)
  infoIndex <- patientIndex * 16 - 12
  newCol <- c(newCol, allData[infoIndex])
}
cart@meta.data$ECOG <- newCol

# Stage Feature
newCol <- NULL
for (i in 1:nrow(x)) {
  patientID <- x$orig.ident[i]
  patientIndex <- findIDIndex(patientID, patientInfo$Sample.ID)
  infoIndex <- patientIndex * 16 - 11
  newCol <- c(newCol, allData[infoIndex])
}
cart@meta.data$Stage <- newCol

# IPI Score Feature
newCol <- NULL
for (i in 1:nrow(x)) {
  patientID <- x$orig.ident[i]
  patientIndex <- findIDIndex(patientID, patientInfo$Sample.ID)
  infoIndex <- patientIndex * 16 - 10
  newCol <- c(newCol, allData[infoIndex])
}
cart@meta.data$IPIScore <- newCol

# Number of Lines of Therapy Feature
newCol <- NULL
for (i in 1:nrow(x)) {
  patientID <- x$orig.ident[i]
  patientIndex <- findIDIndex(patientID, patientInfo$Sample.ID)
  infoIndex <- patientIndex * 16 - 9
  newCol <- c(newCol, allData[infoIndex])
}
cart@meta.data$NumLinesOfPrevTherapy <- newCol

# Previous Refractory Feature
newCol <- NULL
for (i in 1:nrow(x)) {
  patientID <- x$orig.ident[i]
  patientIndex <- findIDIndex(patientID, patientInfo$Sample.ID)
  infoIndex <- patientIndex * 16 - 8
  newCol <- c(newCol, allData[infoIndex])
}
cart@meta.data$PrevRefractory <- newCol

# Prior Auto Translplant Feature
newCol <- NULL
for (i in 1:nrow(x)) {
  patientID <- x$orig.ident[i]
  patientIndex <- findIDIndex(patientID, patientInfo$Sample.ID)
  infoIndex <- patientIndex * 16 - 7
  newCol <- c(newCol, allData[infoIndex])
}
cart@meta.data$PriorAutoTransplant <- newCol

# Max CRS score
newCol <- NULL
for (i in 1:nrow(x)) {
  patientID <- x$orig.ident[i]
  patientIndex <- findIDIndex(patientID, patientInfo$Sample.ID)
  infoIndex <- patientIndex * 16 - 6
  newCol <- c(newCol, allData[infoIndex])
}
cart@meta.data$MaxCRS <- newCol

# Three Month PET/CT Scan Feature
newCol <- NULL
for (i in 1:nrow(x)) {
  patientID <- x$orig.ident[i]
  patientIndex <- findIDIndex(patientID, patientInfo$Sample.ID)
  infoIndex <- patientIndex * 16 - 5
  newCol <- c(newCol, allData[infoIndex])
}
cart@meta.data$ThreeMoPETorCT <- newCol

# Max ICANS Score Feature
newCol <- NULL
for (i in 1:nrow(x)) {
  patientID <- x$orig.ident[i]
  patientIndex <- findIDIndex(patientID, patientInfo$Sample.ID)
  infoIndex <- patientIndex * 16 - 4
  newCol <- c(newCol, allData[infoIndex])
}
cart@meta.data$MaxICANS <- newCol

# ICANS Group Feature
newCol <- NULL
for (i in 1:nrow(x)) {
  patientID <- x$orig.ident[i]
  patientIndex <- findIDIndex(patientID, patientInfo$Sample.ID)
  infoIndex <- patientIndex * 16 - 3
  newCol <- c(newCol, allData[infoIndex])
}
cart@meta.data$ICANSGroup <- newCol

# ICANS Time to Onset Feature
newCol <- NULL
for (i in 1:nrow(x)) {
  patientID <- x$orig.ident[i]
  patientIndex <- findIDIndex(patientID, patientInfo$Sample.ID)
  infoIndex <- patientIndex * 16 - 2
  newCol <- c(newCol, allData[infoIndex])
}
cart@meta.data$ICANSTimeToOnset <- newCol

# ICANS Duration Feature
newCol <- NULL
for (i in 1:nrow(x)) {
  patientID <- x$orig.ident[i]
  patientIndex <- findIDIndex(patientID, patientInfo$Sample.ID)
  infoIndex <- patientIndex * 16 - 1
  newCol <- c(newCol, allData[infoIndex])
}
cart@meta.data$ICANSDuration <- newCol

# Prolonged Cytopenia Feature
newCol <- NULL
for (i in 1:nrow(x)) {
  patientID <- x$orig.ident[i]
  patientIndex <- findIDIndex(patientID, patientInfo$Sample.ID)
  infoIndex <- patientIndex * 16
  newCol <- c(newCol, allData[infoIndex])
}
cart@meta.data$ProlongedCytopenia <- newCol

# Cluster Grouping Feature
newCol <- NULL
for (i in 1:length(cart@active.ident)) {
  newVal <- as.integer(cart@active.ident[i])
  newCol <- c(newCol, newVal)
}
cart@meta.data$Cluster <- newCol


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# plot uniform manifold approximation and projection (UMAP) with centered title
UMAPPlot(cart, label = TRUE, repel = TRUE) + ggtitle("UMAP for all CAR T-Cell Therapy Patients") + theme(plot.title = element_text(hjust = 0.5))

# split cart dataset based on CRS grades
CRS0 <- subset(cart, subset = MaxCRS == 0)
CRS1 <- subset(cart, subset = MaxCRS == 1)
CRS2 <- subset(cart, subset = MaxCRS == 2)
CRS3 <- subset(cart, subset = MaxCRS == 3)
CRS4 <- subset(cart, subset = MaxCRS == 4)

clusters <- FindClusters(cart)

options(max.print=999999)
cart <- PrepSCTFindMarkers(cart)
datafile <- FindAllMarkers(cart)
write.csv(datafile, file = "ClusterBiomarkerList.csv")

# view heatmap of genes
DimHeatmap(cart, cells = 10000, nfeatures = 50)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# conduct covarying neighborhood analysis (CNA)

cartCRS04 <- subset(cart, subset = MaxCRS == '0' | MaxCRS == '4')

cartCRS04@meta.data$MaxCRS_val <- as.numeric(cartCRS04@meta.data$MaxCRS) # comparison between CRS scores of 0 and 4


cartCRS04 <- association.Seurat(seurat_object = cartCRS04, test_var = 'MaxCRS_val', samplem_key = 'orig.ident', 
                                graph_use = 'SCT_nn', verbose = TRUE, batches = NULL, covs = NULL)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# subset the cart dataset into cluster subsets to determine the type of T-cell

cartCluster <- subset(cart, subset = Cluster == '7')
FeaturePlot(cartCluster, features = c("CD3D", "CD4", "CD8A", "SELL", "CD27", "CD69", "CCR7", "CD28", "CD44", "PDCD1", "LAG3", "NCAM1"), label = TRUE, keep.scale = "all")
FeaturePlot(cartCluster, features = c("CD3D", "CD4", "CD8A"), label = TRUE, keep.scale = "all")

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# conduct differential gene expression analysis (DGEA) to isolate significant biomarkers

cartPeriph <- subset(cart, subset = Cluster == c('0','4','5','7','8','10','11','15','18','20','22','27','32','34'))
cartBlob <- subset(cart, subset = Cluster == c('1','2','3','6','9','12','13','14','16','17','19','21','23','24','25','26','28','29','30','31','33'))

cartBlue <- subset(cart, subset = Cluster == c('7','10','11','18','20','34'))
cartOther <- subset(cart, subset = Cluster == c('0','1','2','3','4','5','6','8','9','12','13','14','15','16','17','19','21','22','23','24','25','26','27','28','29','30','31','32','33'))

cart <- PrepSCTFindMarkers(cart)
diff <- FindMarkers(cart, ident.1 = c('7','10','11','18','20','34'), ident.2 = c('0','1','2','4','5','6','8','9','12','13','14','15','16','17','19','21','22','23','24','25','26','27','28','29','30','31','32','33'))
write.csv(diff, "DifferentialGenes.csv")


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# create volcano plots after differential gene expression

dgeData <- read.csv("DifferentialGenes.csv")

# set plot theme
theme_set(theme_classic(base_size = 20) +
            theme(
              axis.title.y = element_text(face = "bold", margin = margin(0,20,0,0), size = rel(1.1), color = 'black'),
              axis.title.x = element_text(hjust = 0.5, face = "bold", margin = margin(20,0,0,0), size = rel(1.1), color = 'black'),
              plot.title = element_text(hjust = 0.5)
            ))

# Add a column to the data frame to specify if they are UP- or DOWN- regulated (log2fc respectively positive or negative)<br /><br /><br />
dgeData$diffexpressed <- "NO"
# if log2Foldchange > 0.6 and pvalue < 0.05, set as "UP"
dgeData$diffexpressed[dgeData$avg_log2FC > 0.6 & dgeData$p_val < 0.05] <- "UP"
# if log2Foldchange < -0.6 and pvalue < 0.05, set as "DOWN"
dgeData$diffexpressed[dgeData$avg_log2FC < -0.6 & dgeData$p_val < 0.05] <- "DOWN"

# add new column with labels for DGEA
newDGECol <- NULL
for(i in 1:nrow(dgeData)) {
  if (i < 21) {
    newDGECol <- c(newDGECol, dgeData[["X"]][i])
  }
  if (i > 20) {
    newDGECol <- c(newDGECol, NA)
  }
}
dgeData$labels <- newDGECol

# plots CNA graph for visual representations of clusters and their correlations
ggplot(data = dgeData, aes(x = avg_log2FC, y = -log10(p_val), col = diffexpressed, label = .data[["labels"]])) +
  geom_vline(xintercept = c(-0.7, 0.7), col = "gray", linetype = 'dashed') +
  geom_hline(yintercept = -log10(0.05), col = "gray", linetype = 'dashed') + 
  geom_point(size = 2) + 
  scale_color_manual(values = c("orange", "red", "blue"), labels = c("Downregulated", "Insignificant", "Upregulated")) +
  coord_cartesian(ylim = c(0, 300), xlim = c(-5, 10)) + # limits set since some genes have infinite -log(p_val_adj)
  labs(x = expression("log"[2]*"FC"), y = expression("-log"[10]*"p-value")) + 
  scale_x_continuous(breaks = seq(-10, 10, 2)) + # to customize the breaks in the x axis
  ggtitle('Gene Expression of CRS = 0 vs. CRS = 4') + # plot title 
  geom_text_repel(max.overlaps = Inf) # shows all labels

# END OF PROGRAM

