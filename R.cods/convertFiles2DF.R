gl <- list.files(path="results/hf1/", pattern="pred*", full.names=T)


cvpaste <- function(str) {
	paste(str, ".cv", sep="")
}

fl <- sapply(gsub("//", "/cv/", gl), cvpaste)

dl <- lapply(fl, read.table)

df <- do.call("cbind", dl)

write.table(df, file="results/hf1/res_cv.dat", row.names=F, col.names=F)


### Readine the predictions for the test dataset
dl <- lapply(gl, read.table)
df <- do.call("cbind", dl)

write.table(df, file="results/hf1/res_test.dat", row.names=F, col.names=F)
