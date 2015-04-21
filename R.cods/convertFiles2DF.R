args <- args <- commandArgs(trailingOnly = TRUE)

gl <- list.files(path=paste("results/", args[1], sep=""), pattern="pred*", full.names=T)
print(gl[1])

cvpaste <- function(str) {
	paste(str, ".cv", sep="")
}

fl <- sapply(gsub("//", "/cv/", gl), cvpaste)
print(fl[1])

dl <- lapply(fl, read.table)
print(str(dl[[1]]))

df <- do.call("cbind", dl)
print(str(df[,1:5]))


write.table(df, file=paste("results/", args[1], "/res_cv.dat", sep=""), row.names=F, col.names=F)


### Readine the predictions for the test dataset
dl <- lapply(gl, read.table)
print(str(dl[[1]]))

df <- do.call("cbind", dl)
print (str(df[,1:5]))

write.table(df, file=paste("results/", args[1], "/res_test.dat", sep=""), row.names=F, col.names=F)
