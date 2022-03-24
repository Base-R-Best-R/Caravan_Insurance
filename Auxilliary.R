## Aux ##

# packages
get.package <- function(package){
  
  lapply(package, function(x){
    if(!require(x, character.only = T)){
      install.packages(x)
    }
    library(x, character.only = T)     
  })
  
}

# invis plot
invis.Map <- function(f, ...) invisible(Map(f, ...))
invis.lapply <- function(x, f) invisible(lapply(x, f))
invis.rapply <- function(object, f, classes = "ANY", deflt = NULL, 
                         how = c("unlist", "replace", "list"), ...){
                         invisible(rapply(object, f, classes = "ANY", deflt = NULL,
                         how = c("unlist", "replace", "list"), ...))}

# performance evaluation curves
Eval_Curve_prel <- function(act_label, pred_val){
  
  lapply(act_label, function(x){
    
    # labels
    labels <- as.numeric(pred_val) 
    
    # order
    labels_ordered <- labels[order(x, decreasing = TRUE)]
    
    # df
    roc_dat <- data.frame(TPR = cumsum(labels_ordered) / sum(labels_ordered),
                          FPR = cumsum(!labels_ordered) / sum(!labels_ordered),
                          PRC = cumsum(labels_ordered) / 1:length(labels_ordered))
    
    # return
    roc_dat
    
  }) 
  
}


Eval_Curve <- function(E_Curve_Prel, col, leg_text, RoC = TRUE, act_label = NULL, diss = FALSE, main = ""){
 
  # RoC
  if(RoC == TRUE){
    
    # plot pane
    plot(E_Curve_Prel[[1]]$FPR, E_Curve_Prel[[1]]$TPR, type = "n", ylab = "Sensitivity",
         xlab = "1 - Specificity", main = main)
    grid()
    
    
    # add RoC lines
    invis.Map(function(x, C){
      
      lines(x$FPR, x$TPR, type = "l", col = C, lwd = 2)
      
    }, E_Curve_Prel, col)
    
    lines(c(0, 1), c(0, 1), lty = "dashed", lwd = 2, col = "grey")
    legend("topleft", legend = leg_text, fill = col, cex = 0.7)
    
    if(diss == TRUE){
      text(x = 0.5,y = 0.52,"Alman vs. Predictor", srt = 33, cex = 0.8, col = "grey")
    }
  } else {
    
    # prel
    noobl <- mean(as.numeric(act_label))
    
    # plot pane
    plot(E_Curve_Prel[[1]]$FPR, E_Curve_Prel[[1]]$TPR, type = "n", ylab = "Precision",
         xlab = "Sensitivity", main = main)
    grid()
    
    # add PS lines
    invis.Map(function(x, c){
      
      lines(x$TPR, x$PRC, type = "l", col = c, lwd = 2)
      
    }, E_Curve_Prel, col)
    
    abline(h = noobl, lty = "dashed", lwd = 2, col = "grey")
    legend("topright", legend = leg_text, fill = col, cex = 0.7)
    
    if(diss == TRUE){
      text(y = noobl + 0.021, x = 0.5, "Alman vs. Predictor", cex = 0.9, col = "grey")
    }                           
  }
}

# COIL Performance evaluation function
COIL_per <- function(pred_prob, act_label, n = 800){
  
  # bind
  cmb <- cbind(pred_prob, act_label)
  
  # order by prob and count correctly predicted cases
  sum(cmb[order(cmb[, 1], decreasing = TRUE), ][1:n, 2])
  
}

# Variable Importance for glmnet - small adaptions from caret::varImp
varImp <- function (object, lambda = NULL, ...) 
{
  if (is.null(lambda)) {
    if (length(lambda) > 1) 
      stop("Only one value of lambda is allowed right now")
    if (!is.null(object$lambdaOpt)) {
      lambda <- object$lambdaOpt
    }
    else stop("must supply a value of lambda")
  }
  beta <- predict.glmnet(object, s = lambda, type = "coef")
  if (is.list(beta)) {
    out <- do.call("cbind", lapply(beta, function(x) x[, 
                                                       1]))
    out <- as.data.frame(out, stringsAsFactors = TRUE)
  }
  else out <- data.frame(Overall = beta[,1])
  out <- abs(out[rownames(out) != "(Intercept)",,drop = FALSE])
  out <- out/max(out)
  out[order(out$Overall, decreasing = TRUE),,drop=FALSE]
}

