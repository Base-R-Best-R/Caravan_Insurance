Predicting Caravan Insurance
================

This repository contains a course project including code as well as a
paper on the application of machine learning methods to predict caravan
insurance takeout. The data originates from a [Kaggle
competition](https://www.kaggle.com/uciml/caravan-insurance-challenge).

# Group Members

1.  Fabian Blasch, 01507223
2.  Gregor Steiner, 01615340
3.  Sophie Steininger, 01614383
4.  Jakob Zellmann, 11708938

# Links

-   [Course
    Paper](https://github.com/Base-R-Best-R/Caravan_Insurance/blob/main/Paper/Paper_Caravan.pdf)

-   [Auxiliary
    Functions](https://github.com/Base-R-Best-R/Caravan_Insurance/blob/main/Auxilliary.R)

    -   COIL\_per() - calculate predictive performance as measured in
        the competition
    -   Eval\_Curve() - generate RoC and PS curves given predicted
        outcomes and actual labels
    -   Eval\_Curve\_Prel() - helper function to Eval\_Curve()
    -   varImp() - adaption to caret::varIMP()
    -   get.package() - check which of the given packages are installed,
        call all packages which are, install the remaining packages
