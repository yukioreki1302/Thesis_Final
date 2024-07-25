#### LDAGM is a deep learning model for predicting lncRNA-disease associations.

#### Extract deep topological features from known associations between lncRNA, diseases, and miRNA, construct a multi-view heterogeneous network, and input it into an MLP model. Use dynamic aggregation layers to control the flow of information between hidden layers, thereby obtaining the optimal feature representation for mining associations between lncRNA and diseases.

---

#### directory structure:

    data_preprocessing: For building heterogeneous networks.
    LDAGM: LDAGM model.
    main: triggering program.
    obtain_nonlinear_features: Nonlinear feature extraction.
