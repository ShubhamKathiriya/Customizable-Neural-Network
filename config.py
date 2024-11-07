class config:


    ######################################### Data samples #########################################

    train_samples = 10000
    validation_samples = 1000
    test_samples = 2000
    
    ################################################################################################



    #########################################  Dimenstions #########################################

    input_dim = 2      # input dimesion
    hidden_1_dim = 64  # first hidden layer dimension
    num_classes = 3    # number of classes

    ################################################################################################




    #########################################  Regularization ######################################

    # L1 Regularization
    weight_regularizer_l1 = 0
    bias_regularizer_l1 = 0


    # L2 Regularization
    weight_regularizer_l2 = 5e-4
    bias_regularizer_l2 = 5e-4

    ################################################################################################



    #########################################  Optimizator  ########################################

    learning_rate = 0.05
    decay = 5e-5
    epsilon = 1e-7

    # rms_prop setting
    rho = 0.9

    # adam_setting
    beta_1 = 0.9
    beta_2 = 0.999

    ################################################################################################



    #########################################  Others  ########################################
    
    epochs = 10000
    log_step = 100
    dropout_rate = 0.1

    output_clip = 1e-7  # for final layer output clipping (reasons mentioned in README.md)

    weight_initiase_factor = 0.01   # random initiasation of weight will multiply with this factor as better initialization

    ################################################################################################

