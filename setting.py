# config of training
SETTING = {
    'people_num': 10,        # How many people
    'train_num': 18,        # How many train images
    'valid_num': 6,         # How many valid images
    'test_num': 5,          # How many test images
    'train_rounds': 12,   # Number of rounds to train the model and average accuracy over

    'resize_padding': False,  # Whether to perform aspect ratio scaling and padding
    'random_choosing_people': False,  # Whether to random choosing people
    'shuffle': False,        # Whether to shuffle each person's data to train with different data
    'write_report' : True,   # Whether to write experiment report in a file
    'clear_previous_result' : False,  # Whether to clear previous result and write the current result in it
                                      # DON'T SET IT TO TRUE IF YOU WANT TO SAVE RESULTS IN THE PAST 
    'save_model' : True,    # Whether to save model
                            # the code will save the code that gave the best performance (base on precision)
    'model_name' : './models/resnet18_model_origin.pt',      # What's the model name to save
                            # It's only used when save_model = True 
                            # and only_prediction = False

    'only_prediction' : True,  # Whether to only do prediction
    'model_for_prediction' : './models/resnet18_model_origin.pt',   # What's the model name for prediction

    'manual_choosing_peoples' : True, # Whether to pick people manually
                                # because doing prediction with loaded model
                                # will need the model trained by the same set of people 
    'choosed_peoples':     # What people to choose, only used when manual_pick_peoples = 'True'
        [2880, 2937, 8692, 5805, 4153, 9040, 6369, 3332, 7081, 1854],
    

    'meta_data_file':
        'identity_CelebA.txt',  # What's your meta data file
    'original_data_folder':     # What's your original data folder
        'img_align_celeba',
    'blur_data_folder':     # What's your blur data folder
        'data/blur',
    'pixel_data_folder':     # What's your pixelize data folder
        'data/pixelate',
    'deblur_data_folder':     # What's your deblurred data folder
        'data/deblur',

    'dp_blur_data_folder':  # What's your deblurred data folder
        'data/DPblur',
    'dp_pixel_data_folder':  # What's your deblurred data folder
        'data/dp4',

    'detailed_report_file': 'report.txt',   # What's your detailed report file, recommand using txt file
                                            # It's only used when write_report = 'True'    
    'brief_report_file': 'report.csv',      # What's your brief report file, recommand using csv file
                                            # It's only used when write_report = 'True'    

    'train_type': 'dp_blur', # What type of image you want to train
                                # 'original' for original images
                                # 'blur' for blurred images
                                # 'pixel' for pixelized images
                                # 'deblur' for deblurred images
                                # 'dp_blur' for blurred images that added dp noises
                                # 'dp_pixel' for pixelized images that added dp noises


    'train_blur_degree':[15,45,99],    # Blur degree in training, it's what data hacker using to train
                                      # It's only used when train_type = 'blur'
    'test_blur_degree':[15],    # Blur degree in testing, it's what data being attacked

    'train_pixel_degree':[2,4,8,16],    # Pixelize degree in training, it's what data hacker using to train
                                      # It's only used when train_type = 'Pixelize'
    'test_pixel_degree':[2,4,8,16],    # Pixel degree in testing, it's what data being attacked

    'train_epsilon_degree':[0.1], # Epsilon degree in training
                                        # it's only used when train_type is 'dp_blur' or 'dp_pixel'
    'test_epsilon_degree':[0.1, 0.5, 1], # Epsilon degree in testing
                                        # it's only used when train_type is 'dp_blur' or 'dp_pixel'

    'train_dp_blur_degree':[15, 45, 99], # Blur degree in dp blurred training
                                        # it's only used when train_type is 'dp_blur'
    'test_dp_blur_degree':[99],    # Blur degree in dp blurred testing
                                # it's only used when train_type is 'dp_blur'

    'train_dp_pixel_degree':[2,4,8,16], # Pixelize degree in dp pixelized training
                                        # it's only used when train_type is 'dp_pixel'
    'test_dp_pixel_degree':[16],    # Pixelize degree in dp pixelized testing
                                # it's only used when train_type is 'dp_pixel'



    'self_adjust': True,    # Whether to perform self parameter adjusting
                            # If self_adjust = True, the initial value of the parem being adjusted doesn't matter
                            # That is, if param_name = n_epochs and param_values = [5,9] and n_epochs = 3 in MODEL_CONFIG, it will directly start at 5, not 3
    'param_category': 'SETTING', # Which category this parameter belong
                                      # 'SETTING' for params in SETTING
                                      # 'MODEL_CONFIG' for params in MODEL_CONFIG
                                      # It's only used when self_adjust = 'True'
    'param_name' : ['model_for_prediction', 'test_epsilon_degree'],      # Which param to adjust
    'param_values' : [['./models/resnet18_model_dp_blur_0_1.pt', './models/resnet18_model_dp_blur_0_5.pt', './models/resnet18_model_dp_blur_1.pt'], [[0.1], [0.5], [1]]],    # What values to give to the parem
}

# config of model
MODEL_CONFIG = {
    'model_type':'resnet-18', # Which model to train
                            # 'resnet-18' for resnet-18
                            # 'resnet-34' for resnet-34
                            # 'resnet-50' for resnet-50
                            # 'resnext-50' for resnext-50
    'n_epochs': 50,         # Number of epochs
    'batch_size': 4,        # Number of one batch
    'learning_rate': 1e-3,  # Learning rate
    'early_stop': 10,       # How many not improving turns to stop
    'freeze_weights': False, # Whether to freeze weights in the pretrained model
}