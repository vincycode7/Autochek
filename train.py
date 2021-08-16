# from process_data import *
from process import *
from model import *
import pickle,argparse,time

#parser
def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = argparse.ArgumentParser("Train a model, a quick command to train is: reset && python3 train.py -bbt 'svr' -lmc 1 -tsm 1 -trm 0 -sm 0 -pmc './model/model1_svr'")

    # -- Add required and optional groups
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    optional.add_argument("-bbt", "--back_bone_type", required=False, type=str, 
                            default="svc",
                            help="type of back bone to use in training.")
    
    optional.add_argument("-lmc", "--load_model_from_checkpoint", type=int, default=0,
                            help="load model from check point.")

    optional.add_argument("-tsm", "--test_model_on_testset", type=int, default=0,
                            help="test model on testset.")
    
    optional.add_argument("-trm", "--trainmodel", type=int, default=0,
                            help="train model")

    optional.add_argument('-sm', "--save_model", type=int, default=0, help="Save trained model?.")
    optional.add_argument('-nmc', "--new_model_checkpoint_file", type=str, default='./model/new_model', help="name to save the new model as.")
    optional.add_argument("-pmc", "--prev_model_checkpoint_file", required=False, type=str, 
                            default="./model/prev_model",
                            help="path to the previous model check file.")
    return parser

def main(args):
    if args.load_model_from_checkpoint:
        assert type(args.prev_model_checkpoint_file) == type("check_point"), "Previous check point path has to be of type 'string'"
        
    if args.save_model:
        assert type(args.new_model_checkpoint_file) == type("check_point"), "New check point path has to be of type 'string'"
        
    if type(args.back_bone_type) == type("back_bone"):
        assert args.back_bone_type in AutochekModel._supported_backbones.keys(), f"The specified backbone is not in the system please choose one of the following, {AutochekModel._supported_backbones.keys()}."
        
    # Initialize a data pipline
    data_pipeline = AutochekDataProcessorPipeline()
    
    # load data
    data_pipeline.load_splits(load_ordinary_processed_data=False, load_train_data=True, load_test_data=True, load_val_data=False)

    # load pipeline
    data_pipeline.load_pipeline_nd_normalizer()
    
    autochekmodel = AutochekModel(model_back_bone=None)
    
    if args.load_model_from_checkpoint:
        autochekmodel.load_model(args.prev_model_checkpoint_file)
    else:
        autochekmodel.init_model_from_supported_backbones(backbone_type=args.back_bone_type)
        
    if args.trainmodel:
            # Transform Train data with pipeline from original data and normalizer from train data
        data_pipeline.pipeline_transform(data="trainset", steps='all', return_result=False, normalize=True)
        autochekmodel.model_back_bone.fit(X=data_pipeline.currentX, y=data_pipeline.currentY)
        if args.trainmodel and args.save_model:
            autochekmodel.save_model(args.new_model_checkpoint_file)
        output = autochekmodel.model_back_bone.predict(X=data_pipeline.currentX)
        train_rmse = mean_squared_error(data_pipeline.currentY, output, squared=False)
        
        # Output rmse for train data
        print(f"Rmse for train is: {train_rmse} ")
    
    if args.test_model_on_testset:
        # Transform Test data with pipeline from original data and normalizer from train data
        data_pipeline.pipeline_transform(data="testset", steps='all', return_result=False, normalize=True)
        output = autochekmodel.model_back_bone.predict(X=data_pipeline.currentX)
        test_rmse = mean_squared_error(data_pipeline.currentY, output,squared=False,)
        # Output rmse for test data
        print(f"Rmse for test is: {test_rmse}")
        

if __name__ == "__main__":
# reset && python3 train.py -tdp dataset/labels/test_new1.csv -tdr dataset/Images/test1/ -vdr dataset/Images/test2/ -vdp dataset/labels/test_new2.csv -tm models/other_models/modelsmodel13_train.pt -vm models/other_models/modelsmodel12_val.pt -mon 'modelsmodel14'
    args = build_argparser().parse_args()
    start = time.time()
    main(args)
    print(f'Training completed in {(time.time()-start)/60} mins')
