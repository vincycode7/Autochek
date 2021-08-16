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

    optional.add_argument("-mc", "--model_checkpoint_file", required=False, type=str, 
                            default="./model/prev_model",
                            help="path to the previous model check file.")
    
    optional.add_argument("-dt", "--dta_type", required=False, type=str, 
                            default="valset",
                            help="Type of data to test with, ['trainset', 'testset' or 'valset'].")
    return parser

def main(args):
    # Initialize a data pipline
    data_pipeline = AutochekDataProcessorPipeline()
    
    # load data
    data_pipeline.load_splits(load_ordinary_processed_data=False, load_train_data=False, load_test_data=False, load_val_data=True)

    # load pipeline
    data_pipeline.load_pipeline_nd_normalizer()
    
    # Init model class
    autochekmodel = AutochekModel(model_back_bone=None)
    
    # Load model from check point
    autochekmodel.load_model(args.model_checkpoint_file)
            
    # Test model
    # Transform Test data with pipeline from original data and normalizer from train data
    data_pipeline.pipeline_transform(data="valset", steps='all', return_result=False, normalize=True)
    output = autochekmodel.model_back_bone.predict(X=data_pipeline.currentX)
    val_rmse = mean_squared_error(data_pipeline.currentY, output,squared=False,)
    # Output rmse for test data
    print(f"Rmse for validation is: {val_rmse}")
        

if __name__ == "__main__":
    args = build_argparser().parse_args()
    start = time.time()
    main(args)
    print(f'Testing completed in {(time.time()-start)/60} mins')
