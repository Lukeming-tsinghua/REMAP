import argparse


def boolean_string(s):
    if s not in {'True', 'False'}:
        raise ValueError("%s: Not a valid Boolean argument string" % s)
    return s == 'True'


def get_args():
    parser = argparse.ArgumentParser(description="Training Arguments")
    parser.add_argument('--train_pkl', type=str, help="path of data pickle")
    parser.add_argument('--valid_pkl', type=str, help="path of data pickle")
    parser.add_argument('--pred_pkl', type=str, help="path of data pickle")
    parser.add_argument('--dict', type=str, help="path of cui2idx dict pickle")
    parser.add_argument('--tokenizer', type=str, help="tokenizer config")
    parser.add_argument('--model', type=str, help="model config")
    parser.add_argument('--sampleNum',
                        type=int,
                        help="number of sentences in an entity pair")
    parser.add_argument('--maxLength',
                        type=int,
                        help="max length of sentences in an entity pair")
    parser.add_argument('--testRate', type=float, help="test set rate")
    parser.add_argument('--trainBatchSize',
                        type=int,
                        help="batchsize for training set")
    parser.add_argument('--testBatchSize',
                        type=int,
                        help="batchsize for validation set")
    parser.add_argument('--weightDecay', type=float, help="l2 penalty")
    parser.add_argument('--epoch', type=int, help="training epoch")
    parser.add_argument('--lr', type=float, help="learning rate")
    parser.add_argument('--gamma', type=float, help="gamma in multi step lr")
    parser.add_argument('--nworkers',
                        type=int,
                        help="number of workers for dataloaders")
    parser.add_argument('--pinMemory', type=bool, help="pin memory")
    parser.add_argument('--output_path', type=str, help="output path")
    parser.add_argument('--graphPath', type=str, help="path of graph data")
    parser.add_argument('--initEmbedding', type=str, help="initial node embedding")
    parser.add_argument('--cuda', type=int, help="cuda rank id")
    parser.add_argument('--accumulate_step', type=int, help="gradient accumulate step")
    parser.add_argument('--do_train', default=False, type=boolean_string, help="do train or not")
    parser.add_argument('--do_eval', default=False, type=boolean_string, help="do eval or not")
    parser.add_argument('--do_pred', default=False, type=boolean_string, help="do prediction or not")
    parser.add_argument('--init_embedding', default=True, type=boolean_string, help="initiate embedding with EHR")
    parser.add_argument('--seed', type=int, help="torch manual seed",default=0)
    parser.add_argument('--valid_epoch', type=int, help="how much epoch between per valid",default=1)
    args = parser.parse_args()
    return args


def print_args(args):
    print('--------args----------')
    for k in list(vars(args).keys()):
        print('%s: %s' % (k, vars(args)[k]))
    print('--------args----------\n')
