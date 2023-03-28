import argparse
import util




def arg_parse():
    parser = argparse.ArgumentParser()
    io_parser = parser.add_mutually_exclusive_group(required=False)
    io_parser.add_argument('--dataset', dest='dataset',
                           help='Input dataset.')
    io_parser.add_argument('--pkl', dest='pkl_fname',
                           help='Name of the pkl data file')
    util.parse_optimizer(parser)
    parser.add_argument('--datadir', dest='datadir',
                        help='Directory where benchmark is located')
    parser.add_argument('--logdir', dest='logdir',
                        help='Tensorboard log directory')
    parser.add_argument('--cuda', dest='cuda',
                        help='CUDA.')
    parser.add_argument('--gpu', dest='gpu', action='store_const',
                        const=True, default=True,
                        help='whether to use GPU.')
    parser.add_argument('--batch_size', dest='batch_size', type=int,
                        help='Batch size.')
    parser.add_argument('--epochs', dest='num_epochs', type=int,
                        help='Number of epochs to train.')
    parser.add_argument('--train_ratio', dest='train_ratio', type=float,
                        help='Ratio of number of graphs training set to all graphs.')
    parser.add_argument('--input_dim', dest='input_dim', type=int,
                        help='Input feature dimension')
    parser.add_argument('--hidden_dim', dest='hidden_dim', type=int,
                        help='Hidden dimension')
    parser.add_argument('--output_dim', dest='output_dim', type=int,
                        help='Output dimension')
    parser.add_argument('--num_classes', dest='num_classes', type=int,
                        help='Number of label classes')
    parser.add_argument('--num_gc_layers', dest='num_gc_layers', type=int,
                        help='Number of graph convolution layers before each pooling')
    parser.add_argument('--bn', dest='bn', action='store_const',
                        const=True, default=False,
                        help='Whether batch normalization is used')
    parser.add_argument('--dropout', dest='dropout', type=float,
                        help='Dropout rate.')
    parser.add_argument('--nobias', dest='bias', action='store_const',
                        const=False, default=True,
                        help='Whether to add bias. Default to True.')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float,
                        help='Weight decay regularization constant.')
    parser.add_argument('--method', dest='method',
                        help='Method. Possible values: base, ')
    parser.add_argument('--debias_ratio', dest='debias_ratio', type=int,
                        help='percentage')
    parser.set_defaults(datadir='data', logdir='log', dataset='credit', opt='adam', opt_scheduler='none',
                        cuda='1', lr=0.001, clip=2.0, batch_size=20, num_epochs=1000,
                        train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1, input_dim=10, hidden_dim=20,
                        output_dim=20, num_classes=2, num_gc_layers=3, dropout=0.0, weight_decay=0.005, method='GAT', debias_ratio=0,
                        explainer_backbone='GNNExplainer',
                        remove=False)
    args=parser.parse_args()

    return args


def arg_parse_explain():

    parser = argparse.ArgumentParser()
    io_parser = parser.add_mutually_exclusive_group(required=False)
    io_parser.add_argument("--dataset", dest="dataset", help="Input dataset.")
    benchmark_parser = io_parser.add_argument_group()
    benchmark_parser.add_argument("--bmname", dest="bmname", help="Name of the benchmark dataset")
    io_parser.add_argument("--pkl", dest="pkl_fname", help="Name of the pkl data file")
    util.parse_optimizer(parser)
    parser.add_argument("--hidden-dim", dest="hidden_dim", type=int, help="Hidden dimension")
    parser.add_argument("--output-dim", dest="output_dim", type=int, help="Output dimension")
    parser.add_argument("--num-gc-layers", dest="num_gc_layers", type=int,
                        help="Number of graph convolution layers before each pooling", )
    parser.add_argument("--bn", dest="bn", action="store_const", const=True, default=False,
                        help="Whether batch normalization is used", )
    parser.add_argument("--clean-log", action="store_true",
                        help="If true, cleans the specified log directory before running.")
    parser.add_argument("--logdir", dest="logdir", help="Tensorboard log directory")
    parser.add_argument("--cuda", dest="cuda", help="CUDA.")
    parser.add_argument("--gpu", dest="gpu", action="store_const", const=True, default=True,
                        help="whether to use GPU.", )
    parser.add_argument("--epochs", dest="num_epochs", type=int, help="Number of epochs to train.")
    parser.add_argument("--dropout", dest="dropout", type=float, help="Dropout rate.")
    parser.add_argument("--nobias", dest="bias", action="store_const", const=False, default=True,
                        help="Whether to add bias. Default to True.", )
    parser.add_argument("--mask-bias", dest="mask_bias", action="store_const", const=True, default=False,
                        help="Whether to add bias. Default to True.", )
    parser.add_argument("--method", dest="method", type=str, help="Method. Possible values: GCN GAT GIN")
    parser.add_argument("--explainer_backbone", dest="explainer_backbone", type=str,
                        help="explainer_backbone. Possible values: GNNExplainer, PGExplainer.")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--fidelity_unfair_weight", type=float)
    parser.add_argument("--fidelity_fair_weight", type=float)
    parser.add_argument("--WD_fair_weight", type=float)
    parser.add_argument("--WD_unfair_weight", type=float)
    parser.add_argument("--KL_weight", type=float)
    parser.add_argument("--debug_mode", default=False)
    parser.add_argument("--baseline", default=False)

    parser.set_defaults(logdir="log", ckptdir="ckpt", dataset="credit", opt="adam", opt_scheduler="none", cuda="0",
                        lr=0.5, clip=2.0*10, batch_size=20, num_epochs=20, hidden_dim=20, output_dim=20, num_gc_layers=3,
                        dropout=0.0, method="GAT", explainer_backbone='GNNExplainer', debug_mode=False, explain_all=False, baseline=False,
                        threshold=500, size_weight=5e-5)
    return parser.parse_args()


