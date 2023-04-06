import argparse


def get_opt():
    parser = argparse.ArgumentParser()

    # add configs
    # ......

    # paths
    parser.add_argument('--data_path', type=str, default='./data.json')

    # checkpoints
    # parser.add_argument()

    # text module configs
    parser.add_argument('--d_transformer', type=int, default=512)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=6)

    # entity params
    parser.add_argument('--ner_corpus', type=str, default='en_core_sci_md')

    # graph params
    parser.add_argument('--gcn_dim', type=int, default=512)
    parser.add_argument('--gcn_layers', type=int, default=2)

    # pretrained mdoel parameters
    parser.add_argument('--version', type=str, default='dmis-lab/biobert-base-cased-v1.2')
    parser.add_argument('--maximum_len', type=int, default=512)

    # Graph Configs
    parser.add_argument('--d_graph', type=int, default=512)

    # Training Configs
    parser.add_argument('--drop_out', type=float, default=0.3)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.001)

    # return args
    return parser.parse_args()

