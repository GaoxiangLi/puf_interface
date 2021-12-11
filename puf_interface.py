import argparse


def main(args):
    plus_control = args.plus
    minus_control = args.minus
    permutation_control = args.permutation

    if plus_control:
        import interface_plus
        # run(n: int, k: int, N: int, seed_sim: int, noisiness: float, BATCH_SIZE: int, interface: bool,
        #     # ghost_bit_len: int, group: int, puf: str) -> dict:
        interface_plus.run(args.n, args.k, args.N, args.seed, args.noise, args.batch_size, args.interface,
                           args.ghost_bit_len,
                           args.group, args.puf)
    if minus_control:
        import interface_minus
        # run(n: int, k: int, N: int, seed_sim: int, noisiness: float, BATCH_SIZE: int, interface: bool,
        #     # double_use_bit_len: int, group: int, puf: str) -> dict:
        interface_minus.run(args.n, args.k, args.N, args.seed, args.noise, args.batch_size, args.interface,
                            args.double_use_bit_len,
                            args.group, args.puf)
    if permutation_control:
        import interface_permutation
        # run(n: int, k: int, N: int, seed_sim: int, noisiness: float, BATCH_SIZE: int, interface: bool,
        #     #  unsatationary_bit_len: int, hop: int, group: int, puf: str) -> dict:
        interface_permutation.run(args.n, args.k, args.N, args.seed, args.noise, args.batch_size, args.interface,
                                  args.unsatationary_bit_len, args.hop,
                                  args.group, args.puf)


def parse_arguments(parser):
    parser.add_argument('--plus', dest='plus', action='store_true', help='Use this option for plus interface')
    parser.set_defaults(plus=False)
    parser.add_argument('--minus', dest='minus', action='store_true', help='Use this option for minus interface')
    parser.set_defaults(minus=False)
    parser.add_argument('--permutation', dest='permutation', action='store_true',
                        help='Use this option for permutation interface')
    parser.set_defaults(permutation=False)

    parser.add_argument('--interface', type=bool, default=1,
                        help='Use this option for applying interface')

    parser.add_argument('--seed', type=int, default=0,
                        help='The seed for generating PUF instance')

    parser.add_argument('--batch_size', type=int, default=100000,
                        help='The batch size for training')

    parser.add_argument('--max_epoch', type=int, default=500,
                        help='The max epoch for training')

    parser.add_argument('--patience', type=int, default=5,
                        help='The patience for training')

    parser.add_argument('--n', type=int, default=64,
                        help='The number of stages of the generated PUF')

    parser.add_argument('--k', type=int, default=1,
                        help='The number of challenges of the generated PUF')

    parser.add_argument('--N', type=int, default=1000000,
                        help='The number of generated CPR for training')

    parser.add_argument('--noise', type=float, default=0.00,
                        help='The noise level of generated PUF')

    parser.add_argument('--ghost_bit_len', type=int, default=1,
                        help='The number of ghost bit for plus interface')

    parser.add_argument('--double_use_bit_len', type=int, default=1,
                        help='The number of ghost bit for minus interface')

    parser.add_argument('--unsatationary_bit_len', type=int, default=3,
                        help='The number of ghost bit for permutation interface')

    parser.add_argument('--group', type=int, default=0,
                        help='The number of groups for applied interface')

    parser.add_argument('--hop', type=int, default=2,
                        help='The hop setting for permutation interface')

    parser.add_argument('--puf', type=str, default='apuf',
                        help='Use this option to choose PUF type')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='puf_interface')
    args = parse_arguments(parser)
    main(args)
