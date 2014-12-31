"""
Building game tree and sequence-form representation for Leduc and Kuhn pokers.
"""

from pycfr.pokergames import (leduc_gametree, kuhn_gametree,
                              half_street_kuhn_gametree)
import matplotlib.pyplot as plt

for name, maker in zip(["Kuhn", "Leduc", "Half-street Kuhn"],
                       [kuhn_gametree, leduc_gametree,
                        half_street_kuhn_gametree]):
    # build game tree, and sequence-form representation thereof
    gt = maker(with_sequence_form=True)
    print "=== %s Poker (%i nodes, %i leafs): ===" % (name, len(gt.nodes),
                                                      len(gt.leafs))
    plt.figure(figsize=(13, 7))
    plt.suptitle("Sequence-form representation of %s Poker" % name)
    plt.gray()
    for player, (E, _) in gt.constraints.iteritems():
        print "Player %i" % player
        print "\t# infosets : %i" % (
            len(gt.get_player_information_sets(player)))
        print "\t# sequences: %i" % len(gt.sequences[player])

        # show constraints on each player's complex
        ax = plt.subplot2grid((2, 2), (0, player))
        ax.matshow(E)
        plt.axis('off')
        plt.title(["$E$", "$F$"][player])

        # show payoff matrices
        ax = plt.subplot2grid((2, 2), (1, player))
        ax.matshow(gt.payoff_matrices[player])
        plt.axis('off')
        plt.title(["$A$", "$B$"][player])
    plt.tight_layout()
    print "_" * 80

plt.show()
