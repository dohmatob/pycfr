"""
Building game tree and sequence-form representation for Leduc and Kuhn pokers.
"""

from pycfr.pokergames import leduc_gametree, kuhn_gametree
import matplotlib.pyplot as plt

# build game tree, and sequence-form representation thereof
for name, maker in zip(["kuhn", "leduc"], [kuhn_gametree, leduc_gametree]):
    gt = maker(with_sequence_form=True)
    print "=== %s poker (%i nodes, %i leafs): ===" % (name, len(gt.nodes),
                                                      len(gt.leafs))
    plt.figure()
    plt.suptitle("%s poker" % name)
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

        # show payoff matrix
        ax = plt.subplot2grid((2, 2), (1, player))
        ax.matshow(gt.payoff_matrices[player])
        plt.axis('off')
        plt.title(["$A$", "$B$"][player])
    print "_" * 80

plt.show()
