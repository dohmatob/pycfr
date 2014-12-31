"""
Building game tree and sequence-form representation for Leduc poker.
"""

from pycfr.pokergames import leduc_gametree
import matplotlib.pyplot as plt

# buiild game tree, and sequence-form representation thereof
leduc = leduc_gametree(with_sequence_form=True)

plt.figure()
plt.gray()
for player, (E, _) in leduc.constraints.iteritems():
    # show constraints on each player's complex
    ax = plt.subplot2grid((2, 2), (0, player))
    ax.matshow(E)
    plt.axis('off')
    plt.title(["$E$", "$F$"][player])

    # show payoff matrix
    ax = plt.subplot2grid((2, 2), (1, player))
    ax.matshow(leduc.payoff_matrices[player])
    plt.axis('off')
    plt.title(["$A$", "$B$"][player])

plt.show()
