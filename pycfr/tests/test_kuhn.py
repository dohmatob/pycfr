from nose.tools import assert_equal
from ..pokergames import kuhn_gametree


def test_infosets():
    # It's known that each non-chance player in Kuhn's poker has exactly 6
    # information sets, each containing 2 nodes
    kuhn_gt = kuhn_gametree()
    for player in xrange(2):
        infosets = kuhn_gt.get_player_information_sets(player)
        assert_equal(len(infosets), 6)
        for infoset in infosets.itervalues():
            assert_equal(len(infoset), 2)


def test_node_count():
    # It's known that the game tree for Kuhn Poker has 55 nodes, of which
    # 30 are leafs
    kuhn_gt = kuhn_gametree()
    assert_equal(len(kuhn_gt.nodes), 55)
    assert_equal(len(kuhn_gt.leafs), 30)


def test_payoffs():
    kuhn_gt = kuhn_gametree()
    assert_equal(len(kuhn_gt.nodes), 55)
    assert_equal(len(kuhn_gt.leafs), 30)
    payoffs = []
    for node in kuhn_gt.nodes:
        if "Terminal" in node.__class__.__name__:
            payoffs.append(node.payoffs[0])
    payoffs = sorted(payoffs)
    assert_equal(payoffs, [-2, -2, -2, -2, -2, -2, -1, -1, -1, -1, -1, -1,
                           -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2,
                           2, 2, 2])
