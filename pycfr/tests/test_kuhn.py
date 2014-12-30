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