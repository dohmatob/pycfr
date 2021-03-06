from nose.tools import assert_equal
from ..pokertrees import (GameTree, PublicTree, GameRules, RoundInfo,
                          HolecardChanceNode, BoardcardChanceNode,
                          ActionNode, TerminalNode)
from ..card import Card
from ..pokergames import leduc_eval, kuhn_eval, leduc_format


rules = GameRules(players=2, deck=[Card(14, 1), Card(13, 2), Card(13, 1),
                                   Card(12, 1)],
                  rounds=[RoundInfo(holecards=1, boardcards=0, betsize=1,
                                    maxbets=[2, 2]),
                          RoundInfo(holecards=0, boardcards=1, betsize=2,
                                    maxbets=[2, 2])], ante=1, blinds=[1, 2],
                  handeval=leduc_eval)


def test_gametree():
    tree = GameTree(rules)
    tree.build()
    assert(type(tree.root) == HolecardChanceNode)
    assert(len(tree.root.children) == 12)
    assert(type(tree.root.children[0]) == ActionNode)
    assert(tree.root.children[0].player == 0)
    assert(len(tree.root.children[0].children) == 2)
    assert(tree.root.children[0].player_view == "As:/:")
    # /f
    assert(type(tree.root.children[0].children[0]) == TerminalNode)
    assert(tree.root.children[0].children[0].payoffs == [-2, 2])
    assert(tree.root.children[0].children[0].bet_history == '/f')
    # /c
    assert(type(tree.root.children[0].children[1]) == ActionNode)
    assert(tree.root.children[0].children[1].bet_history == '/k')
    assert(len(tree.root.children[0].children[1].children) == 1)
    assert(tree.root.children[0].children[1].player == 1)
    assert(tree.root.children[0].children[1].fold_action is None)
    assert(tree.root.children[0].children[1].call_action != None)
    assert(tree.root.children[0].children[1].raise_action is None)
    assert(tree.root.children[0].children[1].player_view == "Kh:/k:")
    # /cc/ [boardcard]
    assert(
        type(tree.root.children[0].children[1].children[0]) == BoardcardChanceNode)
    assert(tree.root.children[0].children[1].children[0].bet_history == '/kk/')
    assert(len(tree.root.children[0].children[1].children[0].children) == 2)
    # /cc/ [action]
    assert(
        type(tree.root.children[0].children[1].children[0].children[0]) == ActionNode)
    assert(tree.root.children[0].children[
           1].children[0].children[0].bet_history == '/kk/')
    assert(
        len(tree.root.children[0].children[1].children[0].children[0].children) == 2)
    assert(tree.root.children[0].children[1].children[0].children[0].player == 0)
    assert(tree.root.children[0].children[
           1].children[0].children[0].fold_action is None)
    assert(tree.root.children[0].children[
           1].children[0].children[0].call_action != None)
    assert(tree.root.children[0].children[
           1].children[0].children[0].raise_action != None)
    assert(tree.root.children[0].children[1].children[
           0].children[0].player_view == 'AsKs:/kk/:')
    # /cc/r
    assert(
        type(tree.root.children[0].children[1].children[0].children[0].children[1]) == ActionNode)
    assert(tree.root.children[0].children[1].children[
           0].children[0].children[1].bet_history == '/kk/r')
    assert(
        len(tree.root.children[0].children[1].children[0].children[0].children[1].children) == 3)
    assert(tree.root.children[0].children[1].children[
           0].children[0].children[1].player == 1)
    assert(tree.root.children[0].children[1].children[
           0].children[0].children[1].fold_action != None)
    assert(tree.root.children[0].children[1].children[
           0].children[0].children[1].call_action != None)
    assert(tree.root.children[0].children[1].children[
           0].children[0].children[1].raise_action != None)
    assert(tree.root.children[0].children[1].children[
           0].children[0].children[1].player_view == 'KhKs:/kk/r:')
    # /cc/c
    assert(
        type(tree.root.children[0].children[1].children[0].children[0].children[0]) == ActionNode)
    assert(tree.root.children[0].children[1].children[
           0].children[0].children[0].bet_history == '/kk/c')
    assert(
        len(tree.root.children[0].children[1].children[0].children[0].children[0].children) == 2)
    assert(tree.root.children[0].children[1].children[
           0].children[0].children[0].player == 1)
    assert(tree.root.children[0].children[1].children[
           0].children[0].children[0].fold_action is None)
    assert(tree.root.children[0].children[1].children[
           0].children[0].children[0].call_action != None)
    assert(tree.root.children[0].children[1].children[
           0].children[0].children[0].raise_action != None)
    assert(tree.root.children[0].children[1].children[
           0].children[0].children[0].player_view == 'KhKs:/kk/c:')
    # /cc/cc
    assert(type(tree.root.children[0].children[1].children[
           0].children[0].children[0].children[0]) == TerminalNode)
    assert(tree.root.children[0].children[1].children[0].children[
           0].children[0].children[0].bet_history == '/kk/cc')
    assert(tree.root.children[0].children[1].children[
           0].children[0].children[0].children[0].payoffs == [-3, 3])
    # /cc/cr
    assert(type(tree.root.children[0].children[1].children[
           0].children[0].children[0].children[1]) == ActionNode)
    assert(tree.root.children[0].children[1].children[0].children[
           0].children[0].children[1].bet_history == '/kk/cr')
    assert(len(tree.root.children[0].children[1].children[
           0].children[0].children[0].children[1].children) == 3)
    assert(tree.root.children[0].children[1].children[
           0].children[0].children[0].children[1].player == 0)
    assert(tree.root.children[0].children[1].children[
           0].children[0].children[0].children[1].fold_action != None)
    assert(tree.root.children[0].children[1].children[
           0].children[0].children[0].children[1].call_action != None)
    assert(tree.root.children[0].children[1].children[
           0].children[0].children[0].children[1].raise_action != None)
    assert(tree.root.children[0].children[1].children[0].children[
           0].children[0].children[1].player_view == 'AsKs:/kk/cr:')
    # /cc/crr
    assert(type(tree.root.children[0].children[1].children[
           0].children[0].children[0].children[1].children[2]) == ActionNode)
    assert(tree.root.children[0].children[1].children[0].children[
           0].children[0].children[1].children[2].bet_history == '/kk/crr')
    assert(len(tree.root.children[0].children[1].children[
           0].children[0].children[0].children[1].children[2].children) == 2)
    assert(tree.root.children[0].children[1].children[0].children[
           0].children[0].children[1].children[2].player == 1)
    assert(tree.root.children[0].children[1].children[0].children[
           0].children[0].children[1].children[2].fold_action != None)
    assert(tree.root.children[0].children[1].children[0].children[
           0].children[0].children[1].children[2].call_action != None)
    assert(tree.root.children[0].children[1].children[0].children[
           0].children[0].children[1].children[2].raise_action is None)
    assert(tree.root.children[0].children[1].children[0].children[
           0].children[0].children[1].children[2].player_view == 'KhKs:/kk/crr:')
    # /cc/crrf
    assert(type(tree.root.children[0].children[1].children[0].children[
           0].children[0].children[1].children[2].children[0]) == TerminalNode)
    assert(tree.root.children[0].children[1].children[0].children[0].children[
           0].children[1].children[2].children[0].bet_history == '/kk/crrf')
    assert(tree.root.children[0].children[1].children[0].children[
           0].children[0].children[1].children[2].children[0].payoffs == [5, -5])
    # /cc/crrc
    assert(type(tree.root.children[0].children[1].children[0].children[
           0].children[0].children[1].children[2].children[1]) == TerminalNode)
    assert(tree.root.children[0].children[1].children[0].children[0].children[
           0].children[1].children[2].children[1].bet_history == '/kk/crrk')
    assert(tree.root.children[0].children[1].children[0].children[
           0].children[0].children[1].children[2].children[1].payoffs == [-7, 7])


def test_publictree():
    tree = PublicTree(rules)
    tree.build()

    assert(type(tree.root) == HolecardChanceNode)
    assert(len(tree.root.children) == 1)
    # /
    assert(type(tree.root.children[0]) == ActionNode)
    assert(tree.root.children[0].player == 0)
    assert(tree.root.children[0].player_view == (
        'As:/:', 'Kh:/:', 'Ks:/:', 'Qs:/:'))
    assert(len(tree.root.children[0].children) == 2)
    assert(tree.root.children[0].fold_action != None)
    assert(tree.root.children[0].call_action != None)
    assert(tree.root.children[0].raise_action == None)
    # /f
    assert(type(tree.root.children[0].children[0]) == TerminalNode)
    assert_equal(tree.root.children[0].children[0].payoffs, {((Card(14, 1),), (Card(13, 1),)): [-2, 2], ((Card(14, 1),), (Card(13, 2),)): [-2, 2], ((Card(14, 1),), (Card(12, 1),)): [-2, 2], ((Card(13, 1),), (Card(14, 1),)): [-2, 2], ((Card(13, 1),), (Card(13, 2),)): [-2, 2], (
        (Card(13, 1),), (Card(12, 1),)): [-2, 2], ((Card(13, 2),), (Card(14, 1),)): [-2, 2], ((Card(13, 2),), (Card(13, 1),)): [-2, 2], ((Card(13, 2),), (Card(12, 1),)): [-2, 2], ((Card(12, 1),), (Card(14, 1),)): [-2, 2], ((Card(12, 1),), (Card(13, 2),)): [-2, 2], ((Card(12, 1),), (Card(13, 1),)): [-2, 2]})
    # /c
    assert(type(tree.root.children[0].children[1]) == ActionNode)
    assert(tree.root.children[0].children[1].player == 1)
    assert(tree.root.children[0].children[1].player_view == (
        'As:/k:', 'Kh:/k:', 'Ks:/k:', 'Qs:/k:'))
    assert_equal(len(tree.root.children[0].children[1].children), 1)
    # /cc/ [boardcard]
    assert(
        type(tree.root.children[0].children[1].children[0]) == BoardcardChanceNode)
    assert(tree.root.children[0].children[1].children[0].bet_history == '/kk/')
    assert(len(tree.root.children[0].children[1].children[0].children) == 4)
    # xAs:/cc/ [action]
    assert(
        type(tree.root.children[0].children[1].children[0].children[0]) == ActionNode)
    assert(tree.root.children[0].children[
           1].children[0].children[0].bet_history == '/kk/')
    assert(
        len(tree.root.children[0].children[1].children[0].children[0].children) == 2)
    assert(tree.root.children[0].children[1].children[0].children[0].player == 0)
    assert(tree.root.children[0].children[
           1].children[0].children[0].fold_action is None)
    assert(tree.root.children[0].children[
           1].children[0].children[0].call_action != None)
    assert(tree.root.children[0].children[
           1].children[0].children[0].raise_action != None)
    assert(tree.root.children[0].children[1].children[0].children[
           0].player_view == ('KhAs:/kk/:', 'KsAs:/kk/:', 'QsAs:/kk/:'))
    # xAs:/cc/r
    assert(
        type(tree.root.children[0].children[1].children[0].children[0].children[1]) == ActionNode)
    assert(tree.root.children[0].children[1].children[
           0].children[0].children[1].bet_history == '/kk/r')
    assert(
        len(tree.root.children[0].children[1].children[0].children[0].children[1].children) == 3)
    assert(tree.root.children[0].children[1].children[
           0].children[0].children[1].player == 1)
    assert(tree.root.children[0].children[1].children[
           0].children[0].children[1].fold_action != None)
    assert(tree.root.children[0].children[1].children[
           0].children[0].children[1].call_action != None)
    assert(tree.root.children[0].children[1].children[
           0].children[0].children[1].raise_action != None)
    assert(tree.root.children[0].children[1].children[0].children[0].children[
           1].player_view == ('KhAs:/kk/r:', 'KsAs:/kk/r:', 'QsAs:/kk/r:'))
    # xAs:/cc/rc
    assert(type(tree.root.children[0].children[1].children[
           0].children[0].children[1].children[1]) == TerminalNode)
    assert(tree.root.children[0].children[1].children[0].children[
           0].children[1].children[1].bet_history == '/kk/rk')
    assert(tree.root.children[0].children[1].children[0].children[0].children[1].children[1].payoffs == {((Card(13, 1),), (Card(13, 2),)): [0, 0], ((Card(13, 1),), (Card(12, 1),)): [
           5, -5], ((Card(13, 2),), (Card(13, 1),)): [0, 0], ((Card(13, 2),), (Card(12, 1),)): [5, -5], ((Card(12, 1),), (Card(13, 2),)): [-5, 5], ((Card(12, 1),), (Card(13, 1),)): [-5, 5]})
    # xKh:/cc/rc
    assert(type(tree.root.children[0].children[1].children[
           0].children[1].children[1].children[1]) == TerminalNode)
    assert(tree.root.children[0].children[1].children[0].children[
           1].children[1].children[1].bet_history == '/kk/rk')
    assert(tree.root.children[0].children[1].children[0].children[1].children[1].children[1].payoffs == {((Card(13, 1),), (Card(14, 1),)): [5, -5], ((Card(13, 1),), (Card(12, 1),)): [
           5, -5], ((Card(14, 1),), (Card(13, 1),)): [-5, 5], ((Card(14, 1),), (Card(12, 1),)): [5, -5], ((Card(12, 1),), (Card(14, 1),)): [-5, 5], ((Card(12, 1),), (Card(13, 1),)): [-5, 5]})


def test_deal_holecards():
    players = 2
    deck = [Card(14, 1), Card(13, 1), Card(12, 1)]
    ante = 1
    blinds = None
    rounds = [RoundInfo(holecards=1, boardcards=0, betsize=1, maxbets=[1, 1])]
    rules = GameRules(players, deck, rounds, ante, blinds, handeval=kuhn_eval,
                     infoset_format=leduc_format)
    tree = GameTree(rules)
    holes, proba = tree.deal_holecards(tree.rules.deck,
                                       tree.rules.roundinfo[0].holecards,
                                       tree.rules.players)
    holes = list(holes)
    assert_equal(len(holes), 6)
    assert_equal(proba, 1. / len(holes))
