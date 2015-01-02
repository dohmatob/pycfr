from copy import deepcopy
from functools import partial
from itertools import combinations, permutations, product
from collections import Counter
import numpy as np
from _mathutils import choose, numeroter
from pokereval.hand_evaluator import HandEvaluator

# Possible player actions. Note that a check is equivalent to calling 0
# (a "free call")
FOLD = 0
CALL = 1
RAISE = 2


def overlap(t1, t2):
    """Helper function to check whether two iterables t1 and t2 overlap."""
    for x in t1:
        if x in t2:
            return True
    return False


def all_unique(hc):
    """Helper function that the elements of hc are pairwise non-overlapping."""
    for i in range(len(hc) - 1):
        for j in range(i + 1, len(hc)):
            if overlap(hc[i], hc[j]):
                return False
    return True


def default_infoset_format(_, holecards, board, bet_history):
    """
    Parameters
    ----------
    bet_history : string
        Sequences of bets observed in the game upto and including the
        current node. Each round begins with "/" character. The other
        possible characters are:
        - "f" : fold
        - "r" : raise
        - "k" : call
        - "c" : check
    """
    return "{0}{1}:{2}:".format("".join([str(x) for x in holecards]),
                                "".join([str(x) for x in board]), bet_history)


class GameRules(object):
    """Game rules of a finte multi-round game.

    Parameters
    ----------
    players : int
        Number of players.

    rounds: list of `RoundInfo` object
        Each item specifies the rules for the given round.

    deck : list of `Card` objects
        ...

    """

    def __init__(self, players, deck, rounds, ante, blinds,
                 handeval=HandEvaluator.evaluate_hand,
                 infoset_format=default_infoset_format):
        if players < 2:
            raise ValueError
        if ante < 0:
            raise ValueError
        if not rounds:
            raise ValueError
        if deck is None:
            raise ValueError
        if len(deck) < 2:
            raise ValueError
        if blinds is not None:
            if type(blinds) is int or type(blinds) is float:
                blinds = [blinds]
        for r in rounds:
            if len(r.maxbets) != players:
                raise ValueError
        self.players = players
        self.deck = deck
        self.roundinfo = rounds
        self.ante = ante
        self.blinds = blinds
        self.handeval = handeval
        self.infoset_format = infoset_format


class RoundInfo(object):
    """Rules (= game parameters) for a given round.

    Parameters
    ----------
    holecards : int
        Number of hole (i.e private) cards per player.

    boardcards: int
        Number of board (i.e community , i.e public) cards.

    betsize : int
        Starting bet size for round.

    maxbets : int
        Maximum bet size for round.

    """

    def __init__(self, holecards, boardcards, betsize, maxbets):
        self.holecards = holecards
        self.boardcards = boardcards
        self.betsize = betsize
        self.maxbets = maxbets


class GameTree(object):
    """Abstraction of game tree.

    Parameters
    ----------
    rules : `GameRules` object
        The rules of the game.

    Attributes
    ----------
    nodes : list of `Node` objects
        The nodes (decision and terminal nodes) of the game tree.

    information_sets : dict
        Each key is a string representing the corresponding information set.
        The correspoonding value is the list of nodes which belongs to this
        information set.
    """

    def __init__(self, rules):
        self.rules = deepcopy(rules)
        self.nodes = []
        self.leafs = []
        self.information_sets = {}
        self.root = None
        self.cur_node_id = numeroter(0)

    def build(self, with_sequence_form=False):
        print "Building game tree..."

        # Assume everyone is in
        players_in = [True] * self.rules.players
        # Collect antes
        committed = [self.rules.ante] * self.rules.players
        bets = [0] * self.rules.players
        # Collect blinds
        next_player = self.collect_blinds(committed, bets, 0)
        holes = [() for _ in xrange(self.rules.players)]
        board = ()
        bet_history = ""
        self.root = self.build_rounds(
            None, players_in, committed, holes, board, self.rules.deck,
            bet_history, 0, bets, next_player)

        if with_sequence_form:
            self.build_sequences()
            self.build_constraints()
            self.build_payoff_matrices()

    def collect_blinds(self, committed, bets, next_player):
        if self.rules.blinds is not None:
            for blind in self.rules.blinds:
                committed[next_player] += blind
                bets[next_player] = int(
                    (committed[next_player] - self.rules.ante)
                    / self.rules.roundinfo[0].betsize)
                next_player = (next_player + 1) % self.rules.players
        return next_player

    def deal_holecards(self, deck, holecards, players):
        """Deals hole (i.e private) cards from given deck.

        Returns
        -------
        holecards : generator of `Card` objects
            All possible hole-card deals.

        proba : float
            Each tuple of hole cards is drawn with this probability.
        """
        cards = permutations(combinations(deck, holecards), players)
        nchoices = choose(len(deck), holecards)

        # XXX use numpy to execute the following computation
        proba = 1.
        low, high = nchoices - players + 1, nchoices
        for salt in xrange(low, high + 1):
            proba *= salt
        proba = 1 / proba

        return cards, proba

    def commit_node(self, node):
        """stores a given node in game tree database."""
        self.nodes.append(node)

        # place node into appropriate information set
        if hasattr(node, "player_view"):
            if node.player_view not in self.information_sets:
                self.information_sets[node.player_view] = []
            self.information_sets[node.player_view].append(node)

        # handle leaf node
        if isinstance(node, TerminalNode):
            self.leafs.append(node)

        return self

    def build_rounds(self, root, players_in, committed, holes, board, deck,
                     bet_history, round_idx, bets=None, next_player=0):
        """Recursively build the rounds of the game hand."""
        # if no rounds left, then end the hand
        if round_idx == len(self.rules.roundinfo):
            tnode = self.showdown(root, players_in, committed, holes, board,
                                  deck, bet_history)
            self.commit_node(tnode)
            return

        bet_history += "/"
        cur_round = self.rules.roundinfo[round_idx]
        while not players_in[next_player]:
            next_player = (next_player + 1) % self.rules.players
        if bets is None:
            bets = [0] * self.rules.players
        min_actions_this_round = players_in.count(True)
        actions_this_round = 0
        if cur_round.holecards:
            return self.build_holecards(
                root, next_player, players_in, committed, holes, board, deck,
                bet_history, round_idx, min_actions_this_round,
                actions_this_round, bets)
        if cur_round.boardcards:
            return self.build_boardcards(
                root, next_player, players_in, committed, holes, board, deck,
                bet_history, round_idx, min_actions_this_round,
                actions_this_round, bets)
        return self.build_bets(
            root, next_player, players_in, committed, holes, board, deck,
            bet_history, round_idx, min_actions_this_round,
            actions_this_round, bets)

    def get_next_player(self, cur_player, players_in):
        """The player to act at the corrent node"""
        next_player = (cur_player + 1) % self.rules.players
        while not players_in[next_player]:
            next_player = (next_player + 1) % self.rules.players
        return next_player

    def build_holecards(self, root, next_player, players_in, committed, holes,
                        board, deck, bet_history, round_idx,
                        min_actions_this_round, actions_this_round, bets):
        cur_round = self.rules.roundinfo[round_idx]

        # Deal holecards
        if not cur_round.holecards:
            raise RuntimeError
        all_hc, proba = self.deal_holecards(
            deck, cur_round.holecards, players_in.count(True))
        hnode = HolecardChanceNode(
            root, committed, holes, board, self.rules.deck, "",
            cur_round.holecards, name=next(self.cur_node_id),
            proba=proba)
        self.commit_node(hnode)

        # Create a child node for every possible distribution
        for cur_holes in all_hc:
            dealt_cards = ()
            cur_holes = list(cur_holes)
            cur_idx = 0
            for i, hc in enumerate(holes):
                # Only deal cards to players who are still in
                if players_in[i]:
                    cur_holes[cur_idx] = hc + cur_holes[cur_idx]
                    cur_idx += 1
            for hc in cur_holes:
                dealt_cards += hc
            cur_deck = filter(lambda x: not (x in dealt_cards), deck)
            if cur_round.boardcards:
                self.build_boardcards(
                    hnode, next_player, players_in, committed, cur_holes,
                    board, cur_deck, bet_history, round_idx,
                    min_actions_this_round, actions_this_round, bets)
            else:
                self.build_bets(
                    hnode, next_player, players_in, committed, cur_holes,
                    board, cur_deck, bet_history, round_idx,
                    min_actions_this_round, actions_this_round, bets)
        return hnode

    def build_boardcards(self, root, next_player, players_in, committed,
                         holes, board, deck, bet_history, round_idx,
                         min_actions_this_round, actions_this_round, bets):
        cur_round = self.rules.roundinfo[round_idx]

        # reveal community cards
        if not cur_round.boardcards:
            raise RuntimeError
        all_bc = combinations(deck, cur_round.boardcards)
        proba = 1. / choose(len(deck), cur_round.boardcards)
        bnode = BoardcardChanceNode(
            root, committed, holes, board, deck, bet_history,
            cur_round.boardcards, name=next(self.cur_node_id), proba=proba)
        self.commit_node(bnode)

        for bc in all_bc:
            cur_board = board + bc
            cur_deck = filter(lambda x: not (x in bc), deck)
            self.build_bets(
                bnode, next_player, players_in, committed, holes, cur_board,
                cur_deck, bet_history, round_idx, min_actions_this_round,
                actions_this_round, bets)
        return bnode

    def build_bets(self, root, next_player, players_in, committed, holes,
                   board, deck, bet_history, round_idx, min_actions_this_round,
                   actions_this_round, bets_this_round):
        # if everyone else folded, end the hand
        if players_in.count(True) == 1:
            tnode = self.showdown(
                root, players_in, committed, holes, board, deck, bet_history)
            self.commit_node(tnode)
            return

        # if everyone checked or the last raisor has been called, end the round
        if actions_this_round >= min_actions_this_round and \
           self.all_called_last_raisor_or_folded(players_in, bets_this_round):
            self.build_rounds(
                root, players_in, committed, holes, board, deck, bet_history,
                round_idx + 1)
            return
        cur_round = self.rules.roundinfo[round_idx]
        anode = ActionNode(root, committed, holes, board, deck,
                           bet_history, next_player, self.rules.infoset_format,
                           name=next(self.cur_node_id))

        # add the node to the information set
        self.commit_node(anode)

        # get the next player to act
        next_player = self.get_next_player(next_player, players_in)

        # add a folding option if someone has bet more than this player
        if committed[anode.player] < max(committed):
            self.add_fold_child(
                anode, next_player, players_in, committed, holes, board, deck,
                bet_history, round_idx, min_actions_this_round,
                actions_this_round, bets_this_round)

        # add a calling/checking option
        self.add_call_child(
            anode, next_player, players_in, committed, holes, board, deck,
            bet_history, round_idx, min_actions_this_round, actions_this_round,
            bets_this_round)

        # add a raising option if this player has not reached their max bet
        # level
        if cur_round.maxbets[anode.player] > max(bets_this_round):
            self.add_raise_child(
                anode, next_player, players_in, committed, holes, board, deck,
                bet_history, round_idx, min_actions_this_round,
                actions_this_round, bets_this_round)
        return anode

    def all_called_last_raisor_or_folded(self, players_in, bets):
        betlevel = max(bets)
        for i, _ in enumerate(bets):
            if players_in[i] and bets[i] < betlevel:
                return False
        return True

    def add_fold_child(self, root, next_player, players_in, committed, holes,
                       board, deck, bet_history, round_idx,
                       min_actions_this_round, actions_this_round,
                       bets_this_round):
        players_in[root.player] = False
        bet_history += 'f'  # '.f%i' % root.player
        self.build_bets(
            root, next_player, players_in, committed, holes, board, deck,
            bet_history, round_idx, min_actions_this_round,
            actions_this_round + 1, bets_this_round)
        root.fold_action = root.children[-1]
        players_in[root.player] = True

    def add_call_child(self, root, next_player, players_in, committed, holes,
                       board, deck, bet_history, round_idx,
                       min_actions_this_round, actions_this_round,
                       bets_this_round):
        player_commit = committed[root.player]
        player_bets = bets_this_round[root.player]
        committed[root.player] = max(committed)
        amnt = max(bets_this_round)
        bets_this_round[root.player] = amnt
        marker = "k" if amnt else "c"
        bet_history += marker  # '.%s%i' % (marker, root.player)
        self.build_bets(
            root, next_player, players_in, committed, holes, board, deck,
            bet_history, round_idx, min_actions_this_round,
            actions_this_round + 1, bets_this_round)
        root.call_action = root.children[-1]
        committed[root.player] = player_commit
        bets_this_round[root.player] = player_bets

    def add_raise_child(self, root, next_player, players_in, committed, holes,
                        board, deck, bet_history, round_idx,
                        min_actions_this_round, actions_this_round,
                        bets_this_round):
        cur_round = self.rules.roundinfo[round_idx]
        prev_betlevel = bets_this_round[root.player]
        prev_commit = committed[root.player]
        bets_this_round[root.player] = max(bets_this_round) + 1
        committed[root.player] += (
            bets_this_round[root.player] - prev_betlevel) * cur_round.betsize
        bet_history += 'r'  # '.r%i' % root.player
        self.build_bets(
            root, next_player, players_in, committed, holes, board, deck,
            bet_history, round_idx, min_actions_this_round,
            actions_this_round + 1, bets_this_round)
        root.raise_action = root.children[-1]
        bets_this_round[root.player] = prev_betlevel
        committed[root.player] = prev_commit

    def showdown(self, root, players_in, committed, holes, board, deck,
                 bet_history):
        if players_in.count(True) == 1:
            winners = [i for i, v in enumerate(players_in) if v]
        else:
            scores = [self.rules.handeval(hc, board) for hc in holes]
            winners = []
            maxscore = -1
            for i, s in enumerate(scores):
                if players_in[i]:
                    if len(winners) == 0 or s > maxscore:
                        maxscore = s
                        winners = [i]
                    elif s == maxscore:
                        winners.append(i)
        pot = sum(committed)
        payoff = pot / float(len(winners))
        payoffs = [-x for x in committed]
        for w in winners:
            payoffs[w] += payoff
        return TerminalNode(root, committed, holes, board, deck, bet_history,
                            payoffs, players_in, name=next(self.cur_node_id))

    def holecard_distributions(self):
        x = Counter(combinations(self.rules.deck, self.holecards))
        d = float(sum(x.values()))
        return zip(x.keys(), [y / d for y in x.values()])

    def get_player_information_sets(self, player):
        """Returns all information sets belonging to given player."""
        return dict((k, v) for k, v in self.information_sets.iteritems()
                    if v[0].player == player)

    def last_node_played(self, path, player):
        """Returns the last node played by given player along given path."""
        while path.parent:
            if isinstance(path.parent, ActionNode):
                if path.parent.player == player:
                    return path
            path = path.parent
        return None

    def chop(self, node, player):
        """Returns list of all the nodes played by player along given path."""
        buf = ()
        while node.parent:
            if isinstance(node.parent, ActionNode):
                if node.parent.player == player:
                    buf += (node,)
            node = node.parent
        return buf[::-1]

    def node2seq(self, node):
        """Returns sequence of information-set-relabelled moves made
        by previous player along this path."""
        moves = []
        if hasattr(node, "author"):
            pieces = self.chop(node, node.author)
            for anc in pieces:
                moves.append((anc.parent.player_view, anc.bet_history[-1]))
        return moves

    def build_sequences(self):
        """Each sequence for a player is of the form (i_1, a_1)(i_2,, a_2)...,
        where each a_j is an action at the information set identified with i_j
        """
        print "Bulding players' move sequences..."
        self.sequences = {}
        for player in xrange(self.rules.players):
            self.sequences[player] = [[]]
        for node in self.nodes:
            if node.parent is None:
                continue
            if not hasattr(node, "author"):
                continue
            prev = node.author
            seq = self.node2seq(node)
            if seq not in self.sequences[prev]:
                self.sequences[prev].append(seq)
        for sequences in self.sequences.itervalues():
            sequences.sort()
        return self.sequences

    def build_constraints(self):
        """Generates matrices for the equality constraints on each player's
        admissible realization plans.

        The constraints for player p are a pair E_p, e_p corresponding to a
        set of linear constraints read as "E_p x = e_p". E_p has as many
        columns as player p has sequences, and as many rows as there
        information sets for player p, plus 1.
        """
        print "Building the constraints on players' strategy profiles..."
        self.constraints = {}

        # loop over players
        for player in self.sequences.keys():
            row = np.zeros(len(self.sequences[player]))
            row[0] = 1.
            E = [row]

            # loop over sequences for player
            for i, sigma in enumerate(self.sequences[player]):
                mem = {}
                # loop over all sequences which are extensions of tau by
                # a single move
                for j, tau in enumerate(self.sequences[player]):
                    if tau and tau[:-1] == sigma:
                        # sigma is the (unique) antecedant of tau
                        h, _ = tau[-1]
                        if h not in mem:
                            mem[h] = []
                        mem[h].append(j)
                # fill row: linear constraint (corresponds to Bayes rule)
                for where in mem.values():
                    row = np.zeros(len(self.sequences[player]))
                    row[i] = -1.
                    row[where] = 1.
                    E.append(row)
            # compute right handside (e) of constraints "Ex = e"
            e = np.zeros(len(self.get_player_information_sets(player)) + 1)
            e[0] = 1.
            self.constraints[player] = np.array(E), e
        return self.constraints

    def build_payoff_matrices(self):
        """Builds payoff matrices for multi-matrix game.

        Returns
        -------
        self.payoff_matrices : list of arrays of shape (n_1, ..., n_#players),
        where n_p is the number of sequences for player p

        References
        ----------
        [1] Benhard von Stengel, "Efficient Computation of Behavior Strategies"
        http://www.maths.lse.ac.uk/personal/stengel/TEXTE/geb1996a.pdf

        TODO
        ----
        For zero-sum heads-up games, Use tricks in equation (38) of
        "Smoothing Techniques for Computing Nash Equilibria of Sequential
        Games" http://repository.cmu.edu/cgi/viewcontent.cgi?article=
        2442&context=compsci to compute the payoff matrix as a block diagonal
        matrix whose blocks are sums of Kronecker products of sparse
        matrices. This can be done by appropriately permuting the list of
        sequences of each (non-chance) player.
        """
        print "Building payoff matrices..."
        self.payoff_matrices = [np.zeros(map(len, self.sequences.values()))
                                for _ in self.sequences]
        for leaf in self.leafs:
            idx = ()
            for player in xrange(self.rules.players):
                idx += (self.sequences[player].index(self.node2seq(
                        self.last_node_played(leaf, player))),)
            for player in xrange(self.rules.players):
                # Compute expected payoff vector when each player plays the
                # sequence indicated by its index in idx. Refer to equation
                # (3.7) on page 228 of of [1]
                self.payoff_matrices[player][idx] += leaf.payoffs[
                    player] * leaf.proba_
        return self.payoff_matrices


def multi_infoset_format(base_infoset_format, player, holecards, board,
                         bet_history):
    return tuple([base_infoset_format(player, hc, board, bet_history)
                  for hc in holecards])


class PublicTree(GameTree):
    # XXX This sub-class should disappea! It's is code duplication!
    def __init__(self, rules):
        GameTree.__init__(
            self, GameRules(
                rules.players, rules.deck, rules.roundinfo, rules.ante,
                rules.blinds, rules.handeval, partial(multi_infoset_format,
                                                      rules.infoset_format)))

    def build(self):
        # Assume everyone is in
        players_in = [True] * self.rules.players
        # Collect antes
        committed = [self.rules.ante] * self.rules.players
        bets = [0] * self.rules.players
        # Collect blinds
        next_player = self.collect_blinds(committed, bets, 0)
        holes = [[()]] * self.rules.players
        board = ()
        bet_history = ""
        self.root = self.build_rounds(
            None, players_in, committed, holes, board, self.rules.deck,
            bet_history, 0, bets, next_player)

    def build_holecards(self, root, next_player, players_in, committed, holes,
                        board, deck, bet_history, round_idx,
                        min_actions_this_round, actions_this_round, bets):
        cur_round = self.rules.roundinfo[round_idx]
        hnode = HolecardChanceNode(
            root, committed, holes, board, self.rules.deck, "",
            cur_round.holecards, name=next(self.cur_node_id))
        # Deal holecards
        all_hc = list(combinations(deck, cur_round.holecards))
        updated_holes = []
        for player in range(self.rules.players):
            if not players_in[player]:
                # Only deal to players who are still in the hand
                updated_holes.append([old_hc for old_hc in holes[player]])
            elif len(holes[player]) == 0:
                # If this player has no cards, just set their holecards to be
                # the newly dealt ones
                updated_holes.append([new_hc for new_hc in all_hc])
            else:
                updated_holes.append([])
                # Filter holecards to valid combinations
                # TODO: Speed this up by removing duplicate holecard
                # combinations
                for new_hc in all_hc:
                    for old_hc in holes[player]:
                        if not overlap(old_hc, new_hc):
                            updated_holes[player].append(old_hc + new_hc)
        if cur_round.boardcards:
            self.build_boardcards(
                hnode, next_player, players_in, committed, updated_holes,
                board, deck, bet_history, round_idx, min_actions_this_round,
                actions_this_round, bets)
        else:
            self.build_bets(
                hnode, next_player, players_in, committed, updated_holes,
                board, deck, bet_history, round_idx, min_actions_this_round,
                actions_this_round, bets)
        return hnode

    def build_boardcards(self, root, next_player, players_in, committed, holes,
                         board, deck, bet_history, round_idx,
                         min_actions_this_round, actions_this_round, bets):
        cur_round = self.rules.roundinfo[round_idx]
        bnode = BoardcardChanceNode(
            root, committed, holes, board, deck, bet_history,
            cur_round.boardcards, name=next(self.cur_node_id))
        all_bc = combinations(deck, cur_round.boardcards)
        for bc in all_bc:
            cur_board = board + bc
            cur_deck = filter(lambda x: not (x in bc), deck)
            updated_holes = []
            # Filter any holecards that are now impossible
            for player in range(self.rules.players):
                updated_holes.append([])
                for hc in holes[player]:
                    if not overlap(hc, bc):
                        updated_holes[player].append(hc)
            self.build_bets(
                bnode, next_player, players_in, committed, updated_holes,
                cur_board, cur_deck, bet_history, round_idx,
                min_actions_this_round, actions_this_round, bets)
        return bnode

    def showdown(self, root, players_in, committed, holes, board, deck,
                 bet_history):
        # TODO: Speedup
        # - Pre-order list of hands
        pot = sum(committed)
        showdowns_possible = self.showdown_combinations(holes)
        if players_in.count(True) == 1:
            fold_payoffs = [-x for x in committed]
            fold_payoffs[players_in.index(True)] += pot
            payoffs = {hands: fold_payoffs for hands in showdowns_possible}
        else:
            scores = {}
            for i in range(self.rules.players):
                if players_in[i]:
                    for hc in holes[i]:
                        if not (hc in scores):
                            scores[hc] = self.rules.handeval(hc, board)
            payoffs = {hands: self.calc_payoffs(hands, scores, players_in,
                                                committed, pot)
                       for hands in showdowns_possible}
        return TerminalNode(root, committed, holes, board, deck, bet_history,
                            payoffs, players_in, name=next(self.cur_node_id))

    def showdown_combinations(self, holes):
        # Get all the possible holecard matchups for a given showdown.
        # Every card must be unique because two players cannot have the same
        # holecard.
        return list(filter(lambda x: all_unique(x), product(*holes)))

    def calc_payoffs(self, hands, scores, players_in, committed, pot):
        winners = []
        maxscore = -1
        for i, hand in enumerate(hands):
            if players_in[i]:
                s = scores[hand]
                if len(winners) == 0 or s > maxscore:
                    maxscore = s
                    winners = [i]
                elif s == maxscore:
                    winners.append(i)
        payoff = pot / float(len(winners))
        payoffs = [-x for x in committed]
        for w in winners:
            payoffs[w] += payoff
        return payoffs


class Node(object):
    """Abstraction of game tree node.

    Parameters
    ----------
    bet_history : string
        Sequences of bets observed in the game upto and including the
        current node. Each round begins with "/" character. The other
        possible characters are:
        - "f" : fold
        - "r" : raise
        - "k" : call
        - "c" : check

    proba : float in the interval [0, 1]
        Probability with which this node is forked from its parent.

    Attributes
    ----------
    proba_ : float in the interval [0, 1]
        The probability (induced by the chance player) of the path from
        the root node to this node.
    """
    def __init__(self, parent, committed, holecards, board, deck, bet_history,
                 name=None, proba=1.):
        self.committed = deepcopy(committed)
        self.holecards = deepcopy(holecards)
        self.board = deepcopy(board)
        self.deck = deepcopy(deck)
        self.bet_history = deepcopy(bet_history)
        self.name = name
        self.proba = self.proba_ = proba
        if parent:
            self.parent = parent
            self.parent.add_child(self)
            self.proba_ *= parent.proba_
        else:
            self.parent = None

    def add_child(self, child):
        if isinstance(self, ActionNode):
            child.author = self.player
        if self.children is None:
            self.children = [child]
        else:
            self.children.append(child)

    def __repr__(self):
        """For pretty-printing the node."""
        tokenize = lambda stuff: ('"%s"' % stuff) if isinstance(
            stuff, basestring) else stuff
        return "%s(" % (self.__class__.__name__) + ", ".join(
            ["%s=%s" % (k, tokenize(getattr(self, k)))
             for k in ["bet_history", "deck", "holecards", "player", "author",
                       "player_view", "committed", "proba_", "payoffs"]
        if hasattr(self, k)]) + ")"


class TerminalNode(Node):
    def __init__(self, parent, committed, holecards, board, deck, bet_history,
                 payoffs, players_in, **kwargs):
        Node.__init__(self, parent, committed,
                      holecards, board, deck, bet_history, **kwargs)
        self.payoffs = payoffs
        self.players_in = deepcopy(players_in)


class HolecardChanceNode(Node):
    def __init__(self, parent, committed, holecards, board, deck, bet_history,
                 todeal, **kwargs):
        Node.__init__(self, parent, committed,
                      holecards, board, deck, bet_history, **kwargs)
        self.todeal = todeal
        self.children = []


class BoardcardChanceNode(Node):
    def __init__(self, parent, committed, holecards, board, deck, bet_history,
                 todeal, **kwargs):
        Node.__init__(self, parent, committed,
                      holecards, board, deck, bet_history, **kwargs)
        self.todeal = todeal
        self.children = []


class ActionNode(Node):
    def __init__(self, parent, committed, holecards, board, deck, bet_history,
                 player, infoset_format, **kwargs):
        Node.__init__(self, parent, committed,
                      holecards, board, deck, bet_history, **kwargs)
        self.player = player
        self.children = []
        self.raise_action = None
        self.call_action = None
        self.fold_action = None
        self.player_view = infoset_format(
            player, holecards[player], board, bet_history)

    def valid(self, action):
        if action == FOLD:
            return self.fold_action
        if action == CALL:
            return self.call_action
        if action == RAISE:
            return self.raise_action
        raise Exception(
            "Unknown action {0}. Action must be FOLD, CALL, or RAISE".format(
                action))

    def get_child(self, action):
        return self.valid(action)
