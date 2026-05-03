
from shiny import ui, render, App, reactive
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
import random
from collections import Counter

############################
# --- Configuration ---
############################

N_CANDIDATES = 11
N_SEATS      = 4

CANDIDATE_POOL = [
    'Aaron Chapman', 'Armor Valor', 'Ashley Fehr', 
    'Azeem Ali', 'Caitlin Stockwell', 'Frances Bula',
    'Iona Bonamis', 'Jarrett Hagglund', 'Mike Tan', 
    'Peter Waldkirch', 'Russil Wvong'
]
candidates = CANDIDATE_POOL[:N_CANDIDATES]

def cand_id(c):
    """Convert candidate name to a valid Shiny input ID."""
    return 'weight_' + c.lower().replace(' ', '_').replace('-', '_')

############################
# --- STV Core Functions ---
############################

def get_active_candidate(state, eliminated, elected):
    for c in state['prefs']:
        if c not in eliminated and c not in elected:
            return c
    return None


def tally_votes(ballot_states, eliminated, elected):
    totals = Counter()
    for state in ballot_states:
        c = get_active_candidate(state, eliminated, elected)
        if c:
            totals[c] += state['weight']
    return totals


def run_election(ballots, my_ballot, n_seats, candidates):
    quota = int(len(ballots) / (n_seats + 1)) + 1

    prop_cycle  = plt.rcParams['axes.prop_cycle'].by_key()['color']
    cand_colour = {c: prop_cycle[i % len(prop_cycle)] for i, c in enumerate(candidates)}

    ballot_states = []
    for b in ballots:
        ballot_states.append({
            'prefs'  : list(b),
            'weight' : 1.0,
            'is_mine': (b is my_ballot),
        })

    elected    = set()
    eliminated = set()
    rounds     = []
    round_num  = 0

    while len(elected) < n_seats:
        round_num += 1
        totals = tally_votes(ballot_states, eliminated, elected)
        if not totals:
            break

        round_result = {
            'round'             : round_num,
            'totals'            : dict(totals),
            'events'            : [],
            'transfer_breakdown': {},
        }

        winners_this_round = [c for c, v in totals.items() if v >= quota]

        if winners_this_round:
            for winner in sorted(winners_this_round, key=lambda c: -totals[c]):
                if len(elected) < n_seats:
                    elected.add(winner)
                    round_result['events'].append(('elected', winner))

                    winner_votes   = totals[winner]
                    surplus        = winner_votes - quota
                    transfer_value = surplus / winner_votes

                    breakdown = {}
                    for state in ballot_states:
                        if get_active_candidate(state, eliminated, elected - {winner}) == winner:
                            dest = get_active_candidate(state, eliminated, elected)
                            dest = dest if dest else 'exhausted'
                            breakdown[dest] = breakdown.get(dest, 0) + state['weight'] * transfer_value
                            state['weight'] *= transfer_value

                    round_result['transfer_breakdown'][winner] = breakdown

                    if len(elected) < n_seats:
                        round_result['events'].append(('transfer', winner, transfer_value))
        else:
            loser = min(totals, key=totals.get)
            breakdown = {}
            for state in ballot_states:
                if get_active_candidate(state, eliminated, elected) == loser:
                    dest = next(
                        (c for c in state['prefs']
                         if c != loser and c not in eliminated and c not in elected),
                        'exhausted'
                    )
                    breakdown[dest] = breakdown.get(dest, 0) + state['weight']

            eliminated.add(loser)
            round_result['transfer_breakdown'][loser] = breakdown
            round_result['events'].append(('eliminated', loser))

        rounds.append(round_result)

    elected_totals = {
        e[1]: r['totals'][e[1]]
        for r in rounds
        for e in r['events']
        if e[0] == 'elected'
    }

    return rounds, quota, cand_colour, ballot_states, elected_totals


def generate_ballot(candidates, weights):
    weight_vals = [weights[c] for c in candidates]
    first = np.random.choice(candidates, p=weight_vals)
    rest  = [c for c in candidates if c != first]
    random.shuffle(rest)
    n_prefs = random.randint(1, len(candidates))
    return [first] + rest[:n_prefs - 1]


############################
# --- Plot Function ---
############################

def make_round_plot(r, rounds, quota, candidates, ballot_states, cand_colour, my_ballot):
    all_votes = [v for rd in rounds for v in rd['totals'].values()]
    x_max = max(max(all_votes) * 1.15, quota * 1.1)

    first_round_totals = rounds[0]['totals']
    candidate_order = sorted(candidates,
                             key=lambda c: first_round_totals.get(c, 0))

    my_state = next(s for s in ballot_states if s['is_mine'])

    # Rebuild state up to and including this round
    elected_so_far    = set()
    eliminated_so_far = set()
    my_weight         = 1.0
    my_exhausted      = 0.0
    my_elected_weights = {}

    for rd in rounds:
        elected_this_round = set()
        for e in rd['events']:
            if e[0] == 'elected':
                elected_so_far.add(e[1])
                elected_this_round.add(e[1])
            elif e[0] == 'eliminated':
                loser = e[1]
                eliminated_so_far.add(loser)
                if get_active_candidate(my_state, eliminated_so_far - {loser}, elected_so_far) == loser:
                    next_pref = get_active_candidate(my_state, eliminated_so_far, elected_so_far)
                    if next_pref is None:
                        my_exhausted += my_weight
                        my_weight     = 0.0
            elif e[0] == 'transfer':
                winner = e[1]
                tv     = e[2]
                my_current_before = get_active_candidate(
                    my_state, eliminated_so_far, elected_so_far - {winner}
                )
                if my_current_before == winner:
                    my_elected_weights[winner] = my_weight * (1 - tv)
                    next_pref = get_active_candidate(my_state, eliminated_so_far, elected_so_far)
                    if next_pref is None:
                        my_exhausted += my_weight * tv
                        my_weight     = my_weight * (1 - tv)
                    else:
                        my_weight *= tv

        if rd['round'] == r['round']:
            break

    totals            = dict(r['totals'])
    events            = r['events']
    breakdown         = r['transfer_breakdown']
    elected_this_round = {e[1] for e in events if e[0] == 'elected'}

    for c in elected_so_far - elected_this_round:
        totals[c] = quota

    my_current = get_active_candidate(my_state, eliminated_so_far, elected_so_far)

    def label_bar(ax, x_end, y_pos, val):
        label = f'{val:.2f}'
        if x_end > x_max * 0.08:
            ax.text(x_end - x_max * 0.01, y_pos, label,
                    va='center', ha='right', fontsize=10,
                    color='white', fontweight='bold')
        else:
            ax.text(x_end + x_max * 0.01, y_pos, label,
                    va='center', ha='left', fontsize=10,
                    color='#333333')

    fig, (ax, ax_ballot) = plt.subplots(
        1, 2, figsize=(13, 6),
        gridspec_kw={'width_ratios': [3, 1]}
    )
    fig.subplots_adjust(right=0.85, wspace=0.4)

    ytick_labels = [f'✓ {c}' if c in elected_so_far else c for c in candidate_order]
    ax.set_yticks(range(len(candidate_order)))
    ax.set_yticklabels(ytick_labels, fontsize=12)

    for i, c in enumerate(candidate_order):
        val = totals.get(c, 0)
        if val == 0:
            continue

        if c in eliminated_so_far and c in breakdown:
            left = 0
            for dest, dval in sorted(breakdown[c].items()):
                colour = cand_colour.get(dest, '#aaaaaa')
                ax.barh(i, dval, left=left, color=colour,
                        edgecolor='white', linewidth=0.5, zorder=3)
                left += dval
            ax.barh(i, val, color='none', edgecolor='#888888',
                    linewidth=1.5, linestyle='--', zorder=4)
            label_bar(ax, val, i, val)

        elif c in elected_so_far:
            if c in elected_this_round:
                actual = r['totals'][c]
                ax.barh(i, min(actual, quota), color=cand_colour[c],
                        edgecolor='white', linewidth=0.5, zorder=3)
                surplus_end = quota
                if c in breakdown:
                    for dest, dval in sorted(breakdown[c].items()):
                        colour = cand_colour.get(dest, '#aaaaaa')
                        ax.barh(i, dval, left=surplus_end, color=colour,
                                edgecolor='white', linewidth=0.5, zorder=3)
                        surplus_end += dval
                label_bar(ax, surplus_end, i, actual)
            else:
                actual = totals.get(c, quota)
                ax.barh(i, quota, color=cand_colour[c],
                        edgecolor='white', linewidth=0.5, zorder=3)
                if actual > quota:
                    ax.barh(i, actual - quota, left=quota,
                            color='#aaaaaa', edgecolor='white',
                            linewidth=0.5, zorder=3)
                label_bar(ax, actual, i, actual)
        else:
            ax.barh(i, val, color=cand_colour[c],
                    edgecolor='white', linewidth=0.6, zorder=3)
            label_bar(ax, val, i, val)

    ax.axvline(x=quota, color='#CC3333', linestyle='--', linewidth=1.5, zorder=5)
    ax.set_xlim(0, x_max)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', labelsize=12)
    ax.xaxis.grid(True, linestyle='--', alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

    event_strs = []
    for e in events:
        if e[0] == 'elected':
            event_strs.append(f'✓ {e[1]} elected')
        elif e[0] == 'eliminated':
            event_strs.append(f'✗ {e[1]} eliminated')
        elif e[0] == 'transfer':
            event_strs.append(f'↳ surplus transfer (TV={e[2]:.4f})')
    xlabel = 'Weighted votes'
    if event_strs:
        xlabel += '\n' + '  |  '.join(event_strs)
    ax.set_xlabel(xlabel, fontsize=12, labelpad=10)
    ax.set_title(f'Round {r["round"]} of {len(rounds)}',
                 fontsize=15, fontweight='bold', pad=10)

    legend_handles = [
        mpatches.Patch(color=cand_colour[c], label=c) for c in candidates
    ]
    legend_handles.append(
        mlines.Line2D([], [], color='#CC3333', linestyle='--',
                      linewidth=1.5, label=f'Quota ({quota})')
    )
    ax.legend(handles=legend_handles, fontsize=10, framealpha=0.7,
              bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

    # ── Your ballot panel ─────────────────────────────────────
    ax_ballot.axis('off')
    ax_ballot.set_title('Your Ballot', fontsize=13, fontweight='bold', pad=10)

    y      = 0.92
    line_h = 0.11

    for i, c in enumerate(my_ballot):
        if c in eliminated_so_far:
            symbol, colour, weight_str, bold = '✗', '#aaaaaa', '', False
        elif c in elected_so_far:
            symbol = '✓'
            colour = cand_colour[c]
            weight_str = f' (weight {my_elected_weights[c]:.4f})' \
                         if c in my_elected_weights else ''
            bold = False
        elif c == my_current:
            symbol, colour = '►', cand_colour[c]
            weight_str = f' (weight {my_weight:.4f})'
            bold = True
        else:
            symbol, colour, weight_str, bold = ' ', '#333333', '', False

        ax_ballot.text(
            0.05, y, f'{i+1}. {symbol} {c}{weight_str}',
            transform=ax_ballot.transAxes,
            fontsize=11, fontweight='bold' if bold else 'normal',
            color=colour, va='top'
        )
        y -= line_h

    ax_ballot.text(
        0.05, y, f'   Exhausted (weight {my_exhausted:.4f})',
        transform=ax_ballot.transAxes, fontsize=11,
        color='#e74c3c' if my_exhausted > 0 else '#cccccc',
        va='top', style='italic'
    )

    fig.tight_layout(rect=[0, 0, 0.85, 1])
    return fig


############################
# --- UI ---
############################

app_ui = ui.page_fluid(
    ui.h2("STV Election Explorer"),
    ui.p(f"Candidates: {', '.join(candidates)} — {N_SEATS} seats"),

    ui.navset_tab(

        # ── Main Tab ──────────────────────────────────────────
        ui.nav_panel(
            "Election",
            ui.br(),
            ui.row(
                ui.column(6,
                ui.h4("Your Ballot"),
                ui.p("Drag candidates into your preferred order. "
                     "Drag to the bottom section to leave unranked."),
                ui.HTML("""
                    <style>
                    .ballot-container { display: flex; gap: 16px; margin-bottom: 12px; }
                    .ballot-list {
                        flex: 1; min-height: 80px; border: 2px solid #ccc;
                        border-radius: 6px; padding: 8px; background: #f9f9f9;
                    }
                    .ballot-list.ranked   { border-color: #4C72B0; background: #eef2fa; }
                    .ballot-list.unranked { border-color: #ccc;    background: #f9f9f9; }
                    .ballot-list h6 { margin: 0 0 6px 0; font-size: 12px;
                                      color: #666; text-transform: uppercase; }
                    .ballot-item {
                        padding: 6px 10px; margin: 4px 0;
                        background: white; border: 1px solid #bbb;
                        border-radius: 4px; cursor: grab;
                        font-size: 13px; user-select: none;
                        display: flex; align-items: center; gap: 8px;
                    }
                    .ballot-item .rank-num {
                        font-weight: bold; color: #4C72B0;
                        min-width: 18px; font-size: 12px;
                    }
                    .ballot-item:active { cursor: grabbing; opacity: 0.7; }
                    .sortable-ghost { opacity: 0.3; background: #c8e6ff; }
                    </style>

                    <div class="ballot-container">
                        <div style="flex:1">
                            <div class="ballot-list ranked" id="ranked-list">
                                <h6>&#9650; Ranked (drag to order)</h6>
                            </div>
                        </div>
                        <div style="flex:1">
                            <div class="ballot-list unranked" id="unranked-list">
                                <h6>Not ranked</h6>
                            </div>
                        </div>
                    </div>
                    <input type="hidden" id="ballot_order_hidden" />

                    <script src="https://cdnjs.cloudflare.com/ajax/libs/Sortable/1.15.0/Sortable.min.js"></script>
                    <script>
                    (function() {
                        // Wait for Shiny and candidates to be available
                        function initBallot() {
                            var candidates = CANDIDATE_LIST_PLACEHOLDER;
                            var ranked   = document.getElementById('ranked-list');
                            var unranked = document.getElementById('unranked-list');
                            var hidden   = document.getElementById('ballot_order_hidden');

                            // Build items
                            candidates.forEach(function(name) {
                                var item = document.createElement('div');
                                item.className = 'ballot-item';
                                item.dataset.name = name;
                                item.innerHTML = '<span class="rank-num"></span>' + name;
                                unranked.appendChild(item);
                            });

                            function updateRanks() {
                                var items = ranked.querySelectorAll('.ballot-item');
                                items.forEach(function(el, idx) {
                                    el.querySelector('.rank-num').textContent = (idx + 1) + '.';
                                });
                                var unrankedItems = unranked.querySelectorAll('.ballot-item');
                                unrankedItems.forEach(function(el) {
                                    el.querySelector('.rank-num').textContent = '';
                                });
                                // Update hidden input and notify Shiny
                                var order = Array.from(items).map(function(el) {
                                    return el.dataset.name;
                                });
                                hidden.value = order.join(',');
                                // Trigger Shiny to pick up the value
                                if (window.Shiny) {
                                    Shiny.setInputValue('my_ballot_input', hidden.value);
                                }
                            }

                            Sortable.create(ranked, {
                                group: 'ballot',
                                animation: 150,
                                ghostClass: 'sortable-ghost',
                                onSort: updateRanks,
                                onAdd:  updateRanks,
                                onRemove: updateRanks,
                            });

                            Sortable.create(unranked, {
                                group: 'ballot',
                                animation: 150,
                                ghostClass: 'sortable-ghost',
                                onSort: updateRanks,
                                onAdd:  updateRanks,
                                onRemove: updateRanks,
                            });

                            updateRanks();
                        }

                        if (document.readyState === 'loading') {
                            document.addEventListener('DOMContentLoaded', initBallot);
                        } else {
                            initBallot();
                        }
                    })();
                    </script>
                """.replace('CANDIDATE_LIST_PLACEHOLDER',
                            str(candidates).replace("'", '"'))),
            ),
                ui.column(3,
                    ui.br(),
                    ui.input_action_button("run", "Run Election",
                                           class_="btn btn-primary"),
                ),
            ),
            ui.br(),
            ui.output_ui("round_selector_ui"),
            ui.output_plot("round_plot", height="550px"),
        ),

        # ── Parameters Tab ────────────────────────────────────
        ui.nav_panel(
            "Election Parameters ⚙",
            ui.br(),
            ui.row(
                ui.column(4,
                    ui.input_numeric("n_votes", "Number of votes", value=200,
                                     min=10, max=10000, step=10),
                    ui.input_numeric("seed", "Random seed (blank = random)",
                                     value=42, min=0, max=99999),
                ),
            ),
            ui.br(),
            ui.h4("Candidate first-preference weights (%)"),
            ui.p("Set weights for specific candidates. "
                 "Unset candidates share the remainder equally. "
                 "Total must not exceed 100%."),
            ui.row(*[
                ui.column(2,
                    ui.input_numeric(
                        cand_id(c), c,
                        value=None, min=0, max=100, step=1
                    )
                )
                for c in candidates
            ]),
        ),
    ),
)


############################
# --- Server ---
############################

def server(input, output, session):

    election_result = reactive.Value(None)

    @reactive.Effect
    @reactive.event(input.run)
    def _run():
        # --- Seed ---
        seed = input.seed()
        if seed is not None:
            np.random.seed(int(seed))
            random.seed(int(seed))

        # --- Weights ---
        custom = {}
        for c in candidates:
            val = getattr(input, cand_id(c))()
            if val is not None and val > 0:
                custom[c] = val

        total_custom = sum(custom.values())
        if total_custom > 100:
            return  # invalid — could add a ui.notification here

        remainder = 1.0 - total_custom / 100
        unset = [c for c in candidates if c not in custom]
        weights = {}
        for c in candidates:
            if c in custom:
                weights[c] = custom[c] / 100
            else:
                weights[c] = remainder / len(unset) if unset else 0.0

        # --- Ballots ---
        n_votes = input.n_votes()
        ballots = [generate_ballot(candidates, weights) for _ in range(n_votes)]

        # --- My ballot ---
        raw = input.my_ballot_input().strip()
        my_ballot_list = [c.strip() for c in raw.split(',') if c.strip()]
        valid = [c for c in my_ballot_list if c in candidates]
        if not valid:
            valid = [candidates[0]]

        # Deduplicate preserving order
        seen = set()
        my_ballot = []
        for c in valid:
            if c not in seen:
                my_ballot.append(c)
                seen.add(c)

        # Add my ballot as a real ballot
        ballots.append(my_ballot)
        my_ballot_ref = ballots[-1]

        # --- Run ---
        rounds, quota, cand_colour, ballot_states, elected_totals = run_election(
            ballots, my_ballot_ref, N_SEATS, candidates
        )

        election_result.set({
            'rounds'        : rounds,
            'quota'         : quota,
            'cand_colour'   : cand_colour,
            'ballot_states' : ballot_states,
            'elected_totals': elected_totals,
            'my_ballot'     : my_ballot_ref,
        })

    @output
    @render.ui
    def round_selector_ui():
        result = election_result()
        if result is None:
            return ui.p("Press 'Run Election' to begin.",
                        style="color: gray; font-style: italic;")
        n_rounds = len(result['rounds'])
        return ui.input_slider("round_num", "Round", min=1, max=n_rounds,
                               value=1, step=1, animate=True)

    @output
    @render.plot
    def round_plot():
        result = election_result()
        if result is None:
            return None
        rn = input.round_num()
        r  = result['rounds'][rn - 1]
        return make_round_plot(
            r,
            result['rounds'],
            result['quota'],
            candidates,
            result['ballot_states'],
            result['cand_colour'],
            result['my_ballot'],
        )


app = App(app_ui, server)
