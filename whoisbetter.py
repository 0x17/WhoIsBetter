import sys


# TODO: Average gap to *optimum* (j30) (bks für j120) auf allen Instanzen für jedes Verfahren berechnen (Benchmark)
# TODO: Dann avg gap jeweils der bks to opt berechnen (Orakel/Best case) (jeweils bks von bestem verfahren)
# TODO: Dann mit Prediction (welcher split?) avg gap to opt (jeweils bks von gewähltem verfahren)

# Erst zeigen auch wenn "on average" ein Verfahren klar schlechter gibt es ein paar Instanzen, die damit am besten (Ensemble=gut)
# Dann zeigen, was maximal zu holen wäre (über alle Verfahren) mit Orakel
# Dann zeigen, dass trotz bisher schlechter Prediction trotzdem guter Speedup!

def csv_to_dict(fn):
    with open(fn, 'r') as fp:
        return {line.split(';')[0].strip(): float(line.split(';')[1].strip()) for line in fp.readlines() if
                len(line.strip()) > 0}


def parse_results(method_names):
    return {method_name: {'results': csv_to_dict(method_name + '_results.txt'),
                          'timeforbks': csv_to_dict(method_name + '_timeforbks.txt')} for method_name in method_names}


def average_gaps_for_singular_methods(method_selection, res_dict, instances_subset=None):
    results = {mn: pair['results'] for mn, pair in res_dict.items() if mn in method_selection}
    avg_gaps = {}

    def inst_pred(instance):
        return instances_subset is None or instance in instances_subset

    for mn, res in results.items():
        accum = 0
        for instance, profit in res.items():
            if inst_pred(instance):
                bks = max(pair[instance] for pair in results.values())
                accum += (bks - profit) / bks
        accum /= len(res)
        avg_gaps[mn] = accum

    return avg_gaps


def average_gap_for_per_instance_choice(method_selection, res_dict, choices, instances_subset=None):
    results = {mn: pair['results'] for mn, pair in res_dict.items() if mn in method_selection}
    mnames = list(results.keys())
    first_method = mnames[0]

    def inst_pred(instance):
        return instances_subset is None or instance in instances_subset

    def chosen_results():
        return {inst: results[mnames[choices[inst]]][inst] for inst in results[first_method].keys() if inst_pred(inst)}

    accum = 0
    for instance, profit in chosen_results().items():
        bks = max(pair[instance] for pair in results.values())
        accum += (bks - profit) / bks
    return accum / len(res)


def write_gaps(method_selection, res_dict, prediction_fn, ordered_instances):
    def choices_from_predfile():
        with open(prediction_fn, 'r') as fp:
            return {ordered_instances[int(line.split(',')[0]) - 1]: int(line.split(',')[2].split(':')[1]) for line in
                    fp.readlines()[1:] if len(line.strip()) > 0}

    def to_s(l): return [str(v) for v in l]

    cvec = choices_from_predfile()
    instances_with_prediction = cvec.keys()

    sing_gaps = average_gaps_for_singular_methods(method_selection, res_dict, instances_with_prediction)
    pred_gap = average_gap_for_per_instance_choice(method_selection, res_dict, cvec, instances_with_prediction)
    # pred_gap = 0.0
    with open('predgaps.csv', 'w') as fp:
        fp.write(';'.join(list(sing_gaps.keys()) + ['predchoice']) + '\n')
        fp.write(';'.join(to_s(list(sing_gaps.values()) + [pred_gap])) + '\n')


def write_who_is_better(method_selection, res_dict, characteristics_fn=None, one_hot=True, treat_gurobi_better=False,
                        profits=False):
    def to_float(l):
        return [float(v) for v in l]

    char_dict = None
    char_header = ''
    if characteristics_fn is not None:
        with open(characteristics_fn, 'r') as fp:
            lines = fp.readlines()
            char_dict = {line.split(';')[0]: to_float(line.split(';')[1:]) for line in lines[1:]}
            char_header = ';'.join(lines[0].split(';')[1:]).replace('\n', '')

    method_names = list(res_dict.keys())

    def is_best(method_name, instance):
        def profit(mn=method_name):
            return res_dict[mn]['results'][instance]

        def timetobks(mn=method_name):
            return res_dict[mn]['timeforbks'][instance]

        best_profit = max(profit(mn) for mn in method_names)
        its_profit = profit()
        its_timetobks = timetobks()
        other_methods_with_bks = [mn for mn in method_names if mn != method_name and profit(mn) == best_profit]
        other_methods_with_bks_timeforbks = [timetobks(mn) for mn in other_methods_with_bks]

        # GUROBI provides bounds, therefore treat it better
        if treat_gurobi_better:
            if method_name == 'Gurobi' and its_profit == best_profit: return 1
            if method_name != 'Gurobi' and profit('Gurobi') == best_profit: return 0

        return 1 if its_profit == best_profit and (not other_methods_with_bks_timeforbks or its_timetobks <= min(
            other_methods_with_bks_timeforbks)) else 0

    class_header = ';' + ';'.join(method_selection) if one_hot or profits else ';best_ix'
    lines = ['instance;' + char_header + class_header]

    instances = [i for i in res_dict[method_names[0]]['results'] if
                 all(i in res_dict[mn]['results'] for mn in method_names[1:]) and all(
                     i in res_dict[mn]['timeforbks'] for mn in method_names)]
    stats = {}
    for instance in instances:
        who_is_best_vector_str = []
        if char_dict is not None:
            if instance not in char_dict: continue
            who_is_best_vector_str += [str(v) for v in char_dict[instance]]
        oh_vec = [str(is_best(method_name, instance)) for method_name in method_names]
        profits_vec = [str(res_dict[method_name]['results'][instance]) for method_name in method_names]
        who_is_best_vector_str += profits_vec if profits else (oh_vec if one_hot else [str(oh_vec.index('1'))])
        if not one_hot and not profits:
            k = int(who_is_best_vector_str[-1])
            stats[k] = 1 if k not in stats else stats[k] + 1
        lines.append(f'{instance};' + ';'.join(who_is_best_vector_str))
    ostr = '\n'.join(lines)
    with open('whoisbetter.csv', 'w') as fp:
        fp.write(ostr)
    if not one_hot and not profits: print(stats)

    return instances


def parse_extended_results(fn):
    def inst_name(s): return s.replace('j30gdx/', '').replace('.sm', '')

    with open(fn, 'r') as fp:
        lines = fp.readlines()
        profits = {inst_name(line.split()[0]): float(line.split()[1]) for line in lines}
        solvetimes = {inst_name(line.split()[0]): float(line.split()[2]) for line in lines}
        return dict(results=profits, timeforbks=solvetimes)


def extended_results_dictionary(solver_names):
    return {solver_name: parse_extended_results(f'GMS_{solver_name.upper()}_ExtendedResults.txt') for solver_name in
            solver_names}

def parse_optimal_results():
    with open('j30opt.csv', 'r') as fp:
        lines = fp.readlines()[1:]
        makespans = { line.split(';')[0]: line.split(';')[1] for line in lines }
        solvetimes = {line.split(';')[0]: line.split(';')[2] for line in lines }
        return dict(results=makespans, timeforbks=solvetimes)



if __name__ == '__main__':
    if len(sys.argv) <= 2:
        print("Usage: python whoisbetter.py (exact|heur) Method1 Method2 ...")
        exit(0)

    method_selection = sys.argv[2:]

    res = extended_results_dictionary(method_selection) if sys.argv[1] == 'exact' else parse_results(method_selection)
    #method_selection = ['best']
    #res = {'best': parse_optimal_results()}

    # method_selection = ['GA0', 'GA3', 'GA4', 'Gurobi']
    # method_selection = ['GA3', 'Gurobi']
    # method_selection = ['GA0', 'GA3', 'Gurobi']
    # instances = write_who_is_better(method_selection, res, 'characteristics.csv', one_hot=False, treat_gurobi_better=False)


    instances = write_who_is_better(method_selection=method_selection,
                                    res_dict=res,
                                    characteristics_fn='flattenedj30.csv',
                                    one_hot=True,
                                    treat_gurobi_better=False,
                                    profits=False)

    # write_gaps(method_selection, res, 'prediction.csv', instances)
