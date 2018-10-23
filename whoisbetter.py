import sys
import compimpact
import os


def csv_to_dict(fn):
    global path_option
    with open(path_option + fn, 'r') as fp:
        return {line.split(';')[0].strip(): float(line.split(';')[1].strip()) for line in fp.readlines() if
                len(line.strip()) > 0}


def parse_results(method_names):
    has_time_for_bks = lambda method_name: os.path.exists(method_name + '_timeforbks.txt')
    return {method_name: {'results': csv_to_dict(method_name + '_results.txt'),
                          'timeforbks': csv_to_dict(method_name + '_timeforbks.txt') if has_time_for_bks(method_name) else None} for method_name in method_names}


def write_who_is_better(method_selection,
                        res_dict,
                        characteristics_fn=None,
                        one_hot=False,
                        treat_gurobi_better=False,
                        regression=False,
                        remove_equal_profits=True):
    global path_option

    def to_float(l):
        return [float(v) for v in l]

    char_dict = None
    char_header = ''
    if characteristics_fn is not None:
        with open(path_option+characteristics_fn, 'r') as fp:
            lines = fp.readlines()
            char_dict = {line.split(';')[0]: to_float(line.split(';')[1:]) for line in lines[1:]}
            char_header = ';'.join(lines[0].split(';')[1:]).replace('\n', '')

    method_names = list(res_dict.keys())

    def is_best(method_name, instance):
        def profit(mn=method_name):
            return res_dict[mn]['results'][instance]

        def timetobks(mn=method_name):
            return 0 if res_dict[mn]['timeforbks'] is None else res_dict[mn]['timeforbks'][instance]

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

    class_header = ';' + ';'.join(method_selection) if one_hot or regression else ';best_ix'
    lines = ['instance;' + char_header + class_header]

    instances = [i for i in res_dict[method_names[0]]['results'] if
                 all(i in res_dict[mn]['results'] for mn in method_names[1:]) and all(
                     res_dict[mn]['timeforbks'] is None or i in res_dict[mn]['timeforbks'] for mn in method_names)]
    stats = {}

    if remove_equal_profits:
        def num_methods_best(instance):
            max_profit = max(res_dict[mn]['results'][instance] for mn in method_selection)
            return sum(1 for mn in method_selection if res_dict[mn]['results'][instance] == max_profit)

        instances = [ instance for instance in instances if num_methods_best(instance) == 1]

    for instance in instances:
        who_is_best_vector_str = []
        if char_dict is not None:
            if instance not in char_dict: continue
            who_is_best_vector_str += [str(v) for v in char_dict[instance]]
        oh_vec = [str(is_best(method_name, instance)) for method_name in method_names]
        profits_vec = [str(res_dict[method_name]['results'][instance]) for method_name in method_names]
        who_is_best_vector_str += profits_vec if regression else (oh_vec if one_hot else [str(oh_vec.index('1'))])
        if not one_hot and not regression:
            k = int(who_is_best_vector_str[-1])
            stats[k] = 1 if k not in stats else stats[k] + 1
        lines.append(f'{instance};' + ';'.join(who_is_best_vector_str))

    ostr = '\n'.join(lines)

    with open(path_option+'whoisbetter.csv', 'w') as fp:
        fp.write(ostr)

    if not one_hot and not regression:
        print(stats)

    return instances


def parse_extended_results(fn):
    def inst_name(s): return s.replace('k20gdx/', '').replace('.sm', '')

    with open(path_option+fn, 'r') as fp:
        lines = fp.readlines()
        profits = {inst_name(line.split()[0]): float(line.split()[1]) for line in lines}
        solvetimes = {inst_name(line.split()[0]): float(line.split()[2]) for line in lines}
        return dict(results=profits, timeforbks=solvetimes)


def extended_results_dictionary(solver_names):
    return {solver_name: parse_extended_results(f'GMS_{solver_name.upper()}_ExtendedResults.txt') for solver_name in
            solver_names}


def parse_optimal_results():
    with open(path_option+'j30opt.csv', 'r') as fp:
        lines = fp.readlines()[1:]
        makespans = {line.split(';')[0]: line.split(';')[1] for line in lines}
        solvetimes = {line.split(';')[0]: line.split(';')[2] for line in lines}
        return dict(results=makespans, timeforbks=solvetimes)


if __name__ == '__main__':
    if len(sys.argv) <= 2:
        print("Usage: python whoisbetter.py (exact|heur) Method1 Method2 ... [root=path]")
        exit(0)

    path_option_str = next((arg for arg in sys.argv if arg.startswith('root=')), None)
    path_option = '' if path_option_str is None else path_option_str.split('=')[1]
    if len(path_option) > 0 and not path_option.endswith('/'): path_option += '/'

    method_selection = [arg for arg in sys.argv[2:] if not arg.startswith('root=')]

    res = extended_results_dictionary(method_selection) if sys.argv[1] == 'exact' else parse_results(method_selection)
    # method_selection = ['best']
    # res = {'best': parse_optimal_results()}

    # method_selection = ['GA0', 'GA3', 'GA4', 'Gurobi']
    # method_selection = ['GA3', 'Gurobi']
    # method_selection = ['GA0', 'GA3', 'Gurobi']
    # instances = write_who_is_better(method_selection, res, 'characteristics.csv', one_hot=False, treat_gurobi_better=False)

    best_pair_ratio = None

    compet_results = compimpact.comp_measure_combinations(res, method_selection)
    for (a, b), stats in compet_results.items():
        print(f'Competitiveness stats for pair {a}, {b}: {stats}')
        if best_pair_ratio is None or stats['ratio'] > best_pair_ratio[1]:
            best_pair_ratio = ((a,b), stats['ratio'])

    best_pair_impact = None

    impact_results = compimpact.comp_measure_combinations(res, method_selection, compimpact.perf_measurefunc('profit'))
    for (a, b), stats in impact_results.items():
        print(f'Performance stats for pair {a}, {b}: {stats}')
        if best_pair_impact is None or stats['potential_impact'] > best_pair_impact[1]:
            best_pair_impact = ((a,b), stats['potential_impact'])

    print(f'Best pair and ratio {best_pair_ratio} with impact {impact_results[best_pair_ratio[0]]}')
    print(f'Best pair and impact {best_pair_impact} with ratio {compet_results[best_pair_impact[0]]}')



    instances = write_who_is_better(method_selection=method_selection,
                                    res_dict=res,
                                    characteristics_fn='characteristics.csv',
                                    one_hot=True,
                                    treat_gurobi_better=False,
                                    regression=False,
                                    remove_equal_profits=True)

    # write_gaps(method_selection, res, 'prediction.csv', instances)
