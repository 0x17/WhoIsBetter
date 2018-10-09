def csv_to_dict(fn):
    with open(fn, 'r') as fp:
        return {line.split(';')[0].strip(): float(line.split(';')[1].strip()) for line in fp.readlines() if len(line.strip()) > 0}


def parse_results(method_names):
    return {method_name: {'results': csv_to_dict(method_name + '_results.txt'),
                          'timeforbks': csv_to_dict(method_name + '_timeforbks.txt')} for method_name in method_names}


def write_who_is_better(method_selection, res_dict, characteristics_fn = None, one_hot = True, binary_classification = False):
    def to_float(l): return [ float(v) for v in l ]
    char_dict = None
    char_header = ''
    if characteristics_fn is not None:
        with open(characteristics_fn, 'r') as fp:
            lines = fp.readlines()
            char_dict = { line.split(';')[0]: to_float(line.split(';')[1:]) for line in lines[1:] }
            char_header = ';'.join(lines[0].split(';')[1:]).replace('\n', '')

    method_names = list(res_dict.keys())

    def is_best(method_name, instance):
        def profit(mn=method_name): return res_dict[mn]['results'][instance]

        def timetobks(mn=method_name): return res_dict[mn]['timeforbks'][instance]

        best_profit = max(profit(mn) for mn in method_names)
        its_profit = profit()
        its_timetobks = timetobks()
        other_methods_with_bks = [mn for mn in method_names if mn != method_name and profit(mn) == best_profit]
        other_methods_with_bks_timeforbks = [timetobks(mn) for mn in other_methods_with_bks]
        return 1 if its_profit == best_profit and (not other_methods_with_bks_timeforbks or its_timetobks <= min(other_methods_with_bks_timeforbks)) else 0

    class_header = ';'+';'.join(method_selection) if one_hot else ';best_ix'
    lines = ['instance;'+char_header+class_header]

    instances = [i for i in res_dict[method_names[0]]['results'] if
                 all(i in res_dict[mn]['results'] for mn in method_names[1:]) and all(i in res_dict[mn]['timeforbks'] for mn in method_names)]
    stats = {}
    for instance in instances:
        who_is_best_vector_str = []
        if char_dict is not None:
            if instance not in char_dict: continue
            who_is_best_vector_str += [str(v) for v in char_dict[instance]]
        oh_vec = [str(is_best(method_name, instance)) for method_name in method_names]
        who_is_best_vector_str += oh_vec if one_hot else [str( (1 if oh_vec.index('1') == 3 else 0) if binary_classification else oh_vec.index('1'))]
        if not one_hot:
            k = int(who_is_best_vector_str[-1])
            stats[k] = 1 if k not in stats else stats[k]+1
        lines.append(f'{instance};' + ';'.join(who_is_best_vector_str))
    ostr = '\n'.join(lines)
    with open('whoisbetter.csv', 'w') as fp:
        fp.write(ostr)
    if not one_hot: print(stats)


if __name__ == '__main__':
    method_selection = ['GA0', 'GA3', 'GA4', 'Gurobi']
    #method_selection = ['GA3', 'Gurobi']
    res = parse_results(method_selection)
    write_who_is_better(method_selection, res, 'characteristics.csv', one_hot=False, binary_classification=False)
