import itertools


def profit(res_dict, mn, inst): return res_dict[mn]['results'][inst]


def treq(res_dict, mn, inst): return res_dict[mn]['timeforbks'][inst]


def dominated_instances(res_dict, method_name, other_mn, all_instances, consider_time=False):
    return [inst for inst in all_instances if profit(res_dict, method_name, inst) > profit(res_dict, other_mn, inst)
            or (consider_time and profit(res_dict, method_name, inst) == profit(res_dict, other_mn, inst)
                and treq(res_dict, method_name, inst) < treq(res_dict, other_mn, inst))]


def dominated_instances_time(res_dict, method_name, other_mn, instances):
    return [inst for inst in instances if treq(res_dict, method_name, inst) < treq(res_dict, other_mn, inst)]


def domination_count_pair(res_dict, mname_a, mname_b, all_instances, consider_time=False):
    return len(dominated_instances(res_dict, mname_a, mname_b, all_instances, consider_time)), len(
        dominated_instances(res_dict, mname_b, mname_a, all_instances, consider_time))


def all_instances_for_pair(res_dict, mname_a, mname_b):
    return [inst for inst in res_dict[mname_a]['results'].keys() if inst in res_dict[mname_b]['results']]


def all_pairs(method_names):
    return list(itertools.combinations(method_names, 2))


def competitiveness_measures(ndom_a, ndom_b, num_instances):
    a = ndom_a
    b = ndom_b
    t = num_instances
    c = 2 * min(a / t, b / t)
    return {
        'ratio': c,
        'equipotency': 2 * min(a / (a + b), b / (a + b)),
        'reach': (a + b) / t,
        'fraction_best': 1 - c / 2
    }


def remove_same_results(res_dict, instances, m_a, m_b, resfunc):
    return [instance for instance in instances if resfunc(res_dict, m_a, instance) != resfunc(res_dict, m_b, instance)]


def comp_measurefunc(res_dict, m_a, m_b):
    instances = all_instances_for_pair(res_dict, m_a, m_b)
    instances_differ = remove_same_results(res_dict, instances, m_a, m_b, profit)
    print(f'Differ count;{m_a};{m_b};{len(instances_differ)};{len(instances)};{len(instances_differ)/len(instances)}')
    nwin_a, nwin_b = domination_count_pair(res_dict, m_a, m_b, instances_differ, consider_time=False)
    return competitiveness_measures(nwin_a, nwin_b, len(instances))


def perf_measurefunc(obj='profit'):
    def func(res_dict, m_a, m_b):
        if obj == 'profit':
            selfunc = max
            objfunc = profit
        else:
            selfunc = min
            objfunc = treq

        obj_on_set = lambda s, mn: sum(objfunc(res_dict, mn, inst) for inst in s)

        instances = all_instances_for_pair(res_dict, m_a, m_b)
        instances_differ = remove_same_results(res_dict, instances, m_a, m_b, profit)
        single_algo_profits = [sum(objfunc(res_dict, mn, inst) for inst in instances_differ) for mn in [m_a, m_b]]
        oracle_profit = sum(selfunc(objfunc(res_dict, mn, inst) for mn in [m_a, m_b]) for inst in instances_differ)
        s_a = dominated_instances(res_dict, m_a, m_b, instances_differ, consider_time=obj != 'profit')
        s_b = dominated_instances(res_dict, m_b, m_a, instances_differ, consider_time=obj != 'profit')
        return {
            'potential_impact': min(abs(obj_on_set(s_a, m_b) - obj_on_set(s_a, m_a)),
                                    abs(obj_on_set(s_b, m_a) - obj_on_set(s_b, m_b))) / oracle_profit,
            'absolute_max_improvement': oracle_profit - selfunc(single_algo_profits),
            'relative_max_improvement': (oracle_profit - selfunc(single_algo_profits)) / oracle_profit,
        }
    return func


def comp_measure_combinations(res_dict, method_selection, measurefunc=comp_measurefunc):
    return {(m_a, m_b): measurefunc(res_dict, m_a, m_b) for m_a, m_b in all_pairs(method_selection)}


def dominated_instances_cons_time(res_dict, method_a, method_b, instances):
    return dominated_instances(res_dict, method_a, method_b, instances, consider_time=True)
