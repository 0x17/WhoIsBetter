import sys

from whoisbetter import parse_results
import whoisbetter

if __name__ == '__main__':
    method_names = sys.argv[1:]
    assert len(method_names) == 2
    pred_fn = 'predictions.csv'
    path = whoisbetter.path_option = "combined_j30_upto_120_rgen/"
    res_dict = parse_results(method_names)
    with open(path + pred_fn, 'r') as fp:
        lines = fp.readlines()
        preds = {line.split(';')[0]: int(float(line.split(';')[1])) for line in lines if len(line) > 0}
        instances = list(preds.keys())
        m_a, m_b = method_names

        a_res = lambda instance: res_dict[m_a]['results'][instance]
        b_res = lambda instance: res_dict[m_b]['results'][instance]
        best_res = lambda instance: max(a_res(instance), b_res(instance))
        pred_res = lambda instance: res_dict[m_a if preds[instance] == 0 else m_b]['results'][instance]


        odict = dict(accuracy_always_a=sum(1 if a_res(instance) == best_res(instance) else 0 for instance in instances)/len(instances),
                     accuracy_always_b=sum(1 if b_res(instance) == best_res(instance) else 0 for instance in instances) / len(instances),
                     profit_always_a=sum(a_res(instance) for instance in instances),
                     profit_always_b=sum(b_res(instance) for instance in instances),
                     profit_oracle=sum(best_res(instance) for instance in instances),
                     profit_selector=sum(pred_res(instance) for instance in instances),
                     gap_always_a=sum((best_res(instance) - a_res(instance))/best_res(instance) for instance in instances)/len(instances),
                     gap_always_b=sum((best_res(instance) - b_res(instance)) / best_res(instance) for instance in instances) / len(instances),
                     gap_selector=sum((best_res(instance) - pred_res(instance)) / best_res(instance) for instance in instances) / len(instances)
                     )
        print(odict)
