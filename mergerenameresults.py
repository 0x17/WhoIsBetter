import os
import shutil

def slurp(fn):
    with open(fn, 'r') as fp:
        return fp.read()


def spit(fn, s):
    with open(fn, 'w') as fp:
        fp.write(s)


def merge_profits(dir_a, dir_b):
    res_fns = [fn.replace(dir_a, '') for fn in os.listdir(dir_a) if fn.endswith('Results.txt')]
    for res_fn in res_fns:
        ofn = res_fn.replace('Results.txt', '_results.txt')
        print(f'merge {dir_a+res_fn} and {dir_b+res_fn} into {ofn}...')
        spit(ofn, slurp(dir_a + res_fn) + slurp(dir_b + res_fn))


def merge_time_for_bks():
    def is_last_improv_fn(fn):
        return fn.endswith('_LastImprovementTime.txt') or fn.endswith('_TimeAtLastImprovementTime.txt')

    last_improv_fn_mapping = {
        'GUROBI': 'Gurobi',
        'TimeWindowBordersGA': 'GA0',
        'FixedCapacityGA': 'GA3',
        'TimeVaryingCapacityGA': 'GA4'
    }
    time_for_bks_fns = [fn for fn in os.listdir('.') if is_last_improv_fn(fn)]
    for fn in time_for_bks_fns:
        src = fn
        dest = last_improv_fn_mapping[fn.replace('_LastImprovementTime.txt', '').replace('_TimeAtLastImprovementTime.txt', '')]+'_timeforbks.txt'
        print(f'copy {src} to {dest}...')
        shutil.copy(src, dest)


if __name__ == '__main__':
    merge_profits('j30res/', 'k30res/')
    merge_time_for_bks()
