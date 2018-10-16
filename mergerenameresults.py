import os
import shutil


def slurp(fn):
    with open(fn, 'r') as fp:
        return fp.read()


def spit(fn, s):
    with open(fn, 'w') as fp:
        fp.write(s)


def merge_profits(dir_a, dir_b=None):
    res_fns = [fn.replace(dir_a, '') for fn in os.listdir(dir_a) if fn.endswith('Results.txt')]
    for res_fn in res_fns:
        ofn = res_fn.replace('Results.txt', '_results.txt')
        info_msg = f'merge not required, just renaming {dir_a+res_fn} into {ofn}...' if dir_b is None else f'merge {dir_a+res_fn} and {dir_b+res_fn} into {ofn}...'
        print(info_msg)
        spit(ofn, slurp(dir_a + res_fn) + (slurp(dir_b + res_fn) if dir_b is not None else ""))


def merge_time_for_bks():
    def is_last_improv_fn(fn):
        return fn.endswith('_LastImprovementTime.txt') or fn.endswith('_TimeAtLastImprovementTime.txt')

    last_improv_fn_mapping = {
        'GUROBI': 'Gurobi',
        'TimeWindowBordersGA': 'GA0',
        'FixedCapacityGA': 'GA3',
        'TimeVaryingCapacityGA': 'GA4',
        'CompareAlternativesGA': 'GA5',
        'GoldenSectionSearchGA': 'GA6',
        'TimeVaryingCapacityRandomKeyGA': 'GA9'
    }
    time_for_bks_fns = [fn for fn in os.listdir('.') if is_last_improv_fn(fn)]
    for fn in time_for_bks_fns:
        src = fn
        dest = last_improv_fn_mapping[
                   fn.replace('_LastImprovementTime.txt', '').replace('_TimeAtLastImprovementTime.txt',
                                                                      '')] + '_timeforbks.txt'
        print(f'copy {src} to {dest}...')
        shutil.copy(src, dest)


if __name__ == '__main__':
    # merge_profits('j30res/', 'k30res/')
    # merge_profits('j120_1800secs/')
    merge_profits('k120_1800secs/', 'j120_1800secs/')
    merge_time_for_bks()
