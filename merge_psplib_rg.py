
suffixes = ['j30', 'j60', 'j90', 'j120', 'rangen']
prefixes = ['j30', 'j60', 'j90', 'j120', 'rgentestset']
method_prefixes = ['GA', 'LocalSolverNative']
method_nums = [0, 3, 4]


def merge_characteristics():
    ostr = ''
    for ix, fn in enumerate([f'characteristics{suffix}.csv' for suffix in suffixes]):
        with open(fn, 'r') as fp:
            ostr += ''.join(fp.readlines()[1 if ix != 0 else 0:])+'\n'
    with open('characteristics.csv', 'w') as fp:
        fp.write(ostr.replace('.rcp', '').replace('.sm', ''))


def merge_results(res_files, ofn):
    ostr = ''
    for fn in res_files:
        with open(fn, 'r') as fp:
            ostr += fp.read()
    with open(ofn, 'w') as fp:
        fp.write(ostr.replace('.rcp', '').replace('.sm', ''))


merge_characteristics()

for prefix, num in [(pf, n) for pf in method_prefixes for n in method_nums]:
    ofn = f'{prefix}{num}_results.txt'
    paths = [dpfx + '_5000schedules' for dpfx in prefixes]
    res_files = [f'{path}/{prefix}{num}Results.txt' for path in paths]
    merge_results(res_files, ofn)
