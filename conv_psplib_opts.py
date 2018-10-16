
def convert_optimals_to_csv(fn):
    ofn = fn.replace('.sm', '.csv')
    ostr = 'instance;ms;solvetime\n'
    with open(fn, 'r') as fp:
        for line in fp.readlines()[22:22+480]:
            parts = line.split()
            instance = f'j30{parts[0]}_{parts[1]}'
            ostr += ';'.join([instance]+parts[2:])+'\n'
    with open(ofn, 'w') as fp:
        fp.write(ostr)


if __name__ == '__main__':
    convert_optimals_to_csv('j30opt.sm')