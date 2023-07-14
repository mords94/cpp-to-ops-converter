

from os import path


list = []
list2 = []
if __name__ == '__main__':
    with open(path.join(path.dirname(__file__), '../ITK_POM2K/pom2k_fun.c')) as f:
        for line in f:
            if line.startswith('void '):
                list.append(line.split('(')[0].split(' ')[1])

    # get fortran functions
    with open(path.join(path.dirname(__file__), '../ITK_POM2K/pom2k.F')) as f:
        for line in f:
            # starts with following regexp: \s*SUBROUTINE\s*
            if line.startswith('      SUBROUTINE '):
                fnName = line.split('(')[0]
                # replace  \s*SUBROUTINE\s*
                fnName = fnName.replace('      SUBROUTINE ', '')
                # replace all whitespaces
                fnName = fnName.replace(' ', '')
                fnName = fnName.strip()
                fnName = f'ext_{fnName}_'
                list2.append(fnName)
                
                
diff = set(list2) - set(list)

print('Functions in pom2k.F but not in pom2k_fun.c:')
for fn in diff:
    print(fn)