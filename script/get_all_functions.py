

from os import path


if __name__ == '__main__':
    with open(path.join(path.dirname(__file__), '../ITK_POM2K/pom2k_fun.c')) as f:
        for line in f:
            if line.startswith('void '):
                print(line.split('(')[0].split(' ')[1])
