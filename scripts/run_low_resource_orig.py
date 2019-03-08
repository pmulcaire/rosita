import subprocess
import sys

def main(target, source):
    sizes = [100, 500, 1000, 0] 
    for size in sizes:
        config = 'ud_{}-{}-{}_elmo-{}-{}'.format(target, size, source, target, source)
        command = 'allennlp train data/configs/low_resource/{}.json -s data/low_resource/{}_1/'.format(config, config)
        print(command)
        subprocess.check_call(command, shell=True)
    
if __name__ == '__main__':
    target = sys.argv[1]
    source = sys.argv[2]
    main(target, source)
