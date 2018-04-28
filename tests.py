import subprocess
import os


RED="\033[0;31m"
GREEN="\033[0;32m"
BLANK_COLOR="\033[0;m"


# Move to tests
def check_equal(arr1, arr2):
    for row1, row2 in zip(arr1, arr2):
        for cell1, cell2 in zip(row1, row2):
            assert abs(cell1 - cell2) < 0.000001


def test_one(program, test_path, test_name):
    with open(test_path + '.out', 'r') as f:
        test_out = f.read()
    nn_file = test_path + '.nn'
    in_file = test_path + '.in'
    try:
        result = subprocess.check_output(program + (nn_file,) + (in_file,))
    except Exception as e:
        print RED + str(e) + BLANK_COLOR
        return False
    if test_out != result:
        print('Test: ' + RED + test_name + BLANK_COLOR + ' failed!')
        with open(test_path + '.result', 'w') as f:
            f.write(result)
        return False
    else:
        print('Test: ' + GREEN + test_name + BLANK_COLOR + ' passed!')
        return True


def run_tests(program):
    for subdir, dirs, files in os.walk('tests'):
        tests = [name[:-3] for name in files if name.endswith('.nn')]
        passed = 0
        for test in tests:
            if test_one(program, os.path.join(subdir, test), test):
                passed += 1
        tests_number = len(tests)
        if passed == tests_number:
            print(GREEN + 'Passed ' + str(passed) + '/' + str(tests_number)
                    + ' tests' + BLANK_COLOR)
        else:
            print(RED + 'Passed ' + str(passed) + '/' + str(tests_number)
                    + ' tests' + BLANK_COLOR)


run_tests(('python', 'run_neural_network.py'))
