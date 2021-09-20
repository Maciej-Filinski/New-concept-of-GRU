from simulation import Result
import os


if __name__ == '__main__':
    path = os.path.abspath('../New-concept-of-GRU/simulation/result')
    files = os.listdir(path)
    files = [file.replace('.json', '') for file in files if file.endswith('.json')]
    files = sorted(files)
    for file in files:
        if file.startswith('toy_problem_v3') and file.endswith('00'):
            result = Result(problem_name=file, structure={}, number_of_train_sample=0, simulation_number=0)
            result.load(os.path.join(path, file))
            result.plot()
