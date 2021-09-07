from simulation import Result
import os


if __name__ == '__main__':
    path = 'C:/Users/macie/Desktop/Programowanie/Python/NewConceptOfGRU/simulation/result'
    files = os.listdir(path)
    for file in files:
        if file.endswith('.json'):
            file = file.replace('.json', '')
            print(file)
            result = Result(problem_name=file, structure={}, number_of_train_sample=0, simulation_number=0)
            result.load(os.path.join(path, file))
            result.plot()
