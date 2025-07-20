import argparse

def calculate_abs_avg(task1_score, task2_score):
    return (task1_score + task2_score) / 2


def calculate_norm_avg(task1_score, task2_score, task1_individual, task2_individual):
    # Normalize each task score using individual fine-tuning score as 100%
    task1_normalized = (task1_score / task1_individual) * 100
    task2_normalized = (task2_score / task2_individual) * 100

    # Calculate average of normalized scores
    return (task1_normalized + task2_normalized) / 2

def main():
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description='Calculate absolute and normalized averages for merged model performance')
    parser.add_argument('merged_task1', type=float, help='Merged model performance score on task 1')
    parser.add_argument('merged_task2', type=float, help='Merged model performance score on task 2')
    parser.add_argument('individual_task1', type=float, help='Individual fine-tuning score on task 1')
    parser.add_argument('individual_task2', type=float, help='Individual fine-tuning score on task 2')

    # Parse arguments
    args = parser.parse_args()

    # Calculate absolute average
    abs_avg = calculate_abs_avg(args.merged_task1, args.merged_task2)
    print(f"Absolute Average: {abs_avg:.2f}")

    # Calculate normalized average
    norm_avg = calculate_norm_avg(args.merged_task1, args.merged_task2, args.individual_task1, args.individual_task2)
    print(f"Normalized Average: {norm_avg:.2f}")


if __name__ == "__main__":
    main()