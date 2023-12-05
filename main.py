import random
from deap import base, creator, tools, algorithms
import json_search as j

hourly_assignments = {day: {shift_num: {hour: None for hour in hours} for shift_num, hours in shifts.items()} for
                      day, shifts in j.schedule.items()}

EMPLOYEES = ['Alice', 'Bob', 'Christine', 'David', 'Eve']

# Employee availability
employee_availability = {
    'Alice': j.alice_calendar,
    'Bob': j.bob_calendar,
    'Christine': j.christine_calendar,
    'David': j.david_calendar,
    'Eve': j.eve_calendar
}

# Define the fitness and individual
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)


def is_available(employee, day, hour):
    return employee_availability[employee].get(day, {}).get(hour, "Free") == "Free"


def create_individual():
    global hourly_assignments
    # Reset the tracker for a new individual
    for day in j.schedule.keys():
        for shift_num in j.schedule[day].keys():
            for hour in j.schedule[day][shift_num].keys():
                hourly_assignments[day][shift_num][hour] = None

    individual = []
    for day, shifts in j.schedule.items():
        for shift_num, hours in shifts.items():
            for hour in hours.keys():
                assigned = [hourly_assignments[day][sn][hour] for sn in j.schedule[day] if hour in j.schedule[day][sn]]
                candidates = [emp for emp in EMPLOYEES if is_available(emp, day, hour) and emp not in assigned]
                chosen_employee = random.choice(candidates + ["Empty"])
                individual.append(chosen_employee)
                if chosen_employee != "Empty":
                    hourly_assignments[day][shift_num][hour] = chosen_employee
    return individual


toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def get_day_hour_from_index(index):
    cumulative_hours = 0
    for day, shifts in j.schedule.items():
        for shift_num, hours in shifts.items():
            for hour in hours.keys():
                if index == cumulative_hours:
                    return day, hour
                cumulative_hours += 1


def custom_mutate(individual):
    global hourly_assignments
    index = 0
    for day, shifts in j.schedule.items():
        for shift_num, hours in shifts.items():
            for hour in hours.keys():
                if individual[index] == "Empty":
                    candidates = [emp for emp in EMPLOYEES if
                                  is_available(emp, day, hour) and hourly_assignments[day][shift_num][hour] != emp]
                    if candidates:
                        individual[index] = random.choice(candidates)
                        if individual[index] != "Empty":
                            hourly_assignments[day][shift_num][hour] = individual[index]
                index += 1
    return individual,


def evaluate(individual):
    contiguous_hours_score = 0
    shift_length_penalty = 0
    last_employee = None
    contiguous_count = 0

    for i, gene in enumerate(individual):
        if gene == last_employee and gene != "Empty":
            contiguous_count += 1
        else:
            if last_employee and (contiguous_count < j.employees[last_employee]['min_shifl_len'] or
                                  contiguous_count > j.employees[last_employee]['max_shift_len']):
                shift_length_penalty -= 5
            contiguous_count = 0

        last_employee = gene if gene != "Empty" else None

    coverage_score = sum(1 for gene in individual if gene != "Empty")
    return coverage_score + contiguous_hours_score + shift_length_penalty,


toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)  # Example crossover
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mutate", custom_mutate)


def print_schedule(best_schedule):
    detailed_schedule = {day: {} for day in j.schedule.keys()}
    i = 0
    for day, shifts in j.schedule.items():
        for shift_num, hours in shifts.items():
            for hour in hours.keys():
                if shift_num not in detailed_schedule[day]:
                    detailed_schedule[day][shift_num] = {}
                detailed_schedule[day][shift_num][hour] = best_schedule[i]
                i += 1

    for day, shifts in detailed_schedule.items():
        print(f"Day: {day}")
        for shift_num, hours in shifts.items():
            print(f"  Shift {shift_num}:")
            for hour, employee in hours.items():
                print(f"    Hour {hour}: {employee}")
        print("\n")


def main():
    population = toolbox.population(n=100)
    ngen = 50
    for gen in range(ngen):
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.1)
        fits = toolbox.map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        population = toolbox.select(offspring, k=len(population))
    return tools.selBest(population, k=1)[0]


if __name__ == "__main__":
    best_schedule = main()
    total_slots = sum(len(hours) for shifts in j.schedule.values() for hours in shifts.values())
    if len(best_schedule) == total_slots:
        print_schedule(best_schedule)
    else:
        print("Error: The size of the best schedule does not match the total number of slots.")
