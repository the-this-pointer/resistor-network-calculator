import random
import sys
import time
import types
from contextlib import redirect_stdout
from prettytable import PrettyTable, ALL, SINGLE_BORDER
from tqdm import tqdm

E6  = 1
E12 = 2
E24 = 3

ResistorType = types.SimpleNamespace()
ResistorType.E6 = 1
ResistorType.E12 = 2
ResistorType.E24 = 3

original_stdout = sys.stdout
vRef = 5.0
adcRef = 1024
e6Values = [1000, 1500, 2200, 3300, 4700, 6800, 
            10000, 15000, 22000]
e12Values = [1000, 1200, 1500, 1800, 2200, 2700, 3300, 3900, 4700, 5600, 6800, 8200, 
             10000, 12000, 15000, 18000, 22000, 27000]
e24Values = [1000, 1100, 1200, 1300, 1500, 1600, 1800, 2000, 2200, 2400, 2700, 3000, 3300, 3600, 3900, 4300, 4700, 5100, 5600, 6200, 6800, 7500, 8200, 9100,
             10000, 11000, 12000, 13000, 15000, 16000, 18000, 20000, 22000, 24000, 27000]
best_score = None

def to_adc_value(gapV):
    return adcRef * (gapV / vRef)

def draw_title(title):
    bordered_title = f"+{'-' * (len(title) + 2)}+"
    title_line = f"| {title} |"
    print(bordered_title)
    print(title_line)
    print(bordered_title)

def find_index(lst, item):
    try:
        return lst.index(item)
    except ValueError:
        return -1
    
def append_index_to_repeats(input_list):
    # Dictionary to keep track of occurrences
    occurrences = {}
    result = []

    for item in input_list:
        if item in occurrences:
            # Increment the count for the item
            occurrences[item] += 1
            # Append the item with its current index
            result.append(f"{item}_{occurrences[item]}")
        else:
            # First occurrence of the item
            occurrences[item] = 1
            result.append(item)

    return result

def draw(n, m, subset_n, subset_m):
    table = PrettyTable()
    table.field_names = []

    cols = [str(res) for res in subset_m]
    cols.insert(1, " ")
    cols.reverse()
    cols.insert(0, "sub_n / sub_m")
    cols = append_index_to_repeats(cols)
    table.field_names = cols

    rev_n = subset_n.copy()
    rev_n.reverse()

    rev_m = subset_m[1:].copy()

    # array to keep track of voltage changes
    voltages = []
    
    voltageRows = []
    for j in range(n+1):
        voltageRow = []
        for i in range(len(rev_m)+1):
            res_sum = (0 if j == 0 else sum(rev_n[0:j])) + (0 if i == 0 else sum(rev_m[0:i]))
            voltage = vRef * (res_sum / (subset_m[0] + res_sum))
            voltages.append(voltage)
            voltageRow.append(voltage)
        voltageRows.append(voltageRow)

    minGap = float('inf')
    voltages.sort()
    rows = []
    for j in range(n+1):
        row = [' ']
        for i in range(len(rev_m)+1):
            voltage = voltageRows[j][i]
            if type(voltage) == type(str):
                row.append(voltage)

            gap = -1
            idx = find_index(voltages, voltage)
            if idx > 0:
                gap = voltages[idx] - voltages[idx - 1]
                if gap < minGap:
                    minGap = gap

            info = f"VLT: {round(voltage,3)}\nADC: {round(to_adc_value(voltage), 1)}\nGAP: {'-' if gap == -1 else round(to_adc_value(gap), 1)}"
            row.append(info) # ({res_sum}, {'[]' if j == 0 else rev_n[0:j]}, {'[]' if i == 0 else rev_m[0:i]})
        row.append(' ' if j == 0 else str(rev_n[j-1]))
        row.reverse()
        rows.append(row)

    rows.reverse()
    table.hrules=ALL
    table.set_style(SINGLE_BORDER)
    table.add_rows(rows)
    print(table)
    print(f"### Minimum ADC gap is {to_adc_value(minGap)}")

def examine(n, m, subset_n, subset_m):
    # return maximum gap if we found mutual elements in the lists (excluding `a` resistor)
    if bool(set(subset_n) & set(subset_m[1:])):
        return float('inf')

    # print(f"Examine, m: {subset_m}, n: {subset_n}")
    
    rev_n = subset_n.copy()
    rev_n.reverse()
    rev_m = subset_m[1:].copy()
    voltages = []
    # m[0] is the `a` resistor
    for j in range(n+1):
        for i in range(len(rev_m)+1):
            res_sum = (0 if j == 0 else sum(rev_n[0:j])) + (0 if i == 0 else sum(rev_m[0:i]))
            voltage = vRef * (res_sum / (subset_m[0] + res_sum))
            # print(f"voltage: {voltage}, a: {subset_m[0]}, ({res_sum}, {'[]' if j == 0 else rev_n[0:j]}, {'[]' if i == 0 else rev_m[0:i]})")
            voltages.append(voltage)
    voltages.sort()
    minGapV = float('inf')
    for i in range(1, len(voltages)):
        gap = abs(voltages[i] - voltages[i - 1])
        if gap < minGapV:
            minGapV = gap
    return minGapV

    # # can we just use the minimum resistor diffs as scores?
    # minDiff = float('inf')
    # for res_j in subset_n:
    #     for res_i in subset_m[1:]:
    #         diff = abs(res_j - res_i)
    #         if diff < minDiff:
    #             minDiff = diff
    # return minDiff

def select_subset(values, size):
    return [random.choice(values) for _ in range(size)]

def mutate(subset, values, mutation_rate=0.1):
    for i in range(len(subset)):
        if random.random() < mutation_rate:
            subset[i] = random.choice(values)
    return subset

def crossover(subset_n, subset_m, n, crossover_rate=0.6):
    n = min(n, len(subset_n), len(subset_m))
    
    if random.random() >= crossover_rate:
        return subset_n, subset_m

    # Swap the first n elements
    subset_n[:n], subset_m[:n] = subset_m[:n], subset_n[:n]
    
    return subset_n, subset_m

def compare_old(list1, list2):
    if len(list1) != len(list2):
        return False
    return all(x == y for x, y in zip(list1, list2))

def compare(list1, list2):
    if not list1 and not list2:
        return True  # Both lists are empty, consider them identical
    if not list1 or not list2:
        return False  # One list is empty, the other isn't

    # Get the length of the longer list
    max_len = max(len(list1), len(list2))

    # Calculate the similarity score
    score = sum(1 for i in range(max_len) 
                if i < len(list1) and i < len(list2) and list1[i] == list2[i])

    return (score / max_len) > 0.85

def populate_data(n, m, population_size, population, resistor_values):
    while len(population) < population_size:
        subset_n = select_subset(resistor_values, n)
        subset_m = select_subset(resistor_values, m)

        if len(population) == 0:
            population.append((subset_n, subset_m))
            continue

        found = False
        for pn, pm in population:
            if compare(pn, subset_n) and compare(pm, subset_m):
                found = True
                break
        if not found:
            population.append((subset_n, subset_m))

def calculate(n, m, resistor_type=ResistorType.E6, auto_stop=True, generations=100000, elite_size=2, population_size=5, mutation_rate=0.1, crossover_rate=0.6):
    global best_score
    global E6, E12, E24
    # Initialize the population
    population = []
    new_population = []
    n = n-1

    resistor_values = e6Values
    match resistor_type:
        case ResistorType.E12:
            resistor_values = e12Values
        case ResistorType.E24:
            resistor_values = e24Values

    # Create initial population with two subsets
    populate_data(n, m, population_size, new_population, resistor_values)

    for generation in tqdm(range(generations)):
        random.seed()
        # Update the population for the next generation
        population = new_population

        # Evaluate fitness of the population
        scores = [(examine(n, m, subset_n, subset_m), (subset_n, subset_m)) for subset_n, subset_m in population]
        
        # Sort the population based on fitness scores (larger is better)
        scores.sort(reverse=True, key=lambda x: -1 if x[0] == float('inf') else x[0])  # Sort by score

        # Select the best individuals (elitism)
        new_population = [individual for score, individual in scores[:elite_size] if score != float('inf')]

        # Create new individuals through mutation
        while len(new_population) < (population_size * 0.5):  # Maintain population size
            parent_n, parent_m = random.choice(scores)[1]
            
            # Create copies of the original lists
            child_n = parent_n.copy()
            child_m = parent_m.copy()

            child_n, child_m = crossover(child_n, child_m, int(n / 2), crossover_rate)

            child_n = mutate(child_n, resistor_values, mutation_rate)
            child_m = mutate(child_m, resistor_values, mutation_rate)

            found = False
            for pn, pm in new_population:
                if compare(pn, child_n) and compare(pm, child_m):
                    found = True
                    break
            if not found:
                new_population.append((child_n, child_m))

        populate_data(n, m, population_size, new_population, resistor_values)

        # Print scores for the current generation
        if (generation + 1) % (0.1 * generations) == 0:
            print(f"=== Generation {generation + 1}: Best Score: {scores[0][0]}, Best Individual: {scores[0][1]}")
            print(f"=== Current ADC gap is {to_adc_value(scores[0][0])}")
            # draw(n, m, scores[0][1][0], scores[0][1][1])
            
            if auto_stop:
                if best_score is not None:
                    if scores[0][0] == best_score:
                        print(f"Best score is not changed for {0.1 * generations} generations. Stopping...")
                        break
                
                best_score = scores[0][0]

    # Final evaluation
    population = [individual for score, individual in scores if score != float('inf')]
    final_scores = [(examine(n, m, subset_n, subset_m), (subset_n, subset_m)) for subset_n, subset_m in population]
    final_scores.sort(reverse=True, key=lambda x: x[0])  # Sort by score

    if len(final_scores) == 0:
        draw_title(f"No solution found")
        return

    for i in range(min(len(final_scores), 3)):
        draw_title(f"Final solution: {i+1}")

        print(f"=============== Final solution: {i}", file=original_stdout)
        score = final_scores[i]
        print(f"### Final Best Individual: {score[1]}, Score: {score[0]}")
        print(f"### Final Best Individual: {score[1]}, Score: {score[0]}", file=original_stdout)
        print(f"### Final ADC gap is {to_adc_value(score[0])}", file=original_stdout)
        draw(n, m, score[1][0], score[1][1])

if __name__ == '__main__':
    with open('output.txt', 'w', encoding="utf-8") as f:
        sys.stdout = f
        start_time = time.time()
        
        # calculate the optimal solution for a n x m resistor network
        calculate(4, 5, ResistorType.E12, auto_stop=True, 
                  generations=100000, elite_size=2, population_size=20, mutation_rate=0.3, crossover_rate=0.6)
        
        # calculate the gap and adc values for a given n x m resistor network
        # n from top to bottom, m from right to left
        # don't forget the -1 for the n
        # draw(4 - 1, 5, [18000, 1800, 1800], [18000, 2700, 2700, 4700, 5600])
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time} seconds")
        print(f"Elapsed time: {elapsed_time} seconds", file=original_stdout)
